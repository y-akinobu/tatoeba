import torch
import torch.nn as nn
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from torch import Tensor
from typing import Iterable, List
import math
import re
import numpy as np
from janome.tokenizer import Tokenizer
from google_drive_downloader import GoogleDriveDownloader

GoogleDriveDownloader.download_file_from_google_drive(file_id="1zMTrsmcyF2oXpWKe0bIZ7Ej1JBjVq7np",dest_path="./model_DS.pt", unzip=False)
GoogleDriveDownloader.download_file_from_google_drive(file_id="13C39jfdkkmE2mx-1K9PFXqGST84j-mz8",dest_path="./vocab_obj_DS.pth", unzip=False)

# デバイスの指定
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('DEVICE :', DEVICE)

# SRC (source) : 原文
SRC_LANGUAGE = 'jpn'
# TGT (target) : 訳文
TGT_LANGUAGE = 'py'

# special_token IDX
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

tokenizer = Tokenizer("tatoeba/user_simpledic.csv", udic_type="simpledic", udic_enc="utf8", wakati=True)

def jpn_tokenizer(text):
  return [token for token in tokenizer.tokenize(text) if token != " " and len(token) != 0]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int,
                 emb_size: int, 
                 nhead: int, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, 
                src: Tensor, 
                tgt: Tensor, 
                src_mask: Tensor,
                tgt_mask: Tensor, 
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, 
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
        
class PositionalEncoding(nn.Module):
    def __init__(self, 
                 emb_size: int, 
                 dropout: float, 
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0),:])
        
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# モデルが予測を行う際に、未来の単語を見ないようにするためのマスク
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

def beam_topk(model, ys, memory, beamsize):
    ys = ys.to(DEVICE)

    tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
    out = model.decode(ys, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    next_prob, next_word = prob.topk(k=beamsize, dim=1)
    
    return next_prob, next_word

# greedy search を使って翻訳結果 (シーケンス) を生成
def beam_decode(model, src, src_mask, max_len, beamsize, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    ys_result = {}

    memory = model.encode(src, src_mask).to(DEVICE)   # encode の出力 (コンテキストベクトル)

    # 初期値 (beamsize)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    next_prob, next_word = beam_topk(model, ys, memory, beamsize)
    next_prob = next_prob[0].tolist()

    # <sos> + 1文字目 の候補 (list の長さはbeamsizeの数)
    ys = [torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word[:, idx].item())], dim=0) for idx in range(beamsize)]

    for i in range(max_len-1):
        prob_list = []
        ys_list = []

        # それぞれの候補ごとに次の予測トークンとその確率を計算
        for ys_token in ys:
            next_prob, next_word = beam_topk(model, ys_token, memory, len(ys))

            # 予測確率をリスト (next_prob) に代入
            next_prob = next_prob[0].tolist()
            # 1つのリストに結合
            prob_list.extend(next_prob)

            ys = [torch.cat([ys_token, torch.ones(1, 1).type_as(src.data).fill_(next_word[:, idx].item())], dim=0) for idx in range(len(ys))]
            ys_list.extend(ys)

        # prob_list の topk のインデックスを prob_topk_idx で保持
        prob_topk_idx = list(reversed(np.argsort(prob_list).tolist()))
        prob_topk_idx = prob_topk_idx[:len(ys)]
        # print('@@', prob_topk_idx)

        # ys に新たな topk 候補を代入
        ys = [ys_list[idx] for idx in prob_topk_idx]

        next_prob = [prob_list[idx] for idx in prob_topk_idx]
        # print('@@orig', prob_list)
        # print('@@next', next_prob)

        pop_list = []
        for j in range(len(ys)):
            # EOS トークンが末尾にあったら、ys_result (返り値) に append
            if ys[j][-1].item() == EOS_IDX:
                ys_result[ys[j]] = next_prob[j]
                pop_list.append(j)

        # ys_result に一度入ったら、もとの ys からは抜いておく
        # (ys の長さが変わるので、ところどころbeamsize ではなく len(ys) を使用している箇所がある)
        for l in sorted(pop_list, reverse=True):
            del ys[l]

        # ys_result が beamsize よりも大きかった時に、処理を終える
        if len(ys_result) >= beamsize:
            break

    return ys_result

class NMT(object):
  vocab: object

  def __init__(self, vocab_file):
    self.vocab = torch.load(vocab_file)
    self.SRC_VOCAB_SIZE = len(self.vocab[SRC_LANGUAGE])
    self.TGT_VOCAB_SIZE = len(self.vocab[TGT_LANGUAGE])
    self.src_transform = sequential_transforms(jpn_tokenizer, #Tokenization
                                               self.vocab[SRC_LANGUAGE], #Numericalization
                                               tensor_transform) # Add SOS/EOS and create tensor
    self.EMB_SIZE = 512
    self.NHEAD = 8
    self.FFN_HID_DIM = 512
    self.BATCH_SIZE = 128
    self.NUM_ENCODER_LAYERS = 3
    self.NUM_DECODER_LAYERS = 3

    self.transformer = Seq2SeqTransformer(self.NUM_ENCODER_LAYERS, self.NUM_DECODER_LAYERS, 
                                 self.EMB_SIZE, self.NHEAD, self.SRC_VOCAB_SIZE, self.TGT_VOCAB_SIZE,
                                 self.FFN_HID_DIM)
    
    for p in self.transformer.parameters():
      if p.dim() > 1:
          nn.init.xavier_uniform_(p)

    self.transformer = self.transformer.to(DEVICE)

  def load(self, trained_model):
    self.transformer.load_state_dict(torch.load(trained_model))

  def translate_beam(self, src_sentence: str, beamsize=5):
        """
        複数の翻訳候補をリストで返す。
        """
        pred_list = []
        self.transformer.eval()
        src = self.src_transform(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = beam_decode(
            self.transformer,  src, src_mask, max_len=num_tokens + 5, beamsize=beamsize, start_symbol=SOS_IDX)
        prob_list = list(tgt_tokens.values())
        tgt_tokens = list(tgt_tokens.keys())
        for idx in list(reversed(np.argsort(prob_list).tolist())):
            pred_list.append(" ".join(self.vocab[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens[idx].cpu().numpy()))).replace("<sos>", "").replace("<eos>", ""))
        return pred_list, sorted(prob_list, reverse=True)

def PyNMT(trained_model, vocab):
    nmt = NMT(vocab)
    nmt.load(trained_model)
    return nmt

nmt = PyNMT('model_DS.pt', 'vocab_obj_DS.pth')

special_token = ['<A>', '<B>', '<C>', '<D>', '<E>']

def generate(sentence):
  candidate = re.findall(r'[a-zA-Z"\']+', sentence)
  for idx in range(len(candidate)):
    sentence = sentence.replace(candidate[idx], special_token[idx])
  print(sentence)
  pred, prob = nmt.translate_beam(sentence)
  print(pred)
  print(prob)

sentence = 'dfの先頭を表示する'
generate(sentence)
