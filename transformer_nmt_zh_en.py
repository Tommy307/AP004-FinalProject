# import nltk
# nltk.download('punkt')

import json
import math
import random
import re
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import datetime

LOG_PATH = "./log.txt"

def write_log(content, log_path="./log.txt", add_timestamp=True):
    """
    功能：将字符串追加到文件末尾
    
    参数:
        content (str): 要写入的内容
        log_path (str): 文件路径
        add_timestamp (bool): 是否自动在行首添加当前时间
    """
    # 1. 自动创建父目录 (如果目录不存在)
    directory = os.path.dirname(log_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    # 2. 准备写入的内容
    final_content = str(content)
    if add_timestamp:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_content = f"[{now}] {final_content}"
        
    # 3. 写入文件
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(final_content + '\n')
    except Exception as e:
        write_log(f"写入日志失败: {e}", log_path=LOG_PATH)


# ========== NLTK (BLEU + 英文 tokenizer) ==========
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate import bleu_score
    _HAS_NLTK = True
    write_log("Use NLTK", log_path=LOG_PATH)
except Exception:
    _HAS_NLTK = False

# ========== Jieba (中文分词) ==========
try:
    import jieba
    write_log("Use Jieba", log_path=LOG_PATH)
except ImportError:
    jieba = None

# ========== HuggingFace Transformers (T5 微调用) ==========
try:
    from transformers import (
        T5ForConditionalGeneration,
        T5TokenizerFast,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


# ===================== 文本清洗 & 分词 =====================

def clean_text(text: str, lang: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if lang == "en":
        text = re.sub(r"[^a-zA-Z0-9\s\.,;:!\?'\"-\(\)]", "", text)
        text = text.lower()
    else:  # zh
        text = re.sub(
            r"[^\u4e00-\u9fff0-9a-zA-Z\s，。、《》？：“”；：（）！…—\-]",
            "",
            text,
        )
    return text


def tokenize_en(text: str) -> List[str]:
    text = clean_text(text, lang="en")
    if not text:
        return []
    if _HAS_NLTK:
        try:
            return [t for t in word_tokenize(text) if t.strip()]
        except LookupError:
            pass
    tokens = []
    for w in text.split():
        parts = re.findall(r"\w+|[^\w\s]", w)
        tokens.extend([p for p in parts if p])
    return tokens


def tokenize_zh(text: str) -> List[str]:
    text = clean_text(text, lang="zh")
    if not text:
        return []
    if jieba is not None:
        return [t.strip() for t in jieba.lcut(text) if t.strip()]
    else:
        return [ch for ch in text if ch.strip()]


# ===================== 词表 =====================

class Vocab:
    def __init__(self, min_freq: int = 1, specials=None):
        self.freqs: Dict[str, int] = {}
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        if specials is None:
            specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.specials = specials
        for sp in specials:
            self.add_token(sp)
        self.min_freq = min_freq

    def add_token(self, tok: str):
        self.freqs[tok] = self.freqs.get(tok, 0) + 1

    def build_vocab(self):
        self.itos = []
        self.stoi = {}
        for sp in self.specials:
            self._insert(sp)
        for tok, f in self.freqs.items():
            if tok in self.stoi:
                continue
            if f >= self.min_freq:
                self._insert(tok)

    def _insert(self, tok: str):
        idx = len(self.itos)
        self.itos.append(tok)
        self.stoi[tok] = idx

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.stoi[UNK_TOKEN]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)


# ===================== NMT 数据集：中文->英文 =====================

class NMTDataset(Dataset):
    """
    jsonl: {"en": "...", "zh": "...", "index": ...}
    这里做：清洗 + 分词 + 长度过滤 + 构建词表（训练集）
    """
    def __init__(
        self,
        path: str,
        src_lang: str = "zh",
        tgt_lang: str = "en",
        src_vocab: Vocab = None,
        tgt_vocab: Vocab = None,
        build_vocab: bool = True,
        min_freq: int = 1,
        max_samples: int = None,
        max_src_len: int = 80,
        max_tgt_len: int = 80,
        truncate_long: bool = True,
        max_vocab_size: int = 1000          ## 固定缩减 vocab 到 1k
    ):
        self.samples = []   # (src_tokens, tgt_tokens)

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._src_tok = tokenize_zh if src_lang == "zh" else tokenize_en
        self._tgt_tok = tokenize_en if tgt_lang == "en" else tokenize_zh

        if src_vocab is None:
            self.src_vocab = Vocab(min_freq=min_freq)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = Vocab(min_freq=min_freq)
        else:
            self.tgt_vocab = tgt_vocab

        dropped_empty = 0
        dropped_long = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                raw_src = obj[src_lang]
                raw_tgt = obj[tgt_lang]
                src_tokens = self._src_tok(raw_src)
                tgt_tokens = self._tgt_tok(raw_tgt)

                if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                    dropped_empty += 1
                    continue

                def handle_len(tokens, max_len):
                    nonlocal dropped_long
                    if len(tokens) > max_len:
                        if truncate_long:
                            return tokens[:max_len]
                        else:
                            dropped_long += 1
                            return None
                    return tokens

                src_tokens = handle_len(src_tokens, max_src_len)
                tgt_tokens = handle_len(tgt_tokens, max_tgt_len)
                if src_tokens is None or tgt_tokens is None:
                    continue

                self.samples.append((src_tokens, tgt_tokens))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        write_log(f"[Dataset] Loaded {len(self.samples)} pairs from {path}", log_path=LOG_PATH)
        write_log(f"[Dataset] Dropped empty={dropped_empty}, dropped_long={dropped_long}", log_path=LOG_PATH)

        if build_vocab:
            for s_tok, t_tok in self.samples:
                for tok in s_tok:
                    self.src_vocab.add_token(tok)
                for tok in t_tok:
                    self.tgt_vocab.add_token(tok)
            self.src_vocab.build_vocab()
            self.tgt_vocab.build_vocab()

            # 限制词表大小
            # if max_vocab_size is not None:
            #     self._shrink_vocab(self.src_vocab, max_vocab_size, name="src")
            #     self._shrink_vocab(self.tgt_vocab, max_vocab_size, name="tgt")

            write_log(f"[Vocab] src size={len(self.src_vocab)}, tgt size={len(self.tgt_vocab)}", log_path=LOG_PATH)

    @staticmethod
    def _shrink_vocab(vocab: Vocab, max_size: int, name="vocab"):
        """根据频率削减词表"""
        if len(vocab) <= max_size:
            return

        specials = vocab.specials
        items = [(tok, freq) for tok, freq in vocab.freqs.items()
                 if tok not in specials]
        items.sort(key=lambda x: x[1], reverse=True)
        keep_tokens = [tok for tok, _ in items[: max_size - len(specials)]]

        vocab.itos = []
        vocab.stoi = {}
        for sp in specials:
            vocab._insert(sp)
        for tok in keep_tokens:
            vocab._insert(tok)

        write_log(f"[Vocab] {name} vocab shrunk to {len(vocab)}", log_path=LOG_PATH)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.samples[idx]
        src_tokens = [BOS_TOKEN] + src_tokens + [EOS_TOKEN]
        tgt_tokens = [BOS_TOKEN] + tgt_tokens + [EOS_TOKEN]
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    pad_idx = 0  # 假设 PAD=0
    B = len(batch)
    src_batch = torch.full((B, max_src), pad_idx, dtype=torch.long)
    tgt_batch = torch.full((B, max_tgt), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_batch[i, : len(s)] = s
        tgt_batch[i, : len(t)] = t
    return src_batch, torch.tensor(src_lens), tgt_batch, torch.tensor(tgt_lens)


# ===================== BLEU-4 评估（使用 greedy 解码） =====================

def compute_bleu_corpus(references, hypotheses):
    if not _HAS_NLTK:
        raise RuntimeError("需要 NLTK 计算 BLEU：`pip install nltk` 并下载 punkt。")
    smoothie = bleu_score.SmoothingFunction().method1
    return bleu_score.corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    ) * 100.0


# ===================== RMSNorm & Position Encoding & RelativeBias =====================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., dim]
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class PositionalEncoding(nn.Module):
    """
    绝对位置编码（正弦）
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class RelativePositionBias(nn.Module):
    """
    T5-style relative position bias:
    bias_table: [num_buckets, n_heads]
    这里简化为：按 (i-j) 映射到 [0, 2*max_distance-1]
    """
    def __init__(self, num_heads, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.num_buckets = 2 * max_distance - 1
        self.bias = nn.Embedding(self.num_buckets, num_heads)

    def forward(self, qlen, klen):
        # 只用于 self-attn（qlen == klen）
        device = self.bias.weight.device
        context_pos = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        memory_pos = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        # relative position: i-j
        rel_pos = memory_pos - context_pos  # [qlen, klen]
        rel_pos = rel_pos + (self.max_distance - 1)
        rel_pos = torch.clamp(rel_pos, 0, self.num_buckets - 1)
        # [qlen, klen, n_heads]
        values = self.bias(rel_pos)
        return values.permute(2, 0, 1)  # [n_heads, qlen, klen]


# ===================== Multi-Head Attention =====================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1,
                 pos_type="absolute", use_bias=True, max_relative_distance=128):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.pos_type = pos_type

        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        if pos_type == "relative":
            self.rel_bias = RelativePositionBias(
                num_heads=n_heads,
                max_distance=max_relative_distance,
            )
        else:
            self.rel_bias = None

    def forward(self, query, key, value, mask=None, is_self_attn=False):
        # query/key/value: [B, T, D]
        B, Tq, _ = query.size()
        B, Tk, _ = key.size()

        Q = self.W_q(query).view(B, Tq, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, Tq, d_k]
        K = self.W_k(key).view(B, Tk, self.n_heads, self.d_k).transpose(1, 2)   # [B, H, Tk, d_k]
        V = self.W_v(value).view(B, Tk, self.n_heads, self.d_k).transpose(1, 2) # [B, H, Tk, d_k]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)      # [B, H, Tq, Tk]

        # relative position bias 仅用于 self-attn（encoder/decoder 自注意力）
        if self.pos_type == "relative" and is_self_attn and (Tq == Tk):
            bias = self.rel_bias(Tq, Tk)  # [H, Tq, Tk]
            scores = scores + bias.unsqueeze(0)

        if mask is not None:
            # mask: [B, 1, 1, Tk] or [B, 1, Tq, Tk]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # [B, H, Tq, d_k]
        context = context.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        output = self.W_o(context)
        return output


# ===================== Transformer Encoder / Decoder =====================

def get_norm(norm_type, dim):
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim)
    else:
        raise ValueError(f"Unknown norm_type {norm_type}")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 norm_type="layernorm", pos_type="absolute"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout,
                                            pos_type=pos_type)
        self.norm1 = get_norm(norm_type, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = get_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # x: [B, T, D]
        attn_out = self.self_attn(x, x, x, mask=src_mask, is_self_attn=True)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x

    ## gemini
    # def forward(self, x, src_mask=None):
    #     # Pre-Norm 结构
    #     # 1. Self-Attention
    #     residual = x
    #     x = self.norm1(x)
    #     x = self.self_attn(x, x, x, mask=src_mask, is_self_attn=True)
    #     x = residual + self.dropout(x)

    #     # 2. Feed Forward
    #     residual = x
    #     x = self.norm2(x)
    #     x = self.ff(x)
    #     x = residual + self.dropout(x)
    #     return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 norm_type="layernorm", pos_type="absolute"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout,
                                            pos_type=pos_type)
        self.norm1 = get_norm(norm_type, d_model)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout,
                                             pos_type="absolute")  # cross-attn 不用相对位置
        self.norm2 = get_norm(norm_type, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = get_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # self-attn
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask, is_self_attn=True)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)

        # cross-attn
        cross_attn_out = self.cross_attn(x, memory, memory, mask=memory_mask, is_self_attn=False)
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff,
                 dropout=0.1, pad_idx=0, pos_type="absolute", norm_type="layernorm",
                 max_len=512):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_type = pos_type
        if pos_type == "absolute":
            self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        else:
            self.pos_enc = None
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, norm_type, pos_type)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        # src: [B, T]
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        x = self.embedding(src) * math.sqrt(self.d_model)
        if self.pos_type == "absolute":
            x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return x, mask  # x: [B,T,D], mask: [B,1,1,T]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff,
                 dropout=0.1, pad_idx=0, pos_type="absolute", norm_type="layernorm",
                 max_len=512):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_type = pos_type
        if pos_type == "absolute":
            self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        else:
            self.pos_enc = None
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, norm_type, pos_type)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, src_mask=None):
        # tgt: [B, T]
        B, T = tgt.size()
        # 目标序列 mask: padding + subsequent mask
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        subsequent_mask = torch.triu(torch.ones((T, T), device=tgt.device), diagonal=1).bool()
        subsequent_mask = ~subsequent_mask  # True=allowed
        tgt_mask = tgt_pad_mask & subsequent_mask.unsqueeze(0).unsqueeze(1)  # [B,1,T,T]

        x = self.embedding(tgt) * math.sqrt(self.d_model)
        if self.pos_type == "absolute":
            x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=src_mask)

        logits = self.out(x)  # [B,T,V]
        return logits


class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1,
                 pad_idx=0, bos_idx=1, eos_idx=2,
                 pos_type="absolute", norm_type="layernorm", max_len=512, device="cpu"):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_heads, num_layers, d_ff,
                                          dropout, pad_idx, pos_type, norm_type, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads, num_layers, d_ff,
                                          dropout, pad_idx, pos_type, norm_type, max_len)
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.device = device

    def forward(self, src, tgt):
        # teacher-forcing 训练：tgt[:,:-1] 作为输入，tgt[:,1:] 作为目标
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        memory, src_mask = self.encoder(src)
        logits = self.decoder(tgt[:, :-1], memory, src_mask=src_mask)
        return logits

    def greedy_decode(self, src, max_len=50):
        self.eval()
        with torch.no_grad():
            src = src.to(self.device)
            memory, src_mask = self.encoder(src)
            B = src.size(0)
            ys = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=self.device)
            finished = [False] * B
            outputs = [[] for _ in range(B)]

            for _ in range(max_len):
                logits = self.decoder(ys, memory, src_mask=src_mask)  # [B,T,V]
                next_logits = logits[:, -1, :]  # [B,V]
                next_tokens = next_logits.argmax(dim=-1)  # [B]
                ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)

                for i in range(B):
                    if not finished[i]:
                        tid = next_tokens[i].item()
                        if tid == self.eos_idx:
                            finished[i] = True
                        else:
                            outputs[i].append(tid)
                if all(finished):
                    break
        return outputs


# ===================== 训练 & 验证（scratch Transformer） =====================

# def train_epoch_scratch(model, dataloader, optimizer, scheduler, criterion, device):
def train_epoch_scratch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for src, src_lens, tgt, tgt_lens in dataloader:
        optimizer.zero_grad()

        src = src.to(device)
        tgt = tgt.to(device)

        logits = model(src, tgt)  # [B, T-1, V]
        target = tgt[:, 1:].contiguous()  # [B, T-1]，此时已经在 device 上

        loss = criterion(logits.view(-1, logits.size(-1)),
                         target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_bleu_scratch(model, dataloader, tgt_vocab, device, max_len=50):
    model.eval()
    references = []
    hyps = []
    pad_tok = PAD_TOKEN
    bos_tok = BOS_TOKEN
    eos_tok = EOS_TOKEN
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens in dataloader:
            preds = model.greedy_decode(src, max_len=max_len)
            B = src.size(0)
            for i in range(B):
                # hyp
                hyp_tokens = []
                for tid in preds[i]:
                    tok = tgt_vocab.itos[tid]
                    if tok in (pad_tok, bos_tok, eos_tok):
                        continue
                    hyp_tokens.append(tok)
                # ref
                gold_ids = tgt[i].tolist()
                ref_tokens = []
                for tid in gold_ids:
                    tok = tgt_vocab.itos[tid]
                    if tok == eos_tok:
                        break
                    if tok in (pad_tok, bos_tok):
                        continue
                    ref_tokens.append(tok)
                if len(ref_tokens) == 0:
                    continue
                references.append([ref_tokens])
                hyps.append(hyp_tokens)
    if len(hyps) == 0:
        return 0.0
    return compute_bleu_corpus(references, hyps)


# ===================== T5 微调（from pretrained LM） =====================

class JsonlTextDataset(Dataset):
    """
    用于 T5：直接返回原始中英句子
    """
    def __init__(self, path, max_samples=None):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.samples.append((obj["zh"], obj["en"]))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
        write_log(f"[T5Dataset] Loaded {len(self.samples)} from {path}", log_path=LOG_PATH)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def t5_collate_fn(batch, tokenizer, max_src_len=80, max_tgt_len=80):
    zh_texts, en_texts = zip(*batch)
    # 这里简单用 "translate Chinese to English: " prompt
    inputs = [f"translate Chinese to English: {t}" for t in zh_texts]
    model_inputs = tokenizer(
        inputs,
        max_length=max_src_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            list(en_texts),
            max_length=max_tgt_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )["input_ids"]
    # 将 pad token id 替换为 -100，以便 loss 忽略
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def train_t5_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_t5_bleu(model, dataloader, tokenizer, device, max_len=80):
    model.eval()
    references = []
    hyps = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # 生成
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_len,
            )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # refs
            # labels 中 pad 已经是 -100，要恢复成 tokens
            label_ids = labels.clone()
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            for ref, hyp in zip(refs, preds):
                ref_tokens = ref.strip().split()
                hyp_tokens = hyp.strip().split()
                if len(ref_tokens) == 0:
                    continue
                references.append([ref_tokens])
                hyps.append(hyp_tokens)
    if len(hyps) == 0:
        return 0.0
    return compute_bleu_corpus(references, hyps)


# ===================== 主程序 =====================

def main_scratch(args):
    LOG_PATH = args.log_path
    device = args.device
    write_log("=== Transformer NMT from scratch (zh->en) ===", log_path=LOG_PATH)
    write_log("Loading datasets ...", log_path=LOG_PATH)
    train_data = NMTDataset(
        args.data_path,
        src_lang="zh",
        tgt_lang="en",
        min_freq=args.min_freq,
        max_samples=args.max_samples,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        truncate_long=args.truncate_long,
    )
    valid_data = NMTDataset(
        args.valid_path,
        src_lang="zh",
        tgt_lang="en",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        build_vocab=False,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        truncate_long=args.truncate_long,
    )
    test_data = NMTDataset(
        args.test_path,
        src_lang="zh",
        tgt_lang="en",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        build_vocab=False,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        truncate_long=args.truncate_long,
    )

    pad_idx = train_data.src_vocab.stoi[PAD_TOKEN]
    bos_idx = train_data.tgt_vocab.stoi[BOS_TOKEN]
    eos_idx = train_data.tgt_vocab.stoi[EOS_TOKEN]

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    write_log("Building Transformer model ...", log_path=LOG_PATH)
    model = TransformerNMT(
        src_vocab_size=len(train_data.src_vocab),
        tgt_vocab_size=len(train_data.tgt_vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        pos_type=args.pos_type,
        norm_type=args.norm_type,
        max_len=max(args.max_src_len, args.max_tgt_len) + 5,
        device=device,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, 
    #     lr_lambda=lambda step: min((step + 1)**-0.5, (step + 1) * (4000**-1.5))
    # )

    best_val_bleu = -1.0

    for epoch in range(1, args.epochs + 1):
        # train_loss = train_epoch_scratch(model, train_loader, optimizer, scheduler, criterion, device)
        train_loss = train_epoch_scratch(model, train_loader, optimizer, criterion, device)
        write_log(f"[Epoch {epoch}] train loss = {train_loss:.4f}", args.log_path)

        val_bleu = evaluate_bleu_scratch(
            model, valid_loader, train_data.tgt_vocab, device, max_len=args.max_tgt_len
        )
        write_log(f"[Epoch {epoch}] valid BLEU-4 = {val_bleu:.2f}", args.log_path)

        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "src_vocab": train_data.src_vocab,
                    "tgt_vocab": train_data.tgt_vocab,
                    "pad_idx": pad_idx,
                    "bos_idx": bos_idx,
                    "eos_idx": eos_idx,
                    "config": {
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "num_layers": args.num_layers,
                        "d_ff": args.d_ff,
                        "dropout": args.dropout,
                        "pos_type": args.pos_type,
                        "norm_type": args.norm_type,
                    },
                },
                args.save_path,
            )
            write_log(f"[Checkpoint] Saved best model (BLEU={best_val_bleu:.2f}) to {args.save_path}", args.log_path)
        
    # 测试集使用 best checkpoint
    write_log("Loading best checkpoint for test ...", args.log_path)
    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_bleu = evaluate_bleu_scratch(
        model, test_loader, train_data.tgt_vocab, device, max_len=args.max_tgt_len
    )
    write_log(f"[Test] BLEU-4 = {test_bleu:.2f}", args.log_path)


def main_t5(args):
    LOG_PATH = args.log_path
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("需要 transformers 库：`pip install transformers`。")
    device = args.device
    write_log("=== Fine-tuning T5 (zh->en) ===", args.log_path)

    if not "t5-small" in args.t5_model_name:
        print("Not implemented this t5")
        exit(1)
    write_log("Loading tokenizer & model ...", args.log_path)
    local_path = "/mnt/afs/250010021/zhoulang/Homework/LLM/Final/model/t5-small" 
    tokenizer = T5TokenizerFast.from_pretrained(local_path, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(local_path, local_files_only=True)
    # tokenizer = T5TokenizerFast.from_pretrained(args.t5_model_name)
    # model = T5ForConditionalGeneration.from_pretrained(args.t5_model_name)
    model.to(device)

    train_data = JsonlTextDataset(args.data_path, max_samples=args.max_samples)
    valid_data = JsonlTextDataset(args.valid_path)
    test_data = JsonlTextDataset(args.test_path)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: t5_collate_fn(b, tokenizer, args.max_src_len, args.max_tgt_len),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: t5_collate_fn(b, tokenizer, args.max_src_len, args.max_tgt_len),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: t5_collate_fn(b, tokenizer, args.max_src_len, args.max_tgt_len),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps
    )

    best_val_bleu = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_t5_epoch(model, train_loader, optimizer, scheduler, device)
        write_log(f"[Epoch {epoch}] T5 train loss = {train_loss:.4f}", args.log_path)

        val_bleu = evaluate_t5_bleu(model, valid_loader, tokenizer, device, max_len=args.max_tgt_len)
        write_log(f"[Epoch {epoch}] T5 valid BLEU-4 = {val_bleu:.2f}", args.log_path)

        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                args.save_path,
            )
            write_log(f"[Checkpoint] Saved best T5 model (BLEU={best_val_bleu:.2f}) to {args.save_path}", args.log_path)

    write_log("Loading best T5 checkpoint for test ...", args.log_path)
    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_bleu = evaluate_t5_bleu(model, test_loader, tokenizer, device, max_len=args.max_tgt_len)
    write_log(f"[Test] T5 BLEU-4 = {test_bleu:.2f}", args.log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-based NMT zh->en")
    parser.add_argument("--mode", type=str, default="scratch", choices=["scratch", "t5"],
                        help="scratch: 自实现 Transformer; t5: 微调预训练 T5")
    parser.add_argument("--data_path", type=str, default="./data/train_100k.jsonl")
    parser.add_argument("--valid_path", type=str, default="./data/valid.jsonl")
    parser.add_argument("--test_path", type=str, default="./data/test.jsonl")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_src_len", type=int, default=80)
    parser.add_argument("--max_tgt_len", type=int, default=80)
    parser.add_argument("--truncate_long", action="store_true")

    # scratch Transformer 超参数 / ablation
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pos_type", type=str, default="absolute",
                        choices=["absolute", "relative"],
                        help="位置编码：absolute(正弦) / relative(T5-style bias)")
    parser.add_argument("--norm_type", type=str, default="layernorm",
                        choices=["layernorm", "rmsnorm"])
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--save_path", type=str,
                        default="./checkpoints/transformer_zh_en_best.pt")
    parser.add_argument("--log_path", type=str,
                        default="./checkpoints/transformer_zh_en_log.txt")

    # T5 相关
    parser.add_argument("--t5_model_name", type=str, default="t5-small",
                        help="如 t5-small / t5-base / mt5-small 等")

    args = parser.parse_args()

    if args.mode == "scratch":
        main_scratch(args)
    else:
        main_t5(args)