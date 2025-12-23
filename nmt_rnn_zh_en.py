import json
from typing import List, Dict
import random
import os
import datetime
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 中文分词库
try:
    import jieba
    print("✓ Using [jieba]")
except ImportError:
    jieba = None

# 英文分词和BLEU评估
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate import bleu_score
    _HAS_NLTK = True
    print("✓ Using [NLTK]")
except Exception:
    _HAS_NLTK = False

# 特殊token定义
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

def collate_fn(batch):
    """
    DataLoader的batch处理函数：
    将不同长度的序列padding到相同长度
    
    返回：
      src_batch: [B, T_src] - 源语言batch
      src_lens : [B] - 每个源句的真实长度
      tgt_batch: [B, T_tgt] - 目标语言batch
      tgt_lens : [B] - 每个目标句的真实长度
    """
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    pad_idx = 0  # PAD token的index
    batch_size = len(batch)

    src_batch = torch.full((batch_size, max_src), pad_idx, dtype=torch.long)
    tgt_batch = torch.full((batch_size, max_tgt), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_batch[i, : len(s)] = s
        tgt_batch[i, : len(t)] = t

    return src_batch, torch.tensor(src_lens, dtype=torch.long), \
           tgt_batch, torch.tensor(tgt_lens, dtype=torch.long)

def write_log(content, log_path="./checkpoints/rnn_zh_en_log.txt", add_timestamp=True):
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
        print(f"写入日志失败: {e}")


class Vocab:
    """
    词表类：用于token和index的相互转换
    """
    def __init__(self, min_freq: int = 1, specials=None):
        self.freqs: Dict[str, int] = {}  # token → 频率
        self.itos: List[str] = []        # index → token (index to string)
        self.stoi: Dict[str, int] = {}   # token → index (string to index)
        
        if specials is None:
            specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        
        self.specials = specials
        for sp in specials:
            self.add_token(sp)
        self.min_freq = min_freq

    def add_token(self, tok: str):
        """统计token出现次数"""
        self.freqs[tok] = self.freqs.get(tok, 0) + 1

    def build_vocab(self):
        """
        构建最终词表：
        1. 保留特殊token（放在前面）
        2. 根据最小频率过滤token
        """
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
        """添加token到词表"""
        idx = len(self.itos)
        self.itos.append(tok)
        self.stoi[tok] = idx

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        """将token列表转换为index列表"""
        unk = self.stoi[UNK_TOKEN]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """将index列表转换为token列表"""
        return [self.itos[i] for i in ids]


def clean_text(text: str, lang: str) -> str:
    """
    简单的文本清洗：
    - 去掉首尾空白、多余空格
    - 只保留有效字符（中英文字符和标点）
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # 统一空白

    if lang == "en":
        # 英文：保留字母、数字、常见标点
        text = re.sub(r"[^a-zA-Z0-9\s\.,;:!\?'\"-\(\)]", "", text)
        text = text.lower()
    else:  # zh
        # 中文：保留CJK字符和中文标点
        text = re.sub(
            r"[^\u4e00-\u9fff0-9a-zA-Z\s，。、《》？：""；：（）！…—\-]",
            "",
            text,
        )
    return text


def tokenize_en(text: str) -> List[str]:
    """英文分词（使用NLTK或正则表达式）"""
    text = clean_text(text, lang="en")
    if not text:
        return []

    if _HAS_NLTK:
        try:
            tokens = word_tokenize(text)
            return [t for t in tokens if t.strip()]
        except LookupError:
            pass

    # Fallback：简单规则分词
    tokens = []
    for word in text.split():
        parts = re.findall(r"\w+|[^\w\s]", word)
        tokens.extend([p for p in parts if p])
    return tokens


def tokenize_zh(text: str) -> List[str]:
    """中文分词（使用jieba或按字符分割）"""
    text = clean_text(text, lang="zh")
    if not text:
        return []

    if jieba is not None:
        return [t.strip() for t in jieba.lcut(text) if t.strip()]
    else:
        # Fallback：按字符切分
        return [ch for ch in text if ch.strip()]
    

class NMTDataset(Dataset):
    """
    神经机器翻译数据集类
    
    输入：jsonl格式文件，每行为 {"en": "...", "zh": "...", "index": 123}
    处理流程：
      1. 文本清洗 → 分词
      2. 过滤短/长句子
      3. 构建词表
      4. 转换为index序列
    """

    def __init__(self,
                 path: str,
                 src_lang: str = "en",
                 tgt_lang: str = "zh",
                 src_vocab: Vocab = None,
                 tgt_vocab: Vocab = None,
                 build_vocab: bool = True,
                 min_freq: int = 1,
                 max_samples: int = None,
                 max_src_len: int = 80,
                 max_tgt_len: int = 80,
                 truncate_long: bool = True,
                 max_vocab_size: int = None):
        """
        Args:
            path: 数据文件路径
            src_lang: 源语言（默认英文）
            tgt_lang: 目标语言（默认中文）
            src_vocab/tgt_vocab: 已有的词表（用于验证集）
            build_vocab: 是否构建词表
            max_src_len: 源语言最大长度
            max_tgt_len: 目标语言最大长度
            truncate_long: True=截断超长句子; False=丢弃
            max_vocab_size: 词表大小限制
        """
        self.samples = []

        # 选择分词函数
        if src_lang == "en":
            self._src_tok = tokenize_en
        else:
            self._src_tok = tokenize_zh

        if tgt_lang == "zh":
            self._tgt_tok = tokenize_zh
        else:
            self._tgt_tok = tokenize_en

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # 初始化或使用现有词表
        if src_vocab is None:
            self.src_vocab = Vocab(min_freq=min_freq)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = Vocab(min_freq=min_freq)
        else:
            self.tgt_vocab = tgt_vocab

        # 读取和预处理数据
        dropped_too_long = 0
        dropped_empty = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)

                raw_src = obj[src_lang]
                raw_tgt = obj[tgt_lang]

                # 清洗+分词
                src_tokens = self._src_tok(raw_src)
                tgt_tokens = self._tgt_tok(raw_tgt)

                if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                    dropped_empty += 1
                    continue

                # 长度过滤/截断
                def handle_len(tokens, max_len):
                    nonlocal dropped_too_long
                    if len(tokens) > max_len:
                        if truncate_long:
                            return tokens[:max_len]
                        else:
                            dropped_too_long += 1
                            return None
                    return tokens

                src_tokens = handle_len(src_tokens, max_src_len)
                tgt_tokens = handle_len(tgt_tokens, max_tgt_len)

                if src_tokens is None or tgt_tokens is None:
                    continue

                self.samples.append((src_tokens, tgt_tokens))

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        write_log(f"[Dataset] Loaded {len(self.samples)} pairs from {path}")
        write_log(f"[Dataset] Dropped empty: {dropped_empty}, "
              f"dropped too long: {dropped_too_long}")

        # 构建词表
        if build_vocab:
            for src_tokens, tgt_tokens in self.samples:
                for tok in src_tokens:
                    self.src_vocab.add_token(tok)
                for tok in tgt_tokens:
                    self.tgt_vocab.add_token(tok)

            self.src_vocab.build_vocab()
            self.tgt_vocab.build_vocab()

            # 限制词表大小
            if max_vocab_size is not None:
                self._shrink_vocab(self.src_vocab, max_vocab_size, name="src")
                self._shrink_vocab(self.tgt_vocab, max_vocab_size, name="tgt")

            write_log(f"[Vocab] src vocab size = {len(self.src_vocab)}")
            write_log(f"[Vocab] tgt vocab size = {len(self.tgt_vocab)}")

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

        write_log(f"[Vocab] {name} vocab shrunk to {len(vocab)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一个样本：源语言和目标语言的index序列
        （自动添加BOS和EOS token）
        """
        src_tokens, tgt_tokens = self.samples[idx]
        src_tokens = [BOS_TOKEN] + src_tokens + [EOS_TOKEN]
        tgt_tokens = [BOS_TOKEN] + tgt_tokens + [EOS_TOKEN]
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# ---
# 
# ## 第6部分：预训练词向量加载
# 
# **作用**：加载预训练的词向量（如Word2Vec, FastText）来初始化Embedding层

# In[6]:


def load_pretrained_embeddings(embedding_path: str,
                               vocab: Vocab,
                               embedding_dim: int,
                               encoding: str = "utf-8") -> torch.Tensor:
    """
    从预训练词向量文件加载embeddings
    
    文件格式示例（每行）：
        word val1 val2 ... valN
    
    Args:
        embedding_path: 预训练向量文件路径
        vocab: 当前词表
        embedding_dim: 向量维度
    
    Returns:
        weight: [vocab_size, embedding_dim] 的张量
    """
    write_log(f"[Emb] Loading pretrained embeddings from {embedding_path} ...")
    word2vec = {}
    
    with open(embedding_path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= embedding_dim:
                continue
            word = parts[0]
            try:
                vec = torch.tensor(
                    [float(x) for x in parts[1:1 + embedding_dim]],
                    dtype=torch.float,
                )
            except ValueError:
                continue
            if vec.size(0) != embedding_dim:
                continue
            word2vec[word] = vec

    vocab_size = len(vocab)
    weight = torch.zeros(vocab_size, embedding_dim)
    # 随机初始化未匹配的embeddings
    nn.init.normal_(weight, mean=0.0, std=0.1)

    oov_count = 0
    for idx, tok in enumerate(vocab.itos):
        if tok in word2vec:
            weight[idx] = word2vec[tok]
        else:
            oov_count += 1

    write_log(f"[Emb] Loaded embeddings for {len(word2vec)} tokens, "
          f"OOV in vocab: {oov_count}/{vocab_size}")
    return weight


# ---
# 
# ## 第7部分：编码器 (Encoder)
# 
# **作用**：将源句子编码为固定的上下文向量

# In[7]:


# class EncoderRNN(nn.Module):
#     """
#     RNN编码器：双向或单向RNN
#     输入：源语言序列
#     输出：最终的隐藏状态和所有时间步的输出
#     """
#     def __init__(self, vocab_size, emb_size, hidden_size, 
#                  num_layers=2, rnn_type="gru", dropout=0.1, pad_idx=0):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn_type = rnn_type.lower()
        
#         # Embedding层
#         self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        
#         # RNN层（GRU或LSTM）
#         rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
#         self.rnn = rnn_cls(
#             emb_size,
#             hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=False,
#             dropout=dropout if num_layers > 1 else 0.0,
#         )

#     def forward(self, src, src_lens):
#         """
#         Args:
#             src: [B, T] - 源句子的token indices
#             src_lens: [B] - 每个句子的真实长度
        
#         Returns:
#             outputs: [B, T, H] - 所有时间步的隐藏状态
#             hidden: [num_layers, B, H] - 最后时间步的隐藏状态
#         """
#         # Embedding: [B, T] → [B, T, E]
#         embedded = self.embedding(src)
        
#         # Pack序列（忽略padding）
#         packed = nn.utils.rnn.pack_padded_sequence(
#             embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
#         )
        
#         # RNN前向传播
#         outputs, hidden = self.rnn(packed)
        
#         # Unpack序列
#         outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
#         return outputs, hidden

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=2, rnn_type="gru", dropout=0.1, pad_idx=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        
        # RNN层（GRU或LSTM）
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            emb_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # 【修改】开启双向
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 【新增】将双向 hidden (2*H) 映射回 Decoder 需要的 (H)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) 
        
        # outputs: [B, T, 2*H] -> 需要映射或者求和，这里简化处理：
        # 将双向输出相加：(Batch, Seq, 2, Hidden) -> Sum -> (Batch, Seq, Hidden)
        # 或者使用 Linear 层映射
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        
        # 处理 hidden state
        # GRU hidden: [num_layers * 2, B, H]
        if self.rnn_type == "gru":
            # 将双向的层重新排列并相加或拼接。简单做法：只取最后一层的前向
            # 但更好的做法是训练一个线性层合并。
            # 为了代码改动最小化，这里采用求和策略：
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size) # [L, 2, B, H]
            hidden = torch.sum(hidden, dim=1) # [L, B, H]
        else: # LSTM (h, c)
            # 类似处理，略
            pass
            
        return outputs, hidden


# ---
# 
# ## 第8部分：注意力机制 (Attention)
# 
# **作用**：让解码器关注源句子中的重要部分

# In[8]:


class Attention(nn.Module):
    """
    注意力机制：支持3种对齐函数
      - dot:        点积（Luong）
      - general:    乘性（Luong general）
      - concat:     加性（Bahdanau）
    """
    def __init__(self, hidden_size, align_type="dot"):
        super().__init__()
        self.hidden_size = hidden_size
        self.align_type = align_type
        
        if align_type == "general":
            self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        elif align_type == "concat":
            self.linear_in = nn.Linear(hidden_size * 2, hidden_size, bias=True)
            self.v = nn.Parameter(torch.rand(hidden_size))

    def score(self, query, keys):
        """
        计算注意力分数
        Args:
            query: [B, H] - 解码器隐藏状态
            keys: [B, T, H] - 编码器输出
        
        Returns:
            scores: [B, T] - 注意力分数
        """
        if self.align_type == "dot":
            # 点积：query·keys
            return torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        
        elif self.align_type == "general":
            # W·query·keys
            q = self.linear_in(query)
            return torch.bmm(keys, q.unsqueeze(2)).squeeze(2)
        
        elif self.align_type == "concat":
            # v^T·tanh(W[query; keys])
            B, T, H = keys.size()
            query_expanded = query.unsqueeze(1).expand(B, T, H)
            concat = torch.cat((query_expanded, keys), dim=2)
            energy = torch.tanh(self.linear_in(concat))
            v = self.v.unsqueeze(0).unsqueeze(2)
            v_expanded = v.expand(B, -1, -1)
            scores = torch.bmm(energy, v_expanded).squeeze(2)
            return scores

    def forward(self, query, keys, mask=None):
        """
        注意力前向传播
        
        Returns:
            context: [B, H] - 上下文向量
            attn_weights: [B, T] - 注意力权重
        """
        scores = self.score(query, keys)
        
        # 应用mask（忽略padding）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax得到注意力权重
        attn_weights = torch.softmax(scores, dim=1)
        
        # 加权求和得到上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        
        return context, attn_weights


# ---
# 
# ## 第9部分：解码器 (Decoder)
# 
# **作用**：逐步生成目标句子，结合注意力机制

# In[9]:


class DecoderRNN(nn.Module):
    """
    RNN解码器：带注意力机制的自回归生成
    """
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=2, 
                 rnn_type="gru", dropout=0.1, pad_idx=0, attention: Attention = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        # 输入 = embedding + 编码器隐藏状态
        self.rnn = rnn_cls(
            emb_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.attention = attention
        # 输出：隐藏状态 + 上下文 → 词表
        self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward_step(self, input_tokens, last_hidden, encoder_outputs, src_mask):
        """
        单步解码
        
        Args:
            input_tokens: [B] - 当前token
            last_hidden: [num_layers, B, H] - 上一步隐藏状态
            encoder_outputs: [B, T_src, H] - 编码器输出
            src_mask: [B, T_src] - 源句mask
        
        Returns:
            logits: [B, V] - 词表上的logits
            hidden: [num_layers, B, H] - 新的隐藏状态
            attn_weights: [B, T_src] - 注意力权重
        """
        # Embedding
        embedded = self.embedding(input_tokens).unsqueeze(1)  # [B, 1, E]
        
        # 注意力query（取顶层隐藏状态）
        if isinstance(last_hidden, tuple):  # LSTM
            query = last_hidden[0][-1]
        else:  # GRU
            query = last_hidden[-1]
        
        # 计算上下文向量
        context, attn_weights = self.attention(query, encoder_outputs, mask=src_mask)
        
        # RNN输入 = embedding + 上下文
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(1)
        
        # 预测logits
        concat_output = torch.cat((output, context), dim=1)
        logits = self.out(concat_output)
        
        return logits, hidden, attn_weights

    def forward(self, tgt, encoder_hidden, encoder_outputs, src_mask, 
                teacher_forcing_ratio=1.0, max_len=None):
        """
        完整解码（带Teacher Forcing）
        
        Args:
            tgt: [B, T_tgt] - 目标句子（包含BOS和EOS）
            teacher_forcing_ratio: 1.0=完全使用真实标签, 0.0=完全自回归
        
        Returns:
            outputs: [B, L, V] - 所有时间步的logits
            hidden: 最终隐藏状态
            all_attn: [B, L, T_src] - 注意力权重序列
        """
        B, T = tgt.size()
        if max_len is None:
            max_len = T - 1
        
        outputs = []
        all_attn = []
        device = tgt.device

        input_tokens = tgt[:, 0]  # BOS
        hidden = encoder_hidden
        
        for t in range(max_len):
            logits, hidden, attn_weights = self.forward_step(
                input_tokens, hidden, encoder_outputs, src_mask
            )
            outputs.append(logits.unsqueeze(1))
            all_attn.append(attn_weights.unsqueeze(1))

            # Teacher Forcing：随机决定使用真实标签还是预测标签
            use_teacher = random.random() < teacher_forcing_ratio
            if use_teacher and t + 1 < T:
                input_tokens = tgt[:, t + 1]
            else:
                input_tokens = logits.argmax(dim=-1)

        outputs = torch.cat(outputs, dim=1)
        all_attn = torch.cat(all_attn, dim=1)
        return outputs, hidden, all_attn


# ---
# 
# ## 第10部分：Seq2Seq 模型
# 
# **作用**：整合编码器和解码器，提供不同的解码策略

# In[10]:


class Seq2Seq(nn.Module):
    """
    完整的Seq2Seq模型
    """
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN,
                 pad_idx=0, bos_idx=1, eos_idx=2, device="cpu"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.device = device

    def make_src_mask(self, src):
        """创建源句mask（非padding位置为1）"""
        return (src != self.pad_idx).int()

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=1.0):
        """
        训练模式的前向传播
        """
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_lens = src_lens.to(self.device)
        
        encoder_outputs, encoder_hidden = self.encoder(src, src_lens)
        src_mask = self.make_src_mask(src)
        
        outputs, hidden, attn = self.decoder(
            tgt, encoder_hidden, encoder_outputs, src_mask, 
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        return outputs, attn

    def greedy_decode(self, src, src_lens, max_len=50):
        """
        贪心解码：每一步选择概率最大的token
        """
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            src = src.to(self.device)
            src_lens = src_lens.to(self.device)
            
            encoder_outputs, encoder_hidden = self.encoder(src, src_lens)
            src_mask = self.make_src_mask(src)
            
            B = src.size(0)
            input_tokens = torch.full((B,), self.bos_idx, dtype=torch.long, device=self.device)
            hidden = encoder_hidden
            decoded_ids = [[] for _ in range(B)]
            finished = [False] * B
            
            for _ in range(max_len):
                logits, hidden, _ = self.decoder.forward_step(
                    input_tokens, hidden, encoder_outputs, src_mask
                )
                next_tokens = logits.argmax(dim=-1)
                input_tokens = next_tokens
                
                for i in range(B):
                    if not finished[i]:
                        token_id = next_tokens[i].item()
                        if token_id == self.eos_idx:
                            finished[i] = True
                        else:
                            decoded_ids[i].append(token_id)
                
                if all(finished):
                    break
        
        return decoded_ids

    def beam_search_decode(self, src, src_lens, beam_size=5, max_len=50, length_norm=True):
        """
        集束搜索解码：维护k个最优候选
        （目前仅支持batch_size=1）
        """
        assert src.size(0) == 1, "beam_search_decode currently only supports batch_size=1"
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            src = src.to(self.device)
            src_lens = src_lens.to(self.device)
            
            encoder_outputs, encoder_hidden = self.encoder(src, src_lens)
            src_mask = self.make_src_mask(src)

            init_token = self.bos_idx
            beam = [(0.0, [init_token], encoder_hidden)]
            completed = []

            for _ in range(max_len):
                new_beam = []
                
                for log_prob, tokens, hidden in beam:
                    last_token = tokens[-1]
                    
                    if last_token == self.eos_idx:
                        completed.append((log_prob, tokens))
                        new_beam.append((log_prob, tokens, hidden))
                        continue

                    input_tokens = torch.tensor([last_token], device=self.device)
                    logits, new_hidden, _ = self.decoder.forward_step(
                        input_tokens, hidden, encoder_outputs, src_mask
                    )
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                    
                    for k in range(beam_size):
                        new_lp = log_prob + topk_log_probs[k].item()
                        new_tokens = tokens + [topk_ids[k].item()]
                        new_beam.append((new_lp, new_tokens, new_hidden))
                
                # 保留top-k候选
                new_beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[: beam_size]
                beam = new_beam

            if not completed:
                completed = [(lp, toks) for lp, toks, _ in beam]
            
            if length_norm:
                completed = [(lp / max(1, len(toks) - 1), toks) for lp, toks in completed]

            best = max(completed, key=lambda x: x[0])
            out_tokens = []
            for tid in best[1][1:]:  # 跳过BOS
                if tid == self.eos_idx:
                    break
                out_tokens.append(tid)
            
            return out_tokens


# ---
# 
# ## 第11部分：BLEU评估函数
# 
# **作用**：计算翻译质量的BLEU分数

# In[11]:


def compute_bleu_corpus(references, hypotheses):
    """
    计算corpus-level BLEU-4分数
    
    Args:
        references: List[List[List[str]]] - 参考翻译
        hypotheses: List[List[str]] - 模型预测
    
    Returns:
        float - BLEU-4分数
    """
    if not _HAS_NLTK:
        raise RuntimeError("需要NLTK来计算BLEU，请pip install nltk")
    
    smoothie = bleu_score.SmoothingFunction().method1
    return bleu_score.corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )

def evaluate_bleu(model, dataloader, tgt_vocab, device, max_len=50):
    model.eval()
    pad_tok = PAD_TOKEN
    bos_tok = BOS_TOKEN
    eos_tok = EOS_TOKEN

    references = []
    hypotheses = []

    with torch.no_grad():
        for src_batch, src_lens, tgt_batch, tgt_lens in dataloader:
            # greedy 解码
            pred_ids_batch = model.greedy_decode(src_batch, src_lens, max_len=max_len)

            # 将 ID 转为 Token
            batch_size = src_batch.size(0)
            for i in range(batch_size):
                pred_ids = pred_ids_batch[i]
                target_ids = tgt_batch[i].tolist()
                
                # 1. 处理预测结果 (Hypothesis)
                hyp_tokens = []
                for tid in pred_ids:
                    tok = tgt_vocab.itos[tid]
                    if tok == eos_tok: break
                    if tok in (pad_tok, bos_tok): continue
                    hyp_tokens.append(tok)
                
                # 2. 处理参考结果 (Reference)
                ref_tokens = []
                for tid in target_ids:
                    tok = tgt_vocab.itos[tid]
                    if tok == eos_tok: break
                    if tok in (pad_tok, bos_tok): continue
                    ref_tokens.append(tok)
                
                # 只有非空才加入统计
                if len(ref_tokens) > 0:
                    references.append([ref_tokens])
                    hypotheses.append(hyp_tokens)

            # beam_search 解码
            # pred_ids = model.beam_search_decode(src_batch, src_lens, max_len=max_len)
            # target_ids = tgt_batch[0].tolist()
            
            # # 1. 处理预测结果 (Hypothesis)
            # hyp_tokens = []
            # for tid in pred_ids:
            #     tok = tgt_vocab.itos[tid]
            #     if tok == eos_tok: break
            #     if tok in (pad_tok, bos_tok): continue
            #     hyp_tokens.append(tok)
            
            # # 2. 处理参考结果 (Reference)
            # ref_tokens = []
            # for tid in target_ids:
            #     tok = tgt_vocab.itos[tid]
            #     if tok == eos_tok: break
            #     if tok in (pad_tok, bos_tok): continue
            #     ref_tokens.append(tok)
            
            # # 只有非空才加入统计
            # if len(ref_tokens) > 0:
            #     references.append([ref_tokens])
            #     hypotheses.append(hyp_tokens)

    # 打印前 3 条样本看看生成效果 (Debug 用)
    if len(hypotheses) > 0:
        print("\n--- Evaluation Examples ---")
        for k in range(min(3, len(hypotheses))):
            print(f"Ref: {' '.join(references[k][0])}")
            print(f"Hyp: {' '.join(hypotheses[k])}")
        print("---------------------------\n")

    if len(hypotheses) == 0:
        return 0.0

    # 计算 BLEU
    bleu = compute_bleu_corpus(references, hypotheses)
    return bleu * 100.0


# ---
# 
# ## 第12部分：训练函数
# 
# **作用**：单个epoch的训练逻辑

# In[12]:


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=1.0):
    """
    训练一个epoch
    
    Args:
        teacher_forcing_ratio: 1.0=完全使用真实标签, 0.0=完全自回归
                               中间值为混合策略
    
    Returns:
        float - 平均loss
    """
    model.train()
    total_loss = 0.0
    
    for src_batch, src_lens, tgt_batch, tgt_lens in dataloader:
        src_batch = src_batch.to(device)
        src_lens = src_lens.to(device)
        tgt_batch = tgt_batch.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs, _ = model(src_batch, src_lens, tgt_batch,
                           teacher_forcing_ratio=teacher_forcing_ratio)
        
        # 计算loss（目标 = tgt去掉BOS）
        target = tgt_batch[:, 1: 1 + outputs.size(1)]
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                        target.reshape(-1))
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ---
# 
# ## 第13部分：主函数和训练循环
# 
# **作用**：完整的训练流程，包括模型初始化、数据加载、训练和评估

# In[13]:


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RNN-based NMT (EN→ZH) with Attention")
    parser.add_argument("--data_path", type=str, default="./data/train_100k.jsonl")
    parser.add_argument("--valid_path", type=str, default="./data/valid.jsonl")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--save_path", type=str, default="./checkpoints/rnn_zh_en_best.pt")
    parser.add_argument("--log_path", type=str, default="./checkpoints/rnn_zh_en_log.txt")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--align_type", type=str, default="dot", choices=["dot", "general", "concat"])
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_src_len", type=int, default=80)
    parser.add_argument("--max_tgt_len", type=int, default=80)
    parser.add_argument("--truncate_long", action="store_true")
    parser.add_argument("--max_vocab_size", type=int, default=None)
    
    parser.add_argument("--src_emb_path", type=str, default=None)
    parser.add_argument("--tgt_emb_path", type=str, default=None)
    
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1.0)
    parser.add_argument("--ratio_decay", type=float, default=0.1)

    args = parser.parse_args()

    # ========== 1. 加载训练集 ==========
    write_log("Loading dataset ...")
    dataset = NMTDataset(
        args.data_path,
        src_lang="zh",
        tgt_lang="en",
        min_freq=args.min_freq,
        max_samples=args.max_samples,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        truncate_long=args.truncate_long,
        max_vocab_size=args.max_vocab_size,
    )

    pad_idx = dataset.src_vocab.stoi[PAD_TOKEN]
    bos_idx = dataset.tgt_vocab.stoi[BOS_TOKEN]
    eos_idx = dataset.tgt_vocab.stoi[EOS_TOKEN]

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # ========== 2. 加载验证集（共享词表） ==========
    write_log("Loading valid dataset ...")
    valid_dataset = NMTDataset(
        args.valid_path,
        src_lang="zh",
        tgt_lang="en",
        src_vocab=dataset.src_vocab,
        tgt_vocab=dataset.tgt_vocab,
        build_vocab=False,
        min_freq=args.min_freq,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        truncate_long=args.truncate_long,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ========== 3. 构建模型 ==========
    write_log("Building model ...")
    encoder = EncoderRNN(
        len(dataset.src_vocab),
        args.emb_size,
        args.hidden_size,
        num_layers=2,
        rnn_type=args.rnn_type,
        pad_idx=dataset.src_vocab.stoi[PAD_TOKEN],
    )
    
    attention = Attention(args.hidden_size, align_type=args.align_type)
    
    decoder = DecoderRNN(
        len(dataset.tgt_vocab),
        args.emb_size,
        args.hidden_size,
        num_layers=2,
        rnn_type=args.rnn_type,
        attention=attention,
        pad_idx=dataset.tgt_vocab.stoi[PAD_TOKEN],
    )

    # ========== 4. 加载预训练词向量（可选） ==========
    if args.src_emb_path is not None:
        src_weight = load_pretrained_embeddings(
            args.src_emb_path, dataset.src_vocab, args.emb_size
        )
        encoder.embedding.weight.data.copy_(src_weight)

    if args.tgt_emb_path is not None:
        tgt_weight = load_pretrained_embeddings(
            args.tgt_emb_path, dataset.tgt_vocab, args.emb_size
        )
        decoder.embedding.weight.data.copy_(tgt_weight)

    encoder.embedding.weight.requires_grad = True
    decoder.embedding.weight.requires_grad = True

    # ========== 5. 创建Seq2Seq模型 ==========
    model = Seq2Seq(
        encoder,
        decoder,
        pad_idx=dataset.src_vocab.stoi[PAD_TOKEN],
        bos_idx=dataset.tgt_vocab.stoi[BOS_TOKEN],
        eos_idx=dataset.tgt_vocab.stoi[EOS_TOKEN],
        device=args.device,
    )

    model.to(args.device)

    # ========== 6. 定义优化器和损失函数 ==========
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    write_log("Start training ...", args.log_path)
    best_val_bleu = -1.0

    # 确保保存目录存在
    save_dir = os.path.dirname(args.save_path)
    if save_dir != "" and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # ========== 7. 训练循环 ==========
    for epoch in range(1, args.epochs + 1):
        # 训练一个epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            args.device,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
        )

    # 在 main 函数的 loop 中
    # for epoch in range(1, args.epochs + 1):
    #     # 动态计算 ratio：从 1.0 开始，每轮减少，最低 0.0 (全自主生成)
    #     # 公式示例：max(0.0, 1.0 - 0.1 * (epoch - 1))
    #     # 建议设置一个地板值，比如 0.3
    #     ratio = max(0.3, 1.0 - (epoch * args.ratio_decay))
        
    #     write_log(f"Epoch {epoch} use Teacher Forcing Ratio: {ratio:.2f}", args.log_path)
        
    #     train_loss = train_epoch(
    #         model, 
    #         train_loader, 
    #         optimizer, 
    #         criterion, 
    #         args.device, 
    #         teacher_forcing_ratio=ratio # 传入动态比率
    #     )

        write_log(f"Epoch {epoch}: train loss = {train_loss:.4f}", args.log_path)

        # 在验证集上计算BLEU
        try:
            val_bleu = evaluate_bleu(
                model,
                valid_loader,
                dataset.tgt_vocab,
                args.device,
                max_len=args.max_tgt_len,
            )
            write_log(f"Epoch {epoch}: valid BLEU-4 = {val_bleu:.2f}", args.log_path)
        except RuntimeError as e:
            write_log(f"Epoch {epoch}: BLEU evaluation skipped ({e})", args.log_path)
            val_bleu = -1.0

        # 如果BLEU提高，保存模型
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "src_vocab": dataset.src_vocab,
                "tgt_vocab": dataset.tgt_vocab,
                "pad_idx": pad_idx,
                "bos_idx": bos_idx,
                "eos_idx": eos_idx,
                "config": {
                    "emb_size": args.emb_size,
                    "hidden_size": args.hidden_size,
                    "rnn_type": args.rnn_type,
                    "align_type": args.align_type,
                    "num_layers": 2,
                },
            }
            torch.save(checkpoint, args.save_path)
            write_log(f"[Checkpoint] New best model saved with BLEU-4 = {best_val_bleu:.2f} "
                  f"to {args.save_path}", args.log_path)


if __name__ == "__main__":
    main()