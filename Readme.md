# Chinese–English Machine Translation with RNN and Transformer

**Student ID**: 250010021 
**Name**: 周朗
**Group**: Group 8 

Code & Checkpoints Repository: [https://github.com/Tommy307/AP004-FinalProject](https://github.com/Tommy307/AP004-FinalProject)

---

## 1. Introduction

In this project, we implement and compare Chinese–English machine translation models based on both RNN and Transformer architectures. 

The goal is to analyze their architectural differences, training efficiency, translation quality, and practical trade-offs through systematic experiments.

---

## 2. Dataset

### 2.1 Dataset Overview

The dataset consists of four JSONL files: a small training set, a large training set, a validation set, and a test set. Each line in the files contains a parallel Chinese–English sentence pair.

| Split        | Size |
|-------------|------|
| Small Train | 10k  |
| Large Train | 100k |
| Validation  | 500  |
| Test        | 200  |

All experiments in this project are conducted on the **larger training set**.

---

### 2.2 Data Preprocessing

The following preprocessing steps are applied:

- **Data Cleaning**: Illegal characters are removed, and excessively long sentence pairs are filtered or truncated.
- **Tokenization**:
  - Chinese sentences are segmented using **Jieba**.
  - English sentences are tokenized using **NLTK**.
- **Vocabulary Construction**: A vocabulary is built from the training data, with low-frequency tokens optionally filtered.
- **Embedding Initialization**: Word embeddings are initialized either randomly or using pretrained vectors and fine-tuned during training.

---

## 3. RNN-based Neural Machine Translation

### 3.1 Model Architecture

The RNN-based NMT model follows an encoder–decoder architecture. Both the encoder and decoder consist of two unidirectional recurrent layers implemented with either LSTM or GRU units.

An attention mechanism is incorporated to allow the decoder to dynamically focus on relevant encoder hidden states during translation.

---

### 3.2 Attention Mechanisms

We experiment with different alignment functions, including:

- Dot-product attention
- Multiplicative attention
- Additive attention

These variants are evaluated to investigate their impact on translation performance and convergence behavior.

In this experiment:

- The vocabulary size is fixed to 1k
- Teacher forcing ratio is decaying during training from 1 to 0.3, with decreasing 0.1 each epoch 
- All using greedy decoding

|                          | Best-Epoch | Ckpt (./checkpoints)                                         | Val BLEU-4 | Test BLEU-4 |
| ------------------------ | ---------- | ------------------------------------------------------------ | ---------- | ----------- |
| Dot-product attention    | 20         | rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1              | 15.80      | 18.54       |
| Multiplicative attention | 9          | rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1_attention-general | 13.77      | 14.51       |
| Additive attention       | 28         | rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1_attention-concat | **16.30**  | **17.34**   |



---

### 3.3 Training and Decoding Strategies

- **Training Policy**:
  - Teacher Forcing
  - Free Running
- **Decoding Policy**:
  - Greedy decoding
  - Beam search decoding

The effectiveness of different training and decoding strategies is compared experimentally.

In this experiment:

- The vocabulary size is fixed to 1k
- Attention type is using additive attention
- All using greedy decoding

|                 | Ratio | Best-Epoch | Ckpt (./checkpoints)                                         | Val BLEU-4 | Test BLEU-4 |
| --------------- | ----- | ---------- | ------------------------------------------------------------ | ---------- | ----------- |
| Teacher Forcing | 1     | 12         | rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1_attention-concat_teaFor-1.pt | 20.20      | 23.74       |
|                 | 0.7   | 23         | rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1_attention-concat_teaFor-0-7.pt | 17.21      | 21.22       |
|                 | 0.5   | 12         | rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1_attention-concat_teaFor-0-5.pt | 17.00      | 17.06       |
|                 | 0.3   | 19         | rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1_attention-concat_teaFor-0-3.pt | 16.77      | 18.90       |
| Free Running    | 0     | 24         | rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1_attention-concat_teaFor-0.pt | 14.58      | 15.85       |



In this experiment:

- The vocabulary size is fixed to 1k
- Attention type is using additive attention
- Teacher Forcing Ratio 1
- CKPT: rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1_attention-concat_teaFor-1.pt

|                      | Test BLEU-4 |
| -------------------- | ----------- |
| Greedy decoding      | 23.74       |
| Beam search decoding | 22.32       |



---

### 3.4 Experimental Results (RNN)

- Observed strengths and weaknesses:
  - The RNN can not handling big vocabulary in this experiments.
    - That's the reason we **fix the max vocabulary length at 1k**
    - Thought it will limit the expressing ability of our model.

  - The teacher forcing strategy is imprtant for rnn model to learn from corpus
  - Finally, the greedy decoding method is fast and better.


---

## 4. Transformer-based Neural Machine Translation

### 4.1 Transformer from Scratch

A Transformer-based encoder–decoder model is implemented from scratch. The model relies entirely on self-attention and feed-forward layers, enabling parallel computation during training.

- **We use greedy search decoing strategy in all below experiments!!!**
- **Max epoch is set to 50!!!**

---

This experiment is aimed at finding suitable min_freq and vocabulary length

| Min_freq | Vocabulary Length    | Best Epoch | Val BLEU-4 |
| -------- | -------------------- | ---------- | ---------- |
| 3        | 25758, 22929         | 21         | 5.52       |
| 5        | src=18522, tgt=17232 | 12         | 6.29       |
| 10       | 11797, 11676         | 9          | 6.85       |
| 20       | **7451, 7626**       | 9          | **7.62**   |
| 30       | 5673, 5897           | 20         | 8.5        |
| 40       | 4638, 4847           | 25         | 9.06       |
| 50       | 3957, 4160           | 40         | 9.26       |

In order to balance the bleu and output quality, we choose **min_freq at 20**.

---

### 4.2 Architectural Ablation Studies

We investigate the impact of architectural choices, including:

- **Position Embedding**:
  - Absolute positional encoding
  - Relative positional encoding
  
  |                              | Best-Epoch | Val BLEU-4 | Test BLEU-4 |
  | ---------------------------- | ---------- | ---------- | ----------- |
  | Absolute positional encoding | 48         | 7.58       | 7.18        |
  | Relative positional encoding | 46         | 11.42      | 10.00       |
  
- **Normalization Method**:
  
  - In this experiment, we use relative positional encoding!
  - Layer Normalization
  - RMS Normalization
  
  |                     | Best-Epoch | Val BLEU-4 | Test BLEU-4 |
  | ------------------- | ---------- | ---------- | ----------- |
  | Layer Normalization | 46         | 11.42      | 10.00       |
  | RMS Normalization   | 47         | 11.47      | 9.15        |

Each variant is trained from scratch and evaluated on the validation and test sets.

---

### 4.3 Hyperparameter Sensitivity

To analyze model robustness, we vary the following hyperparameters:

- Batch size: default 64
- Learning rate: default 1e-4
- hidden size: default 256
- number of layers: default 4

In this experiments, we use Relative positional encoding and RMS Normalization!

| Batch size | Best-Epoch | Val BLEU-4 | Test BLEU-4 |
| ---------- | ---------- | ---------- | ----------- |
| 64         | 47         | 11.47      | 9.15        |
| 32         | 46         | **12.41**  | 10.13       |
| 16         | 24         | 11.55      | **10.53**   |



| Learning rate | Best-Epoch | Val BLEU-4 | Test BLEU-4 |
| ------------- | ---------- | ---------- | ----------- |
| 5e-5          | 42         | 10.20      | 8.23        |
| 1e-4          | 47         | 11.47      | 9.15        |
| 5e-4          | 37         | **12.57**  | **10.46**   |
| 1e-3          | 45         | 11.99      | 10.04       |



| hidden size | Best-Epoch | Val BLEU-4 | Test BLEU-4 |
| ----------- | ---------- | ---------- | ----------- |
| 128         | 50         | 9.96       | 8.43        |
| 256         | 47         | 11.47      | 9.15        |
| 512         | 43         | 11.79      | **10.58**   |



| number of layers | Best-Epoch | Val BLEU-4 | Test BLEU-4 |
| ---------------- | ---------- | ---------- | ----------- |
| 2                | 50         | 8.96       | 7.68        |
| 4                | 47         | 11.47      | 9.15        |
| 8                | 42         | 12.94      | **11.17**   |



The sensitivity of model performance to these factors is examined.

---

### 4.4 Fine-tuning a Pretrained Language Model

A pretrained sequence-to-sequence language model (e.g., T5) is fine-tuned on the Chinese–English translation task. Its performance is compared with Transformer models trained from scratch.

|                       | Val BLEU-4 | Test BLEU-4 |
| --------------------- | ---------- | ----------- |
| From scratch          |            |             |
| Fine-tune on T5-small |            |             |

---

## 5. Evaluation Metrics

### 5.1 BLEU Score

Translation quality is evaluated using the BLEU metric, which measures the n-gram overlap between model-generated translations and reference translations. BLEU is reported on the test set for all models.

---

## 6. Analysis and Comparison

### 6.1 Model Architecture

RNN-based models process sequences sequentially, which limits parallelization but naturally models temporal dependencies. In contrast, Transformer models rely on self-attention, enabling full parallel computation and more flexible modeling of long-range dependencies.

---

### 6.2 Training Efficiency

RNN models generally require longer training times due to their sequential nature, while Transformer models converge faster but demand higher memory and computational resources.

---

### 6.3 Translation Performance

Transformer-based models typically achieve higher BLEU scores and produce more fluent translations, especially for long sentences. RNN-based models may struggle with long-range dependencies despite the use of attention mechanisms.

---

### 6.4 Scalability and Generalization

Transformers scale better with data and model size and demonstrate stronger performance on long sentences. RNN-based models may perform competitively in low-resource settings but show limited scalability.

---

### 6.5 Practical Trade-offs

RNN models are simpler to implement and require fewer resources, whereas Transformer models offer superior performance at the cost of increased complexity and resource consumption.

---

## 7. Conclusion

In this project, we implemented and compared RNN-based and Transformer-based neural machine translation models for Chinese–English translation. Through comprehensive experiments, we analyzed their architectural differences, performance characteristics, and practical trade-offs.

---

## 8. Demo

Run the `inference.py` can try the model trained in this projest.

- For RNN-based, we choose the CKPT: `./checkpoints/rnn_zh_en_best_100k_maxVocab1k_teaRatioDecay0-1_attention-concat_teaFor-1.pt`
  - Do notice that our model use only 1k vocab.
- For Transformer-based, we choose the CKPT: `./checkpoints/transformer_zh_en_best_exp_n-layer-8.pt`

---

## 9. References

- Vaswani et al., *Attention Is All You Need*, 2017  
- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, 2015  
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, 2020  