import torch
import argparse
from nmt_rnn_zh_en import (
    Vocab, 
    EncoderRNN, DecoderRNN, Attention, Seq2Seq,
    # 如果有 Transformer 请在这里导入，例如:
    # TransformerModel,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
)

def build_model_from_checkpoint(checkpoint, model_type, device):
    """根据 checkpoint 重新构建模型"""
    src_vocab = checkpoint["src_vocab"]
    tgt_vocab = checkpoint["tgt_vocab"]
    config = checkpoint["config"]
    pad_idx = checkpoint["pad_idx"]
    bos_idx = checkpoint["bos_idx"]
    eos_idx = checkpoint["eos_idx"]

    if model_type == "rnn":
        encoder = EncoderRNN(
            vocab_size=len(src_vocab),
            emb_size=config["emb_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            rnn_type=config["rnn_type"],
            pad_idx=pad_idx,
        )
        attention = Attention(config["hidden_size"], align_type=config["align_type"])
        decoder = DecoderRNN(
            vocab_size=len(tgt_vocab),
            emb_size=config["emb_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            rnn_type=config["rnn_type"],
            attention=attention,
            pad_idx=pad_idx,
        )
        model = Seq2Seq(encoder, decoder, pad_idx, bos_idx, eos_idx, device)
    else:
        # 这里预留 Transformer 的加载逻辑，请根据你实际的 Transformer 类名修改
        # model = TransformerModel(len(src_vocab), len(tgt_vocab), config, ...)
        print("[Error] Transformer 构造逻辑尚未在 demo 中定义，请检查代码。")
        return None, None, None

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab

def translate_sentence(sentence, model, src_vocab, tgt_vocab, device, method="greedy", max_len=50, beam_size=5):
    """
    处理单句输入并调用模型的 decode 方法
    """
    model.eval()
    
    # 1. 预处理输入：分词、加 BOS/EOS、转 ID
    # 注意：这里的 split() 是简单处理，如果训练时用了更复杂的分词请保持一致
    tokens = [BOS_TOKEN] + sentence.strip().split() + [EOS_TOKEN]
    # src_ids = [src_vocab.stoi.get(t, src_vocab.unk_idx) for t in tokens]
    unk_idx = src_vocab.stoi.get(UNK_TOKEN) 
    src_ids = [src_vocab.stoi.get(t, unk_idx) for t in tokens]
    
    # 构造 batch (size=1)
    src_batch = torch.LongTensor([src_ids]).to(device)
    src_lens = torch.LongTensor([len(src_ids)]).to(device)

    with torch.no_grad():
        if method == "greedy":
            # 匹配你 evaluate_bleu 中的调用方式
            pred_ids_batch = model.greedy_decode(src_batch, src_lens, max_len=max_len)
            pred_ids = pred_ids_batch[0] # 取 batch 中的第一句
        else:
            # 假设你的模型有 beam_search_decode 方法
            # 如果没有，可能需要你在 Seq2Seq 类中实现它
            if hasattr(model, 'beam_search_decode'):
                pred_ids = model.beam_search_decode(src_batch, src_lens, max_len=max_len, beam_size=beam_size)
                # 有些实现 beam_search 返回的是 batch，有些是单句，这里假设取第一名
                if isinstance(pred_ids, list) and len(pred_ids) > 0 and isinstance(pred_ids[0], list):
                    pred_ids = pred_ids[0]
            else:
                print("[Warning] 模型未实现 beam_search_decode，自动降级为 greedy。")
                pred_ids = model.greedy_decode(src_batch, src_lens, max_len=max_len)[0]

    # 2. 将 ID 转为 Token 并过滤特殊符号
    res_tokens = []
    for tid in pred_ids:
        # 兼容 tensor 或 int
        idx = tid.item() if torch.is_tensor(tid) else tid
        tok = tgt_vocab.itos[idx]
        if tok == EOS_TOKEN:
            break
        if tok in (PAD_TOKEN, BOS_TOKEN):
            continue
        res_tokens.append(tok)
    
    return "".join(res_tokens) if "zh" in str(type(tgt_vocab)).lower() else " ".join(res_tokens)

def main():
    parser = argparse.ArgumentParser(description="NMT Interaction Demo")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. 加载 Checkpoint
    print(f"正在从 {args.checkpoint} 加载模型配置...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    # 2. 用户选择模型架构
    print("\n" + "="*30)
    print(" 请选择模型架构类型:")
    print(" 1. RNN-based (Seq2Seq + Attention)")
    print(" 2. Transformer-based")
    arch_choice = input("请输入编号 [默认 1]: ").strip()
    model_type = "transformer" if arch_choice == "2" else "rnn"

    model, src_vocab, tgt_vocab = build_model_from_checkpoint(checkpoint, model_type, args.device)
    if model is None: return

    if model_type == "transformer":
        args.checkpoint = args.checkpoint.replace("checkpoints", "checkpoints_trans")

    # 3. 用户选择解码方式
    print("\n 请选择解码策略:")
    print(" 1. Greedy Search (速度快)")
    print(" 2. Beam Search (质量高)")
    search_choice = input("请输入编号 [默认 1]: ").strip()
    method = "beam" if search_choice == "2" else "greedy"
    
    beam_size = 5
    if method == "beam":
        b_val = input("请输入 Beam Size [默认 5]: ").strip()
        if b_val.isdigit(): beam_size = int(b_val)

    # 4. 进入交互循环
    print("\n" + "="*30)
    print(f"系统就绪！[架构: {model_type.upper()} | 策略: {method.upper()}]")
    print("输入 'q' 或 'exit' 退出程序。")
    print("="*30 + "\n")

    while True:
        sentence = input("User (EN): ").strip()
        if not sentence: continue
        if sentence.lower() in ['q', 'exit']: break

        try:
            translation = translate_sentence(
                sentence, model, src_vocab, tgt_vocab, args.device,
                method=method, max_len=80, beam_size=beam_size
            )
            print(f"Model (ZH): {translation}")
            print("-" * 20)
        except Exception as e:
            print(f"翻译出错: {e}")

    print("\n[Info] Demo 已退出。")

if __name__ == "__main__":
    main()