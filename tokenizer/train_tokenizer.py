import os
import sentencepiece as spm
from tokens import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN


def train_tokenizer(input_file, vocab_size=16_000, delete_input=False):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=f"tokenizer_{vocab_size}",
        vocab_size=vocab_size,
        model_type="bpe",
        byte_fallback=True,
        num_threads=os.cpu_count(),
        allow_whitespace_only_pieces=True,
        pad_id=3,
        unk_id=4,
        bos_id=1,
        eos_id=2,
        pad_piece=PAD_TOKEN,
        unk_piece=UNKNOWN_TOKEN,
        bos_piece=BOS_TOKEN,
        eos_piece=EOS_TOKEN,
        normalization_rule_name="identity",
    )

    if delete_input:
        os.remove(input_file)
        print(f"Deleted input file: {input_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=16_384)
    parser.add_argument("--input_file", type=str, default="train.txt")
    parser.add_argument("--delete-corpus", type=str)

    args = parser.parse_args()
    train_tokenizer(
        vocab_size=args.vocab_size,
        input_file=args.input_file,
    )
