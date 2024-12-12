import argparse
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokens import SPECIAL_TOKENS, UNKNOWN_TOKEN


def load_text(data_files="TinyStories-train.txt"):
    dataset = load_dataset(
        "roneneldan/TinyStories", data_files=data_files, split="train"
    )
    return [row["text"] for row in dataset]


def train_tokenizer(texts, vocab_size=16_000):
    tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN, ignore_merges=True))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=16_000)
    parser.add_argument("--output", type=str, default="tokenizer.json")

    args = parser.parse_args()

    texts = load_text()
    tokenizer = train_tokenizer(texts, args.vocab_size)
    tokenizer.save(args.output)
    print(f"Tokenizer saved to {args.output}")
