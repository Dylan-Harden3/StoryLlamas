import sentencepiece as spm
from tqdm import tqdm


def encode_line(tokenizer, line):
    tokens = tokenizer.encode_as_ids(line)
    num_tokens = len(tokens)
    num_chars = len(line)
    return num_tokens, num_chars


def get_compression_ratio(tokenizer, input_file):
    total_tokens = 0
    total_chars = 0

    lines = []
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Counting tokens", unit="line"):
        num_tokens, num_chars = encode_line(tokenizer, line.strip())
        total_tokens += num_tokens
        total_chars += num_chars

    return total_tokens, total_chars


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, default="train.txt")

    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor(model_file=args.model_file)

    total_tokens, total_chars = get_compression_ratio(tokenizer, args.input_file)

    print(
        f"Total Characters: {total_chars} Total Tokens: {total_tokens} Compression Ratio: {total_chars/total_tokens}"
    )
