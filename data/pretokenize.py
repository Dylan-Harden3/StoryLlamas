import argparse
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import sentencepiece as spm


def pretokenize(dataset, tokenizer, output_file):
    examples = []
    for example in tqdm(dataset):
        story = example["text"].strip()
        token_ids = tokenizer.encode_as_ids(story, add_bos=True)
        examples.extend(token_ids)

    corpus = np.array(examples, dtype=np.uint16)
    with open(output_file, "wb") as f:
        f.write(corpus.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, help="output .bin file")
    parser.add_argument("--model_file", type=str, help="tokenizer .model file")
    parser.add_argument("--dataset_file", type=str, help="which file from hf to use")
    args = parser.parse_args()

    dataset = load_dataset("roneneldan/TinyStories", data_files=args.dataset_file, split="train")
    tokenizer = spm.SentencePieceProcessor(model_file=args.model_file)

    pretokenize(dataset, tokenizer, args.output_file)
