from config import CONFIGS
from model import DyLLM
import torch
import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, choices=CONFIGS.keys())
parser.add_argument("--checkpoint-path", type=str, required=True)
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--top-p", type=float, default=0.95)
parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer_4096.model")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DyLLM(CONFIGS[args.config.lower()])
tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)

checkpoint = torch.load(args.checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)

token_ids = model.generate(args.prompt, tokenizer, args.max_tokens, args.top_k, args.top_p, args.temperature, device=device)

print(tokenizer.decode(token_ids[0].tolist()))