from model import DyLLM
from config import CONFIGS
from data.dataset import PretokenizedDataset
from torch.utils.data import DataLoader
import torch
import argparse
import time


def train(args):
    config = CONFIGS.get(args.config.lower())
    if not config:
        raise ValueError(
            f"invalid config {config}, must be one of {",".join(CONFIGS.keys())}"
        )

    dataset = PretokenizedDataset(args.corpus_path, config.context_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=0,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DyLLM(config)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=args.learning_rate
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    total_batch_size = args.total_batch_size
    grad_accum_steps = total_batch_size // (args.batch_size * config.context_length)

    assert total_batch_size % (args.batch_size * config.context_length) == 0

    n_steps = len(dataset) // args.batch_size

    model.to(device)

    context = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

    for step in range(n_steps):
        model.train()
        loss_train = 0.0
        start = time.time()

        for grad_accum_step in range(grad_accum_steps):
            x, y = next(iter(loader))
            x, y = x.to(device), y.to(device)

            with context:
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps

            loss_train += loss
            scaler.scale(loss).backward()

        end = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        print(
            f"step {step}/{n_steps} | loss: {loss_train:.5f} | time: {end - start:2f}"
        )

    output_path = args.output_path
    if output_path is None:
        output_path = f"model_{args.config}.bin"

    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=CONFIGS.keys())
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="path to pretokenized dataset .bin file",
    )
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--total-batch-size", type=int, default=2**17)
    parser.add_argument("--learning-rate", type=float, default=8e-4)

    args = parser.parse_args()
    train(args)
