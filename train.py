from model import Llama3
from config import CONFIGS
from data.dataset import PretokenizedDataset
from torch.utils.data import DataLoader
import torch
import argparse
import time
from math import cos, pi

def get_lr(step, warmup_steps, decay_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    if step > decay_steps:
        return min_lr
    ratio = (step - warmup_steps) / (decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + cos(pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def train(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = CONFIGS.get(args.config.lower())
    if not config:
        raise ValueError(f"invalid config {config}, must be one of {",".join(CONFIGS.keys())}")

    dataset = PretokenizedDataset(args.corpus_bin, config.context_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=0,
        pin_memory=True,
    )

    val_datset = PretokenizedDataset(args.corpus_val_bin, config.context_length)
    val_loader = DataLoader(
        val_datset,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=0,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Llama3(config)
    model.to(device)
    optimizer = model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    start_step = 0
    
    if args.init_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]

    total_batch_size = args.total_batch_size
    grad_accum_steps = total_batch_size // (args.batch_size * config.context_length)

    assert total_batch_size % (args.batch_size * config.context_length) == 0
    n_steps = args.n_steps
    if n_steps is None:
        n_steps = len(dataset) // args.batch_size
    warmup_steps = int(n_steps * 0.01)


    context = torch.amp.autocast(device_type=device, dtype=torch.float16)
    min_val_loss = 10e99
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(start_step, n_steps):
        if step > 0 and step % args.val_every == 0:
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for _ in range(args.val_steps):
                    x, y = next(iter(val_loader))
                    x, y = x.to(device), y.to(device)

                    logits, loss = model(x, y)
                    val_loss += loss
                val_loss /= args.val_steps
            print(f"val loss: {val_loss}")
            if val_loss < min_val_loss:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                    "min_val_loss": min_val_loss,
                    "args": args
                }
                torch.save(checkpoint, args.checkpoint_path)
                min_val_loss = val_loss

        model.train()
        optimizer.zero_grad()

        loss_train = 0.0
        start = time.time()

        lr = get_lr(step, warmup_steps=warmup_steps, decay_steps=n_steps, lr=args.learning_rate, min_lr=args.learning_rate * 0.1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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

        print(f"step {step+1:4d}/{n_steps} | loss: {loss_train:.6f} | lr: {lr:.2e} | time: {end - start:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=CONFIGS.keys())
    parser.add_argument("--corpus-bin", type=str, required=True, help="path to train dataset .bin file")
    parser.add_argument("--corpus-val-bin", type=str, required=True, help="path to val dataset .bin file")
    parser.add_argument("--val-every", type=int, help="how many steps for val loss", default=100)
    parser.add_argument("--val-steps", type=int, default=20)
    parser.add_argument("--init-checkpoint", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoint.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--total-batch-size", type=int, default=2**17)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--weight-decay", type=float, default=0.1)

    args = parser.parse_args()
    train(args)
