from model import DyLLM
from config import DYLLM_CONFIG_2M
from data.dataset import PretokenizedDataset
from torch.utils.data import DataLoader
import torch

batch_size = 256
context_length = DYLLM_CONFIG_2M.context_length

dataset = PretokenizedDataset("data/corpus.bin", context_length)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    sampler=None,
    num_workers=0,
    pin_memory=True,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = DyLLM(DYLLM_CONFIG_2M)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=True)

TOTAL_BATCH_SIZE = 2**17
grad_accum_steps = TOTAL_BATCH_SIZE // (batch_size * context_length)

assert TOTAL_BATCH_SIZE % (batch_size * context_length) == 0

n_steps = len(dataset) // batch_size

model.to(device)

context = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

for step in range(n_steps):
    model.train()
    loss_train = 0.0

    for grad_accum_step in range(grad_accum_steps):
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        
        with context:
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
        loss_train += loss
        scaler.scale(loss).backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    print(f"step {step}/{n_steps} | loss: {loss_train:.5f}")

torch.save(model.state_dict(), "model.bin")