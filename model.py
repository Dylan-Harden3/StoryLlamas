import torch.nn as nn
from config import LlamaArgs
from transformer.transformer_block import TransformerDecoderBlock
from transformer.rope import precompute_freqs_cis
from transformer.normalization import RMSNorm
import torch
from typing import Optional
import torch.nn.functional as F
import math


class Llama3(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.args = args
        self.token_embeddings = nn.Embedding(args.vocab_size, args.model_dim)

        self.transformer_blocks = nn.ModuleList()
        for i in range(args.num_layers):
            self.transformer_blocks.append(TransformerDecoderBlock(i, args))

        self.norm = RMSNorm(args.model_dim, args.norm_epsilon)
        self.classifier = nn.Linear(args.model_dim, args.vocab_size, bias=False)
        self.classifier.SKIP_INIT = 1
        self.token_embeddings.weight = self.classifier.weight
        self.apply(self._init_weights)

        for param_name, param in self.named_parameters():
            if param_name.endswith("w3.weight") or param_name.endswith("head_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * args.num_layers))

        freqs_cos, freqs_sin = precompute_freqs_cis(args.model_dim // args.num_heads, args.context_length)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if not hasattr(module, "SKIP_INIT"):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        _, context_length = x.shape
        h = self.token_embeddings(x)

        freqs_cos, freqs_sin = (
            self.freqs_cos[:context_length],
            self.freqs_sin[:context_length],
        )

        for transformer in self.transformer_blocks:
            h = transformer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        logits = self.classifier(h)
        
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        return logits, loss

    def configure_optimizers(self, learning_rate, weight_decay):
        params = {name: param for name, param in self.named_parameters() if param.requires_grad}

        decay_params = [param for _, param in params.items() if param.dim() >= 2]
        nodecay_params = [param for _, param in params.items() if param.dim() < 2]

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")

        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer
    
    @torch.no_grad()
    def generate(self, prompt, tokenizer, max_length=256, top_k=50, top_p=0.95, temperature=1.0, eos_id=None, device="cpu"):
        tokens = torch.tensor(tokenizer.encode_as_ids(prompt, add_bos=True)).reshape(1, -1).to(device)
        self.eval()

        for _ in range(max_length - tokens.size(0)):
            seq = tokens[:, -self.args.context_length:]
            logits, _ = self.forward(seq)
            logits = logits [:, -1, :] / temperature

            if top_k > 0:
                topk_probs, topk_indices = torch.topk(logits, top_k, dim=-1)
                topk_probs = F.softmax(topk_probs, dim=-1)
                sampled_index = torch.multinomial(topk_probs, 1)
                sampled_token = topk_indices.gather(-1, sampled_index)
            elif top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_logits[sorted_indices_to_remove] = -float('Inf')

                probs = F.softmax(sorted_logits, dim=-1)
                sampled_token = torch.multinomial(probs, 1)
                sampled_token = sorted_indices.gather(-1, sampled_token)
            else:
                probs = F.softmax(logits, dim=-1)
                sampled_token = torch.multinomial(probs, 1)

            if eos_id is not None and sampled_token == eos_id:
                break
            
            if tokens[:, -3:].reshape(-1).tolist() == [385, 387, 384]:
                tokens = tokens[:, :-3]
                break
            
            tokens = torch.cat((tokens, sampled_token), dim=-1)
        
        return tokens