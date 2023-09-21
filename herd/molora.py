import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        super(Router, self).__init__()
        self.linear = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        router_logits = self.linear(x)
        router_probs = F.softmax(router_logits, dim=-1)
        return router_probs

class MoLoRa(nn.Module):
    def __init__(self, hidden_dim, rank=4, num_experts=1, output_dim=None, alpha=16):
        super(MoLoRa, self).__init__()

        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_experts = num_experts
        self.output_dim = output_dim if output_dim else hidden_dim
        self.alpha = alpha

        # Define the router
        self.router = Router(hidden_dim, num_experts)

        # LORA A and B matrices
        self.lora_A = nn.Parameter(torch.randn(num_experts, hidden_dim, rank) * 2e-2)
        self.lora_B = nn.Parameter(torch.zeros(num_experts, rank, self.output_dim))

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]

        # Compute ax using einsum
        ax = torch.einsum('bsd,edr->bser', x, self.lora_A)

        # Compute bax using einsum
        bax = torch.einsum('bser,erd->bsed', ax, self.lora_B)

        # Get router probabilities
        router_probs = self.router(x)

        # Combine using router probabilities
        output = torch.einsum('...e,...ed->...d', router_probs, bax)

        return output * (self.alpha / self.rank)
