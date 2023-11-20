import torch

# Define the sinkhorn_batched function
def sinkhorn_batched(x, y, p=2, eps=1e-3, max_iters=100, stop_thresh=1e-5):
    # Ensure we're working with float32 data (required for PyKeOps)
    x, y = x.float(), y.float()

    # Reshape x and y to [batch_size * seq_len, input_dim] and [batch_size * seq_len * num_experts, output_dim]
    batch_size, seq_len, input_dim = x.shape
    _, _, num_experts, output_dim = y.shape
    x = x.view(-1, input_dim)
    y = y.view(-1, output_dim)

    # Initialize weights if not provided
    w_x = torch.ones(x.size(0), device=x.device) / x.size(0)
    w_y = torch.ones(y.size(0), device=y.device) / y.size(0)

    # Compute the cost matrix for the batch
    C = torch.cdist(x.unsqueeze(1), y.unsqueeze(0), p=p).squeeze()

    # Initialize Sinkhorn iterations
    u = torch.zeros_like(w_x)
    v = torch.zeros_like(w_y)
    for _ in range(max_iters):
        u_prev = u.clone()
        v_prev = v.clone()

        # Clamp values to avoid numerical issues
        C_clamped = torch.clamp(C, min=-1e5, max=1e5)
        u = eps * torch.log(torch.clamp(w_x, min=1e-9)) - torch.logsumexp(((-C_clamped + v.unsqueeze(0)) / eps), dim=1)
        v = eps * torch.log(torch.clamp(w_y, min=1e-9)) - torch.logsumexp(((-C_clamped + u.unsqueeze(1)) / eps), dim=0)

        # Check for convergence
        if torch.max(torch.abs(u - u_prev)) < stop_thresh and torch.max(torch.abs(v - v_prev)) < stop_thresh:
            break

    # Check for any NaNs in the output
    if torch.isnan(u).any() or torch.isnan(v).any():
        raise ValueError("NaN values detected in Sinkhorn iterations.")

    # Recover the transport plan
    P = torch.exp(u.unsqueeze(1) + v.unsqueeze(0) - C / eps)

    # Sum the transport plan over the experts dimension to get the routing weights
    routing_weights = P.view(batch_size, seq_len, num_experts, -1).sum(dim=-1)

    # Normalize the routing weights to avoid NaNs in the softmax step
    routing_weights = torch.softmax(routing_weights, dim=-1)

    return routing_weights

# Dummy data with the specified dimensions
batch_size = 2
seq_len = 3
input_dim = 4
num_experts = 5
output_dim = 4

# Input tensor x with shape [batch_size, seq_len, input_dim]
x = torch.randn(batch_size, seq_len, input_dim)

# Experts tensor bax with shape [batch_size, seq_len, num_experts, output_dim]
bax = torch.randn(batch_size, seq_len, num_experts, output_dim)

# Define parameters for the Sinkhorn operation
p = 2
eps = 0.1
max_iters = 50
stop_thresh = 1e-4

# Call the sinkhorn_batched function
routing_weights = sinkhorn_batched(x, bax, p=p, eps=eps, max_iters=max_iters, stop_thresh=stop_thresh)

# Show the routing weights
print(routing_weights)
