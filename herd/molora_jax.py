from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import partitioning
from flaxformer.types import Array, Initializer, DType

from src import routing
from src import expert_dense as dense


class MoLoRa(nn.Module):

  router: routing.Router
  rank: int = 4
  lora_init_A: Initializer = nn.initializers.normal(stddev=2e-2)
  lora_init_B: Initializer = nn.initializers.zeros
  lora_axis_names_A: Sequence[str] = ('unmodeled', 'mlp', 'unmodeled')
  lora_axis_names_B: Sequence[str] = ('unmodeled', 'unmodeled', 'mlp')
  alpha = 16
  num_experts: int = 1
  dtype: DType = jnp.float32
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
  axis: Union[Iterable[int], int] = -1
  reshape_kernel: bool = True
  output_dim: Optional[int] = None

  @nn.compact
  def __call__(self, x: Array, **kwargs) -> Array:

    *rest, hidden = x.shape

    #x = jax.lax.convert_element_type(x, self.dtype)

    #[num_experts, hidden, rank]
    molora_a = partitioning.param_with_axes(
              'lora_A',
              self.lora_init_A,
              (self.num_experts, hidden, self.rank),
              jnp.float32,
              axes=self.lora_axis_names_A)

    molora_a = jax.lax.convert_element_type(molora_a, self.dtype)

    #[batch, seq_len, num_experts, rank]
    ax = jnp.einsum('bsd,edr->bser',
                         x,
                         molora_a)

    # Add expert axis name to the partitioning axes
    ax = partitioning.with_sharding_constraint(ax, ('batch', 'length', 'expert', 'rank'))
    ax = jax.lax.convert_element_type(ax, self.dtype)

    #[num_experts, rank, output_dim]
    molora_b = partitioning.param_with_axes(
              'lora_B',
              self.lora_init_B,
              (self.num_experts, self.rank, (self.output_dim if self.output_dim else hidden)),
              jnp.float32,
              axes=self.lora_axis_names_B)

    molora_b = jax.lax.convert_element_type(molora_b, self.dtype)

    #[batch, seq_len, num_experts, rank]
    bax = jnp.einsum('bser,erd->bsed',
                         ax,
                         molora_b)

    bax = partitioning.with_sharding_constraint(bax, ('batch', 'length', 'expert') + tuple([self.lora_axis_names_B[-1]]))
    bax = jax.lax.convert_element_type(bax, self.dtype)

    #[batch, seq_len, num_experts]
    router_probs, _, _  = self.router(x, self.num_experts)
    router_probs = partitioning.with_sharding_constraint(router_probs,
                                                         ('batch', 'length', 'expert'))

    #[batch, seq_len, hidden_dim]
    bax = jnp.einsum('...e,...ed->...d',
                         router_probs,
                         bax)

    return bax * (self.alpha / self.rank)