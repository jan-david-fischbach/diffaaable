# %%
import jax
import jax.numpy as np
import jax.tree_util

def f(x, a):
  return x*a

def nested(x, f_a):
  return f_a(x)

def top_level(a):
  f_a = jax.tree_util.Partial(f, a = a)
  x = 3.0
  return nested(x, f_a)

# %%
if __name__ == "__main__":
  print(jax.grad(top_level)(2.0))
# %%
