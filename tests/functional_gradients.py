# %%
import jax
import jax; jax.config.update('jax_platforms', 'cpu')
import jax.numpy as np
import jax.tree_util

def f(a, b, x):
  return x*a*b

def nested_default(x, f_a):
  return f_a(x)

def top_level(a, b):
  f_a = jax.tree_util.Partial(f, a, b)
  x = 3.0
  return nested_default(x, f_a)

# %%
if __name__ == "__main__":
  print(jax.grad(top_level, argnums=(0,1))(2.0, 3.0))

@jax.custom_jvp
def nested(x, f_a):
  return f_a(x)

@nested.defjvp
def nested_jvp(primals, tangents):
  x, f_a = primals
  f = f_a.func # unpartial the function

  a_flat, _ = jax.tree.flatten(f_a)

  x_dot, f_a_dot = tangents
  a_dot_flat, _ = jax.tree.flatten(f_a_dot)

  primal_out, tangent_out = jax.jvp(f, (*a_flat, x), (*a_dot_flat, x_dot))
  return primal_out, tangent_out

def top_level(a, b):
  f_a = jax.tree_util.Partial(f, a, b)
  x = 3.0
  return nested_default(x, f_a)

if __name__ == "__main__":
  print(jax.grad(top_level, argnums=(0,1))(2.0, 3.0))