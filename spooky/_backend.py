import os

# Read from environment variable, default is 'numpy'
BACKEND = os.environ.get('NUMPY_BACKEND', 'numpy').lower()

# --- Backend Import and Aliasing ---
if BACKEND == 'jax':
    try:
        import jax.numpy as xnp
        print("Using JAX backend.")
        JAX_ENABLED = True
    except ImportError:
        print("JAX requested but not installed. Falling back to NumPy.")
elif BACKEND == 'numpy':
    import numpy as xnp
    print("Using NumPy backend.")
    JAX_ENABLED = False
else:
    raise ValueError(f"Unknown backend: {BACKEND}. Must be 'numpy' or 'jax'.")

def _jax_index_update(arr, indices, values=0.0):
    """JAX-specific update (immutable, array.at)."""
    # The 'arr' is guaranteed to be a JAX array here
    return arr.at[indices].set(values)

def _numpy_index_update(arr, indices, values=0.0):
    """NumPy-specific update (uses copy for functional behavior)."""
    # The 'arr' is guaranteed to be a NumPy array here
    arr_copy = arr.copy()
    arr_copy[indices] = values
    return arr_copy

# Define backend functions
if JAX_ENABLED:
    index_update = _jax_index_update

    from jax import jit, grad
    from functools import partial
    def apply_jit(func):
        return partial(jit, static_argnums=(0,))(func)

    def copy_arr(arr):
        return arr
else:
    index_update = _numpy_index_update

    def jit(func):
        return func
    def grad(func):
        raise NotImplementedError("Grad is only available with JAX backend.")
    def apply_jit(func):
        return func

    def copy_arr(arr):
        return xnp.copy(arr)
