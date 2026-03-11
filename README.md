Collection of algorithms and tools based on pseudospectral methods for fluid flows

- solvers: Collection of time marching solvers
    - Kuramoto-Sivashinsky 1D
    - Kolmogorov Flow 2D
    - Wrappers for GHOST and [SPECTER](https://github.com/specter-cfd/SPECTER)
- DynSys: 
    - UPOs: calculates unstable periodic orbits using a Newton shooting method.
    - Floquet/Lyapunov analysis: calculate exponents.

If Jax is installed, the backend can be changed by setting the environment
variable
$ export NUMPY_BACKEND='jax'

By default, JAX uses float, this can be changed by doing
$ export JAX_ENABLE_X64 = 1
This is mandatory for DynSys routines, but makes running in GPUs painfully slow
    

### Installation
```
pip install spookyflows
```
