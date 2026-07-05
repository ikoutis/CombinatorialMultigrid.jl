
# CombinatorialMultigrid.jl


This package implements the Combinatorial Multigrid Preconditioner *[1]*. The code handles input matrices  that are symmetric diagonally dominant with negative off-diagonal entries, a class of matrices that includes graph Laplacians. This work has been supported by NSF grant CCF-#1149048.


In order to install the package simply do the following: 
``` 
Pkg.add("CombinatorialMultigrid") 
using CombinatorialMultigrid
```

We present a quick example. Lets load a fairly large example matrix ```X``` from the ```example``` directory and build the ```b``` side. ```X``` is an adjacency matrix. ```LX``` is the corresponding Laplacian matrix. 

```
## load example matrix
file = matopen("../example/X.mat"); X = read(file, "X"); close(file)
LX = lap(X);
b = rand(Float64, size(X, 1));
b = b .- sum(b)/length(b);
```
CMG outputs a preconditioner function that can be used to solve a linear system in the input system. This is done here:

```
t = @elapsed (pfunc, h) = cmg_preconditioner_lap(LX);
@info "Time Required to build CMG Solver: $(t) seconds"
t = @elapsed x = pfunc(b);
@info "Runtime of preconditioner: $(t) seconds"
```

The second output ```h``` is a hierarchy of graphs that is implicitly used in ```pfunc``` and it is exposed for its potential in other applications. Alternatively, one can bypass the computation of the Laplacian matrix as follows:

```
(pfunc, h) = cmg_preconditioner_adj(X);
```

The above script generates the following output: 
```
[ Info: Time Required to build CMG Solver: 3.258120718 seconds
[ Info: Runtime of preconditioner: 0.194163587 seconds
```
We now solve a linear system using CMG. For this purpose we leverage ```pcg``` from the ```Laplacians``` package. We run the following script: 
```
f = pcgSolver(LX,pfunc);
t = @elapsed x = f(b, maxits=40, tol=1e-6,verbose=true);
@info "Time Required to solve system: $(t) seconds"
```
This generates the following output: 
```
PCG stopped after: 7.828 seconds and 29 iterations with relative error 8.429320186485909e-7.
[ Info: Time Required to solve system: 8.034237971 seconds
```

For comparison we run ```approxchol_lap``` which is the fastest solver from the ```Laplacians``` package. We run the following script: 
```
solver = approxchol_lap(X; tol=1e-6, maxits=1000, maxtime=Inf, verbose=true, pcgIts=Int[], params=ApproxCholParams());
@info "Time Required to build Lap Solver: $(t) seconds"
t = @elapsed x = solver(b);
@info "Time Required to find x: $(t) seconds"
```
This generates the following output: 
```
Using greedy degree ordering. Factorization time: 25.985280990600586
Ratio of operator edges to original edges: 4.548086602013639
ratio of max to min diagonal of laplacian : 583453.0510646806
Solver build time: 26.293 seconds.
[ Info: Time Required to build Lap Solver: 26.304803977 seconds
PCG stopped after: 10.288 seconds and 26 iterations with relative error 9.697886904926194e-7.
[ Info: Time Required to find x: 12.226966502 seconds
```

```CMG``` builds the preconditioner in ```3.26 seconds``` compared to ```26.31 seconds``` with ```approxchol_lap```. Solving the system takes ```7.8 seconds``` with ```cmg``` and ```10.29 seconds``` with ```approxchol_lap```.

## K-cycle solver (optional)

The package also provides a **K-cycle** variant of the CMG preconditioner: the
stationary repeats of the classic cycle are replaced by a few inner
flexible-CG (Krylov) iterations at each coarse level, preconditioned by the
recursive K-cycle below. On anisotropic and high-contrast problems this
typically reduces both iterations and wall time. The legacy cycle remains the
default everywhere; the K-cycle is selected explicitly.

The simplest entry point is the new solver `cmg_solve`, which wraps a
flexible-CG outer loop around the CMG hierarchy:

```julia
(x, stats) = cmg_solve(LX, b)                     # K-cycle by default
(x, stats) = cmg_solve(LX, b; cycle = :vcycle)    # classic cycle (plain PCG)
stats.iterations, stats.relres, stats.converged
```

To reuse a hierarchy across right-hand sides, build it once and pass it in:

```julia
(pfunc, h) = cmg_preconditioner_lap(LX);          # legacy default, unchanged
(x, stats) = cmg_solve(h, b);                     # K-cycle solve on the same hierarchy
```

Knobs of `cmg_solve`:

| keyword | default | meaning |
|---|---|---|
| `cycle` | `:kcycle` | `:kcycle` or `:vcycle` (stationary legacy cycle) |
| `tol` | `1e-8` | relative residual tolerance |
| `maxit` | `500` | maximum outer iterations |
| `theta` | `0.75` | work-budget cap for the inner iterations; `0.0` opts out into fixed local repeats |
| `inner_tol` | `0.25` | adaptive early-stopping for the inner FCG; `0.0` disables |
| `collect_stats` | `false` | record per-level K-cycle visit counts in `stats.level_visits` |

A single K-cycle application is also available from the preconditioner
constructors via a knob:

```julia
(pfunc_k, h) = cmg_preconditioner_lap(LX; cycle = :kcycle);
x ≈ pfunc_k(b)   # one K-cycle sweep
```

> **Warning.** The K-cycle operator is *nonlinear* (it runs inner Krylov
> iterations), so `pfunc_k` must **not** be passed as a preconditioner to
> `Laplacians.pcgSolver` or any other standard PCG — use `cmg_solve`, whose
> flexible-CG outer loop supports it. The default (`cycle = :vcycle`)
> preconditioner remains a fixed linear operator and stays PCG-safe.

`example/bench_kcycle.jl` compares the two cycles on uniform and anisotropic
grids.

**Citations:**

[1] Ioannis Koutis, Gary L. Miller, David Tolliver, Combinatorial preconditioners and multilevel solvers for problems in computer vision and image processing, Computer Vision and Image Understanding, Volume 115, Issue 12, 2011, Pages 1638-1646, ISSN 1077-3142, https://doi.org/10.1016/j.cviu.2011.05.013.*
