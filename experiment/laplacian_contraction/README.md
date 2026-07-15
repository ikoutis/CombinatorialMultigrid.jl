# Experimental Laplacian Contraction

This directory contains a standalone experiment for replacing the algebraic
coarse-level contraction

```julia
Rt = sparse(cI, 1:n, 1.0, nc, n)
L_coarse = Rt * L * Rt'
```

with a sequential combinatorial contraction routine.

The experiment is intentionally **not wired into the solver**. It is meant only
to confirm correctness and compare runtime/allocation behavior against the two
sparse matrix multiplications above.

## Combinatorial rule

Given a fine Laplacian `L` and a cluster id `cI[i]` for each fine vertex `i`, the
coarse graph has one vertex per cluster. For every fine edge `(i, j)` with weight
`w = -L[i,j]`:

- if `cI[i] == cI[j]`, the edge is internal to a contracted cluster and
  disappears;
- otherwise, add weight `w` to the coarse edge between clusters `cI[i]` and
  `cI[j]`.

The resulting coarse Laplacian has off-diagonal entries equal to negative summed
inter-cluster weights, and diagonal entries equal to the sum of incident coarse
edge weights.

## Files

- `combinatorial_contraction.jl`: standalone contraction routines.
- `run_experiment.jl`: correctness checks plus a simple runtime/allocation
  comparison against `Rt * L * Rt'`.

## Running

From the repository root:

```bash
julia experiment/laplacian_contraction/run_experiment.jl
```

Optional positional arguments set the timing problem size:

```bash
julia experiment/laplacian_contraction/run_experiment.jl <n> <random_edges> <clusters> <reps>
```

For example:

```bash
julia experiment/laplacian_contraction/run_experiment.jl 50000 200000 5000 7
```

The script reports correctness errors, median elapsed time, and median allocated
memory for both the combinatorial contraction and the sparse-matmul baseline.
