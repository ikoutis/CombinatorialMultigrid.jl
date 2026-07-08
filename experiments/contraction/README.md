# Combinatorial Laplacian contraction — isolated experiment

**Status: experiment only.** Nothing here is wired into the solver or the
hierarchy build; the code is self-contained in this directory.

## The operation

Each level of the CMG hierarchy contracts the current Laplacian/SDD matrix
with a sparse triple product (`src/cmgAlg.jl`, `build_hierarchy`):

```julia
Rt = sparse(cI, 1:n, 1, nc, n)      # restriction: fine node i -> cluster cI[i]
A_ = Rt * A * Rt'                    # two sparse matmuls
```

Algebraically `A_[C, D] = Σ_{i∈C, j∈D} A[i, j]`. Combinatorially that is just:

- every fine edge `(i, j, w)` with `cI[i] ≠ cI[j]` adds `w` to the coarse edge
  `(cI[i], cI[j])` — weights between clusters **sum**;
- intra-cluster edges cancel out of the off-diagonals and fold into the coarse
  diagonal, together with any SDD excess of the cluster's rows.

So a single linear pass over the edges with an accumulator suffices — no
intermediate `Rt*A` product, no SpGEMM machinery.

## The three variants (`contract.jl`)

| routine | idea | cost |
|---|---|---|
| `contract_matmul` | the verbatim build expression (baseline) | two SpGEMMs |
| `contract_coo` | one CSC pass emitting mapped triplets `(cI[i], cI[j], v)`, then one `sparse(I,J,V)` (sums duplicates linearly) | `O(nnz)` + 3 nnz-length temporaries |
| `contract_spa` | counting-sort fine columns into coarse buckets, then per coarse column dedupe with a mark/slot sparse accumulator, building the coarse CSC directly (columns sorted via a reusable pair buffer) | `O(n + nc + nnz)` + per-column sorts of the output |

All three return the identical `nc × nc` `SparseMatrixCSC` (up to
floating-point summation order).

## Run

```bash
# correctness (grids 2D/3D, Laplacian, near-tree, random SDD; random/identity/
# all-in-one clusterings and CMG's real steiner_group clustering):
julia --project=. experiments/contraction/test_contraction.jl

# runtime comparison (min-of-3, warmup + GC between runs; --quick for small sizes):
julia --project=. experiments/contraction/bench_contraction.jl
julia --project=. experiments/contraction/bench_contraction.jl --quick
```

The benchmark verifies agreement per configuration before timing and prints
`matmul / coo / spa` seconds plus speedups over the matmul baseline, across 2D
grids (1M/4M nodes), a 3D grid (1.7M), and a 1M-node near-tree, each with
random clusterings at coarsening ratios 2/8/32 and with the clustering CMG's
`steiner_group` actually produces.

## If the numbers confirm the hypothesis

`build_hierarchy` calls this contraction once per level; replacing
`Rt * A * Rt'` with `contract_spa` would be a small, separate change (the
routine is a drop-in for that line). Parallelizing the per-coarse-column loop
is another later option — columns are independent given the buckets.

## Validation provenance

The `contract_spa` algorithm (same bucket + mark/slot design) was
cross-validated against `Rt * L * Rt'` in a scipy harness at machine precision
across the full case battery before being ported here.
