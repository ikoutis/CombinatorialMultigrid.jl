# near_tree_demo.jl
#
# Self-contained demo of the degree-1/2 elimination branch on artificial
# "near-tree" Laplacians (a spanning tree plus a few off-tree edges), which
# mimic the structure of the Spielman IPM graphs.
#
# For each instance it reports the surviving "core" size after exact degree-1/2
# elimination, and compares plain CMG against CMG-with-elimination on both build
# time and solve time (K-cycle), plus iterations and the true relative residual.
#
# Run from the repo root:
#     julia --project=. example/near_tree_demo.jl
# Optionally pass "n:offtree" pairs to override the default sizes:
#     julia --project=. example/near_tree_demo.jl 50000:50 200000:200

using Random
using SparseArrays
using LinearAlgebra
using Laplacians
using CombinatorialMultigrid
using Printf

# ---------------------------------------------------------------------------
# Build a near-tree graph Laplacian: a random weighted spanning tree on n nodes
# plus `n_off` random off-tree edges. Returns lap(adjacency).
# ---------------------------------------------------------------------------
function near_tree_lap(n::Int, n_off::Int; seed::Int = 1, wlo = 0.5, whi = 2.0)
    Random.seed!(seed)
    local I = Int[]
    local J = Int[]
    local V = Float64[]
    sizehint!(I, 2 * (n - 1 + n_off))
    sizehint!(J, 2 * (n - 1 + n_off))
    sizehint!(V, 2 * (n - 1 + n_off))
    randw() = wlo + (whi - wlo) * rand()

    # spanning tree: attach node v to a uniformly random earlier node
    for v = 2:n
        p = rand(1:v-1)
        w = randw()
        push!(I, v); push!(J, p); push!(V, w)
        push!(I, p); push!(J, v); push!(V, w)
    end
    # off-tree edges (a few, chosen at random)
    added = 0
    while added < n_off
        i = rand(1:n); j = rand(1:n)
        if i != j
            w = randw()
            push!(I, i); push!(J, j); push!(V, w)
            push!(I, j); push!(J, i); push!(V, w)
            added += 1
        end
    end
    A = sparse(I, J, V, n, n)   # sparse() sums any accidental parallel edges
    return lap(A)
end

# ---------------------------------------------------------------------------
# Time build + solve for plain CMG and for CMG-with-elimination on the same L.
# ---------------------------------------------------------------------------
function run_instance(n::Int, n_off::Int; cycle::Symbol = :kcycle, tol = 1e-8, maxit = 500)
    local L = near_tree_lap(n, n_off)
    local b = randn(n); b .-= sum(b) / n           # rhs in the Laplacian range

    # core size after exact degree-1/2 elimination
    local elims, ind, A_red, is_lap = CombinatorialMultigrid.eliminate_deg12(L)
    local core = length(ind)

    # --- plain CMG ---
    local tb_plain = @elapsed (_, H) = cmg_preconditioner_lap(L; cycle = cycle)
    local ts_plain = @elapsed (x0, s0) = cmg_solve(H, b; cycle = cycle, tol = tol, maxit = maxit)
    local err_plain = norm(L * x0 - b) / norm(b)

    # --- CMG with elimination ---
    local tb_elim = @elapsed (_, EH) = cmg_preconditioner_lap(L; cycle = cycle, eliminate = true)
    local ts_elim = @elapsed (xe, se) = cmg_solve(EH, b; cycle = cycle, tol = tol, maxit = maxit)
    local err_elim = norm(L * xe - b) / norm(b)

    return (; n, n_off, core,
            tb_plain, ts_plain, its_plain = s0.iterations, conv_plain = s0.converged, err_plain,
            tb_elim, ts_elim, its_elim = se.iterations, conv_elim = se.converged, err_elim)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# default sizes (n, off-tree edges); override via ARGS as "n:offtree"
default_sizes = [(10_000, 10), (50_000, 50), (100_000, 100), (50_000, 0)]
sizes = isempty(ARGS) ? default_sizes :
        [(parse(Int, split(a, ':')[1]), parse(Int, split(a, ':')[2])) for a in ARGS]

println("BLAS threads: ", BLAS.get_num_threads(), "   (cycle = :kcycle, tol = 1e-8, maxit = 500)\n")

# warm up compilation on a tiny instance so timings exclude JIT
print("warming up (JIT)… "); flush(stdout)
run_instance(2_000, 5)
println("done\n")

@printf("%9s %7s %8s %7s | %9s %9s %6s %5s %9s | %9s %9s %6s %5s %9s | %6s\n",
        "n", "offtree", "core", "core%",
        "build_p", "solve_p", "its_p", "cvg", "relres_p",
        "build_e", "solve_e", "its_e", "cvg", "relres_e", "solve↑")
println(repeat("-", 150))

for (n, n_off) in sizes
    r = run_instance(n, n_off)
    speedup = r.ts_elim > 0 ? r.ts_plain / r.ts_elim : Inf
    @printf("%9d %7d %8d %6.2f | %9.3f %9.3f %6d %5s %9.1e | %9.3f %9.3f %6d %5s %9.1e | %5.1fx\n",
            r.n, r.n_off, r.core, 100 * r.core / r.n,
            r.tb_plain, r.ts_plain, r.its_plain, r.conv_plain ? "yes" : "NO", r.err_plain,
            r.tb_elim, r.ts_elim, r.its_elim, r.conv_elim ? "yes" : "NO", r.err_elim,
            speedup)
end

println("\nColumns: _p = plain CMG, _e = CMG with degree-1/2 elimination.")
println("core%  = surviving core as a fraction of n (small => strongly near-tree).")
println("solve↑ = plain solve time / elim solve time. The n_off=0 row is a pure")
println("         tree: elimination alone is an exact solve (core = 1, its_e = 0).")
