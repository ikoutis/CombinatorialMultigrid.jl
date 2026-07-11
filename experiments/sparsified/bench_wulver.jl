# Real-data benchmark skeleton: sparsify-on-stall vs cmg-k-elim vs approxchol
# on chimera graphs. RUN ON WULVER.
#
#     julia --project=. experiments/sparsified/bench_wulver.jl
#
# This is the one benchmark that is NOT self-contained: it needs
#   (a) Laplacians.jl (for approxchol_lap -- the 'ac' baseline), which is a
#       test-only dependency, not a package dependency; and
#   (b) a chimera generator (uni_chimera / wted_chimera) -- these are the
#       standard Laplacians/benchmark graphs used on Wulver and are NOT shipped
#       in this repo. Plug yours into `make_problem` below.
#
# It compares, on one chimera problem:
#   * cmg-k-elim         -- the package default (k-cycle + degree-1/2 elimination)
#   * cmg + sparsify(L)  -- sparsify-on-stall, L-cycle driver
#   * approxchol ('ac')  -- ApproxChol from Laplacians
# reproducing the CMG-python port-back plan's headline comparison.

using Random, SparseArrays, LinearAlgebra, Printf, CombinatorialMultigrid
using Laplacians                         # approxchol_lap (and, if you use theirs,
const CMG = CombinatorialMultigrid       # a chimera generator)

# --- plug in your chimera generator here (Wulver) -------------------------
# Must return a connected weighted ADJACENCY matrix (SparseMatrixCSC), e.g.
#   uni_chimera(10^6, 10)     # unweighted
#   wted_chimera(10^6, 5)     # weighted
function make_problem()
    error("plug in uni_chimera / wted_chimera here (not shipped in this repo)")
end
# --------------------------------------------------------------------------

lap_(adj) = spdiagm(0 => vec(sum(adj, dims = 2))) - adj
relres_(A, x, b) = norm(A * x - b) / norm(b)

function main()
    adj = make_problem()
    L = lap_(adj)
    n = size(L, 1)
    b = randn(n); b .-= sum(b) / n
    @printf("chimera  n=%d  nnz(L)=%d\n\n", n, nnz(L))
    @printf("%-20s %8s %6s %6s %10s\n", "solver", "time(s)", "its", "conv", "relres")
    println("-"^54)

    # cmg-k-elim: the package default (k-cycle + elimination)
    x1, s1 = cmg_solve(L, b; tol = 1e-6, maxit = 1000)              # compile
    t1 = @elapsed ((x1, s1) = cmg_solve(L, b; tol = 1e-6, maxit = 1000))
    @printf("%-20s %8.2f %6d %6s %10.1e\n", "cmg-k-elim", t1, s1.iterations,
        s1.converged, relres_(L, x1, b))

    # cmg + sparsify-on-stall (L-cycle driver)
    x2, s2 = cmg_solve(L, b; sparsify_on_stall = true, cycle = :legacy, tol = 1e-6, maxit = 1000)
    t2 = @elapsed ((x2, s2) = cmg_solve(L, b; sparsify_on_stall = true, cycle = :legacy,
        tol = 1e-6, maxit = 1000))
    @printf("%-20s %8.2f %6d %6s %10.1e\n", "cmg + sparsify(L)", t2, s2.iterations,
        s2.converged, relres_(L, x2, b))

    # approxchol ('ac') from Laplacians
    solver = approxchol_lap(adj; tol = 1e-6, maxits = 1000)         # compile
    t3 = @elapsed (solver = approxchol_lap(adj; tol = 1e-6, maxits = 1000))
    x3 = solver(b)
    @printf("%-20s %8.2f %6s %6s %10.1e\n", "approxchol (ac)", t3, "-", "-",
        relres_(L, x3, b))
end

main()
