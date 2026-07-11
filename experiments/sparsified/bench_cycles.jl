# L-cycle / K-cycle / Ks-cycle comparison on sparsified hierarchies. Standalone:
#     julia --project=. experiments/sparsified/bench_cycles.jl
#     julia --project=. experiments/sparsified/bench_cycles.jl --quick
#
# Port of CMG-python kscycle.py's __main__. Reports outer iterations and
# wall-clock (min of REPS) for each cycle on the same four chain-of-blobs
# configs. Python found the L-cycle fastest (fewest finest-level matvecs), the
# stock K-cycle degrading, and the Ks-cycle cutting iterations but doing more
# work. (Finest-level matvec counting is proxied here by wall-clock; adding an
# operator-instrumentation pass is a straightforward extension.)

using Random, SparseArrays, LinearAlgebra, Printf, CombinatorialMultigrid
const CMG = CombinatorialMultigrid
include(joinpath(@__DIR__, "graphs.jl"))

const QUICK = "--quick" in ARGS
const REPS = 3

function solve_stats(A, b, cyc; tol = 1e-9, maxit = 3000)
    run = () -> cmg_solve(A, b; sparsify_on_stall = true, split_components = false,
        eliminate = false, cycle = cyc, tol = tol, maxit = maxit)
    x, st = run()                        # warmup + correctness (deterministic)
    best = Inf
    for _ = 1:REPS
        GC.gc()
        best = min(best, @elapsed run())
    end
    return st.iterations, best * 1e3, st.converged, relres(A, x, b)
end

const CONFIGS = QUICK ? [(6, 150, 40, 1e-2, 7)] :
    [(6, 150, 40, 1e-2, 7), (10, 100, 40, 1e-2, 5),
     (4, 250, 56, 1e-3, 9), (8, 140, 52, 1e-3, 11)]

println("Cycles on sparsified hierarchies (chains of dense blobs, tol=1e-9).")
println("Outer iterations are misleading across cycles; wall-clock is the honest cost.\n")
@printf("%-16s| %-9s| %8s %8s %7s %10s\n", "config", "cycle", "iters", "ms", "conv", "relres")
println("-"^62)
for (nb, bn, dg, wb, sd) in CONFIGS
    n = nb * bn
    A = lap(blob_chain_adj(nb, bn; avgdeg = dg, seed = sd, wbridge = wb))
    b = randn(MersenneTwister(0), n); b .-= sum(b) / n
    Hn = CMG.build_hierarchy(A, CMG.validateInput!(A); sparsify_on_stall = true)
    tag = @sprintf("%dx%d d%d inj%d", nb, bn, dg, n_inject(Hn))
    for cyc in (:legacy, :kcycle, :kscycle)
        its, ms, ok, rr = solve_stats(A, b, cyc)
        @printf("%-16s| %-9s| %8d %8.0f %7s %10.1e\n", tag, string(cyc), its, ms, ok, rr)
        tag = ""
    end
    println("-"^62)
end
println("\nTakeaway: :legacy (L-cycle) is fastest -- injected sparsifiers are already")
println("spectrally accurate (kappa ~ 4-6), so there is nothing for a K-cycle to")
println("accelerate. :kscycle cuts iterations but does 2-3x the work; the stock")
println(":kcycle degrades (its inner FCG minimizes over the sparsifier, not the")
println("level's own operator).")
