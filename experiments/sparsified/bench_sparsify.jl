# Validation + efficiency benchmarks for sparsify-on-stall. Standalone:
#     julia --project=. experiments/sparsified/bench_sparsify.jl
#     julia --project=. experiments/sparsified/bench_sparsify.jl --quick
#
# Reproduces the CMG-python numbers (see PORT-NOTES.md): the stall -> resume
# edge-ratio drop + kappa(A_sp^-1 A) ~ 4-6, the spanner being essential
# (uniform-only kappa is orders worse), and the greedy vs Baswana-Sen spanner
# speedup (~150x at n = 3600).

using Random, SparseArrays, LinearAlgebra, Printf, CombinatorialMultigrid
const CMG = CombinatorialMultigrid
include(joinpath(@__DIR__, "graphs.jl"))

const QUICK = "--quick" in ARGS
const REPS = 3

function time_min(f; reps = REPS)
    f()                                  # warmup (JIT)
    local best = Inf
    for _ = 1:reps
        GC.gc()
        best = min(best, @elapsed f())
    end
    return best
end

println("== stall -> resume + spectral quality ==")
let
    A = spd_op(dense_blob_adj(400; avgdeg = 32, seed = 1))
    e0 = CMG.edges_of(A)
    sp_e, _ = CMG.sparsify(400, e0; keep_frac = 0.5, rng = MersenneTwister(0))
    Asp = CMG.sdd_from_edges(400, sp_e, CMG.slack_of(A))
    @printf("dense_blob(400,d32)  edge_ratio %.3f -> %.3f   reduction %.2fx   kappa(A_sp^-1 A) %.1f\n",
        edge_ratio(A), edge_ratio(Asp), length(e0) / length(sp_e), gkappa(A, Asp))
end

println("\n== the spanner is essential (two blobs + one weak bridge) ==")
let
    A = spd_op(dense_blob_pair_bridge_adj(400; avgdeg = 32, seed = 2, wbridge = 1e-3);
        slack = 1e-10)
    e0 = CMG.edges_of(A)
    span_k = _mean([gkappa(A, CMG.sdd_from_edges(400,
        CMG.sparsify(400, e0; rng = MersenneTwister(i))[1], CMG.slack_of(A))) for i = 1:5])
    rng = MersenneTwister(7)
    unif_k = _mean([begin
        kept = [(u, v, w / 0.25) for (u, v, w) in e0 if rand(rng) < 0.25]
        gkappa(A, CMG.sdd_from_edges(400, kept, CMG.slack_of(A)))
    end for _ = 1:5])
    @printf("spanner+uniform kappa %.1f    uniform-only kappa %.3g   (%.0fx worse)\n",
        span_k, unif_k, unif_k / span_k)
end

println("\n== greedy vs Baswana-Sen spanner efficiency (min of $REPS) ==")
let
    sizes = QUICK ? [900] : [900, 3600]
    @printf("%8s | %12s | %14s | %8s\n", "n", "greedy (s)", "baswana-sen (s)", "speedup")
    for n in sizes
        A = spd_op(dense_blob_adj(n; avgdeg = 32, seed = 3))
        e0 = CMG.edges_of(A)
        tg = time_min(() -> CMG.greedy_spanner(n, e0, log2(n)))
        tb = time_min(() -> CMG.spanner_baswana_sen(n, e0; rng = MersenneTwister(1)))
        @printf("%8d | %12.3f | %14.4f | %7.0fx\n", n, tg, tb, tg / tb)
    end
end
