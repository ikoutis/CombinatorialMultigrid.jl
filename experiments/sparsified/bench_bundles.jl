# B=1 vs B=2 spanner bundles at a fixed keep-fraction (keep_frac = 0.25).
# Standalone:
#     julia --project=. experiments/sparsified/bench_bundles.jl
#     julia --project=. experiments/sparsified/bench_bundles.jl --quick
#
# Port of the CMG-python scratch experiment (exp_bundles.py). At the SAME edge
# budget, does keeping 2 peeled spanners (more spanner-structured, fewer
# random-sampled edges) give a spectrally better sparsifier than 1? Python's
# finding: at keep = 0.25, B=2 is NEUTRAL-TO-WORSE by median kappa, and the
# default Baswana-Sen spanner is already ~24% of the edges, so two of them
# (~46%) cannot fit in a 25% budget at all -- only the small greedy spanner can.
# This reproduces the greedy B=1 vs B=2 spectral comparison.

using Random, SparseArrays, LinearAlgebra, Printf, CombinatorialMultigrid
const CMG = CombinatorialMultigrid
include(joinpath(@__DIR__, "graphs.jl"))

const QUICK = "--quick" in ARGS
const KEEP = 0.25

# sparsify with diagnostics exposed (mirrors CMG.sparsify but returns S, p, |kept|)
function sparsify_diag(n, e0, B; keep_frac = KEEP, spanner = :greedy,
                       rng = MersenneTwister(0))
    t = max(2.0, log2(n))
    bundle, off = CMG.spanner_bundle(n, e0, t, B; spanner = spanner, rng = rng)
    m, S = length(e0), length(bundle)
    (m <= S || isempty(off)) && return collect(e0), 0.0, S, length(e0)
    p = clamp((keep_frac * m - S) / (m - S), 0.0, 1.0)
    kept = copy(bundle)
    if p > 0.0
        for (u, v, w) in off
            rand(rng) < p && push!(kept, (u, v, w / p))
        end
    end
    return kept, p, S, length(kept)
end

const SEEDS = QUICK ? (0:2) : (0:5)
const GRAPHS = QUICK ? [(400, 32)] : [(400, 32), (500, 40)]

println("B=1 vs B=2 at keep_frac=$KEEP with the greedy spanner (both hit the same")
println("budget; the default Baswana-Sen spanner is too fat for two to fit in 25%).\n")
@printf("%13s | %2s | %12s | %6s | %12s | %10s | %s\n",
    "graph", "B", "|bundle| S", "p", "kept", "edge_ratio", "kappa mean")
println("-"^80)
for (n, deg) in GRAPHS
    A = spd_op(dense_blob_adj(n; avgdeg = deg, seed = 1))
    e0 = CMG.edges_of(A); m = length(e0)
    for B in (1, 2)
        Ss = Float64[]; ps = Float64[]; ks = Float64[]; ers = Float64[]; kepts = Float64[]
        for s in SEEDS
            kept, p, S, nk = sparsify_diag(n, e0, B; spanner = :greedy, rng = MersenneTwister(s))
            Asp = CMG.sdd_from_edges(n, kept, CMG.slack_of(A))
            push!(Ss, S); push!(ps, p); push!(kepts, nk)
            push!(ers, edge_ratio(Asp)); push!(ks, gkappa(A, Asp))
        end
        @printf("%13s | %2d | %5.0f(%4.1f%%) | %6.3f | %5.0f(%4.1f%%) | %10.3f | %6.1f\n",
            "n=$n d=$deg", B, _mean(Ss), 100 * _mean(Ss) / m, _mean(ps),
            _mean(kepts), 100 * _mean(kepts) / m, _mean(ers), _mean(ks))
    end
end
println("\nFinding (validated in Python over 16 seeds): at keep=0.25, B=2 does NOT")
println("beat B=1 -- neutral-to-worse median kappa, only lower variance. Keep B=1 as")
println("the default. Bundles may pay off at higher keep-fractions (where two")
println("spanners fit); a keep_frac sweep is a one-line change above.")
