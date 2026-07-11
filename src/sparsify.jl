# Adaptive spectral sparsifier + options + SparseMatrixCSC<->edge bridge.
#
# port: CMG-python src/pycmg/_sparsify.py (SparsifyOptions, spanner_bundle,
#       sparsify, edges_of, slack_of, sdd_from_edges).
#
# The sparsifier keeps a spanner (the "bundle") at weight 1 and keeps each
# off-bundle edge with a single probability `p` chosen to hit a target
# keep-fraction, reweighting survivors by `1/p` (an unbiased estimator of the
# off-bundle part). Because a spanner bounds off-spanner leverage, this uniform
# sampling concentrates.

"""
    SparsifyOptions

Tuning knobs for sparsify-on-stall (see `cmg_solve(A, b; sparsify_on_stall=true,
sparsify_opts=SparsifyOptions(...))`). Defaults are the production values
validated in the CMG-python package.

- `keep_frac`  : target keep-fraction for off-bundle edges (0.5 -> C_op ~ 2).
                 For the L-cycle this sets the injected level's repeat multiplier
                 = floor(1/keep_frac - 1): 0.5 -> repeat 1 (cheapest), 1/3 ->
                 repeat 2. Lower keep_frac sparsifies harder but raises the repeat.
- `bundles`    : number of peeled spanners/forests kept at full weight (default
                 3, i.e. a 3-forest Nagamochi–Ibaraki bundle for `:mst`). More
                 bundles -> lower effective stretch / higher cut connectivity;
                 each is an extra forest build (cheap for `:mst`: O(m) each).
- `t`          : greedy-spanner stretch; `nothing` -> `max(2, log2 n)` per level.
- `stall_ratio`: per-level NODE-coarsening threshold; a level is productive iff
                 aggregation drops the node count below `stall_ratio * n`
                 (`nc > stall_ratio * n` is a stall -> sparsify). This is the
                 validated CMG-python criterion; the original Julia port
                 mistakenly tested edges (`m_c <= stall_ratio*m`), which fired on
                 normal levels -- corrected to nodes here.
- `max_inject` : cap on injected sparsifier levels (guarantees termination).
- `nnz_budget` : cumulative operator-complexity budget, a multiple of input nnz
                 (the same 5x guard stock CMG uses). `Inf` disables it.
- `spanner`    : `:mst` (default; a `bundles`-forest maximum-spanning-forest
                 bundle — a Nagamochi–Ibaraki cut certificate, O(m) to build, and
                 on the high-conductance stall levels where sparsify fires cut ≈
                 spectral so it is a good spectral preconditioner there at a
                 fraction of Baswana–Sen's cost — see `max_spanning_forest`),
                 `:baswana_sen` (the randomized (2k-1)-spanner), or `:greedy`.
- `k`          : Baswana–Sen stretch parameter; `nothing` -> `ceil(log2 n)`.
- `bundles_growth` : grow the bundle count by one per successive injection.
- `base`       : direct-solve threshold (matches `build_hierarchy`'s 500).
- `seed`       : RNG seed (the spanner + sampling are randomized; seeding makes
                 the preconditioner reproducible).
"""
Base.@kwdef struct SparsifyOptions
    keep_frac::Float64 = 0.5
    bundles::Int = 3
    t::Union{Nothing,Float64} = nothing
    stall_ratio::Float64 = 0.9   # NODE-coarsening threshold: a level is
                                 # productive iff aggregation drops nc below
                                 # stall_ratio*n; nc > stall_ratio*n is a stall.
    max_inject::Int = 10
    nnz_budget::Float64 = 5.0
    spanner::Symbol = :mst
    k::Union{Nothing,Int} = nothing
    bundles_growth::Bool = false
    base::Int = 500
    seed::Int = 0
end

# ---------------------------------------------------------------------------
# spanner bundle + adaptive sparsify
# ---------------------------------------------------------------------------
"""
    spanner_bundle(n, edges, t, bundles; spanner, k, rng) -> (bundle, off)

Peel `bundles` spanners: extract a spanner, remove its edges, repeat. Returns
`(bundle_edges, off_bundle_edges)`. More bundles -> lower effective stretch ->
aggregation resumes harder at the same edge budget.
"""
function spanner_bundle(n::Integer, edges::AbstractVector, t::Real, bundles::Integer;
                        spanner::Symbol = :baswana_sen, k = nothing,
                        rng::AbstractRNG = Random.default_rng())
    _one(rem) = spanner === :greedy      ? greedy_spanner(n, rem, t) :
                spanner === :baswana_sen ? spanner_baswana_sen(n, rem; k = k, rng = rng) :
                spanner === :mst         ? max_spanning_forest(n, rem) :
                error("unknown spanner $(repr(spanner)); use :greedy, :baswana_sen, or :mst")
    ekey(e) = (min(e[1], e[2]), max(e[1], e[2]))
    kept = SpanEdge[]
    rem = collect(SpanEdge, edges)
    for _ in 1:bundles
        s = _one(rem)
        append!(kept, s)
        sset = Set(ekey(e) for e in s)
        rem = [e for e in rem if !(ekey(e) in sset)]
    end
    return kept, rem
end

"""
    sparsify(n, edges; t, bundles, keep_frac, spanner, k, rng) -> (kept, p)

`t=nothing` -> `t = max(2, log2 n)`. `p = clamp((keep_frac*m - S)/(m - S), 0, 1)`
with `S = |bundle|`, `m = |edges|`; the bundle is kept at weight 1 and each
off-bundle edge is kept w.p. `p` and reweighted `w/p`. A sparse level (`S ~ m`)
drives `p -> 0`; a dense/fill level (`S << m`) gives `p ~ keep_frac`.
"""
function sparsify(n::Integer, edges::AbstractVector; t = nothing, bundles::Integer = 1,
                  keep_frac::Real = 0.5, spanner::Symbol = :baswana_sen, k = nothing,
                  rng::AbstractRNG = Random.default_rng())
    tt = t === nothing ? (n > 1 ? max(2.0, log2(n)) : 2.0) : float(t)
    bundle, off = spanner_bundle(n, edges, tt, bundles; spanner = spanner, k = k, rng = rng)
    m, S = length(edges), length(bundle)
    (m <= S || isempty(off)) && return collect(SpanEdge, edges), 0.0
    p = min(1.0, max(0.0, (keep_frac * m - S) / (m - S)))
    kept = copy(bundle)
    if p > 0.0
        @inbounds for (u, v, w) in off
            if rand(rng) < p
                push!(kept, (u, v, w / p))
            end
        end
    end
    return kept, p
end

# ---------------------------------------------------------------------------
# SparseMatrixCSC <-> edge-list bridge (SDD ops: L(graph) + diag(slack), >= 0)
# ---------------------------------------------------------------------------
"""
    edges_of(A) -> Vector{Tuple{Int,Int,Float64}}

Off-diagonal edges `(u, v, w)` with `u < v`, `w = -A[u,v] > 0`, from a sparse
SDD/Laplacian operator. Explicit zeros are skipped (not edges).
"""
function edges_of(A::SparseMatrixCSC)
    edges = SpanEdge[]
    rows = rowvals(A); vals = nonzeros(A)
    @inbounds for j in 1:size(A, 2)
        for r in nzrange(A, j)
            i = rows[r]
            if i < j                                # strict upper triangle
                val = vals[r]
                val != 0.0 && push!(edges, (Int64(i), Int64(j), -float(val)))
            end
        end
    end
    return edges
end

"""
    slack_of(A) -> Vector{Float64}

Per-row SDD slack = `diag(A) - (sum of incident conductances)`, which for a
symmetric SDD/Laplacian equals the row sums of `A`: `>= 0` for SDD, `~0` for a
Laplacian. Preserving it keeps a sparsified level SPD-consistent with its
parent; at every internal CMG level (a Laplacian) it is ~0, so the sparsifier
there is a pure Laplacian with the same null space.
"""
slack_of(A::SparseMatrixCSC) = vec(sum(A, dims = 2))

"""
    sdd_from_edges(n, edges, slack) -> SparseMatrixCSC

Build `L(edges) + diag(slack)`. `slack` may be a scalar or a length-`n` vector
(e.g. `slack_of(A)`).
"""
function sdd_from_edges(n::Integer, edges::AbstractVector, slack)
    deg = zeros(Float64, n)
    ne = length(edges)
    I = Vector{Int64}(undef, 2 * ne)
    J = Vector{Int64}(undef, 2 * ne)
    V = Vector{Float64}(undef, 2 * ne)
    @inbounds for (idx, e) in enumerate(edges)
        u, v, w = Int64(e[1]), Int64(e[2]), float(e[3])
        I[2idx - 1] = u; J[2idx - 1] = v; V[2idx - 1] = -w
        I[2idx]     = v; J[2idx]     = u; V[2idx]     = -w
        deg[u] += w; deg[v] += w
    end
    slackv = slack isa Number ? fill(float(slack), n) : collect(Float64, slack)
    dI = collect(Int64, 1:n)
    A = sparse(vcat(I, dI), vcat(J, dI), vcat(V, deg .+ slackv), Int(n), Int(n))
    return A
end
