# Spanners in the resistance metric for sparsify-on-stall.
#
# port: CMG-python src/pycmg/_sparsify.py (greedy_spanner, spanner_baswana_sen,
#       _baswana_sen_impl), validated there against the real aggregation.
#
# An edge is a tuple `(u, v, w)` with `1 <= u < v <= n` and `w > 0` a
# conductance. The resistance-metric edge *length* is `1/w`. A t-spanner bounds
# every off-spanner edge's effective resistance by `t * r_uv` (its statistical
# leverage by `t`), which is what makes uniform sampling of the off-spanner
# edges concentrate (see sparsify.jl).
#
# Two spanners:
#   * `spanner_baswana_sen` — the randomized (2k-1)-spanner, O(k*m) expected,
#     array-based (epoch-stamped cluster accumulator, no heap). The default /
#     production spanner. Julia is natively fast here, so no JIT is needed
#     (the Python port used numba for exactly this kernel).
#   * `greedy_spanner` — the deterministic Althofer greedy t-spanner (bounded
#     Dijkstra per candidate edge). The correctness reference; O(m * Dijkstra).

const SpanEdge = Tuple{Int64,Int64,Float64}

# ---------------------------------------------------------------------------
# greedy (Althofer) t-spanner in the resistance metric (edge length = 1/w)
# ---------------------------------------------------------------------------
"""
    greedy_spanner(n, edges, t) -> Vector{Tuple{Int,Int,Float64}}

Greedy (Althofer) `t`-spanner in the resistance metric. Keep edge `(u, v)`
unless the current spanner already connects `u, v` with resistance `<= t / w`.
`edges` is a vector of `(u, v, w)` conductances. Deterministic; O(m * bounded
Dijkstra), so use `spanner_baswana_sen` at scale.
"""
function greedy_spanner(n::Integer, edges::AbstractVector, t::Real)
    adj = [Tuple{Int64,Float64}[] for _ in 1:n]     # spanner adjacency: (nbr, length)
    span = SpanEdge[]
    # stiffest first: ascending edge length 1/w  (== descending conductance w)
    order = sortperm(edges; by = e -> 1.0 / float(e[3]))
    @inbounds for i in order
        u, v, w = edges[i]
        L = 1.0 / float(w)
        if !_within(adj, Int64(u), Int64(v), t * L)
            push!(adj[u], (Int64(v), L))
            push!(adj[v], (Int64(u), L))
            push!(span, (Int64(u), Int64(v), float(w)))
        end
    end
    return span
end

# ---------------------------------------------------------------------------
# maximum-weight spanning forest (Kruskal + union-find), in conductance
# ---------------------------------------------------------------------------
"""
    max_spanning_forest(n, edges) -> Vector{Tuple{Int,Int,Float64}}

Maximum-weight spanning forest by conductance: scan edges in descending weight,
keep the heaviest that joins two components (Kruskal + union-find). `edges` is a
vector of `(u, v, w)` conductances; returns the kept edges (a subset).
O(m log m) sort + O(m α(n)) unions.

A bundle of `k` peeled maximum spanning forests is a **Nagamochi–Ibaraki
`k`-connectivity certificate** — it preserves cuts, i.e. it is a CUT sparsifier.
On the high-conductance (expander-like) levels where CMG stalls — the only
levels where sparsify is invoked — cut ≈ spectral, so this cheap forest bundle
is a good spectral preconditioner there, at a fraction of the Baswana–Sen build
cost, with a fixed density-independent bundle size of `k·(n-1)` edges. It also
keeps every connectivity bridge (forced by the spanning-forest property),
preserving the small eigenvalues from weak cuts.
"""
function max_spanning_forest(n::Integer, edges::AbstractVector)
    parent = collect(Int64, 1:n)
    rank = zeros(Int8, n)
    function find(x::Int64)
        root = x
        @inbounds while parent[root] != root
            root = parent[root]
        end
        @inbounds while parent[x] != root                # path compression
            parent[x], x = root, parent[x]
        end
        return root
    end
    kept = SpanEdge[]
    order = sortperm(edges; by = e -> -float(e[3]))       # descending conductance
    @inbounds for i in order
        u, v, w = edges[i]
        ru, rv = find(Int64(u)), find(Int64(v))
        ru == rv && continue
        if rank[ru] < rank[rv]
            ru, rv = rv, ru
        end
        parent[rv] = ru
        rank[ru] == rank[rv] && (rank[ru] += 1)
        push!(kept, (Int64(u), Int64(v), float(w)))
    end
    return kept
end

# bounded-radius Dijkstra on the current spanner: is there an s->tgt path of
# length <= cap? (decrease-key priority queue; no stale entries to skip.)
function _within(adj::Vector{Vector{Tuple{Int64,Float64}}}, s::Int64, tgt::Int64,
                 cap::Float64)
    d = Dict{Int64,Float64}(s => 0.0)
    pq = PriorityQueue{Int64,Float64}()
    pq[s] = 0.0
    while !isempty(pq)
        kv = dequeue_pair!(pq)
        u = kv.first
        du = kv.second
        u == tgt && return du <= cap
        du > cap && return false
        @inbounds for (v, l) in adj[u]
            nd = du + l
            if nd < get(d, v, Inf)
                d[v] = nd
                pq[v] = nd                          # insert or decrease-key
            end
        end
    end
    return false
end

# ---------------------------------------------------------------------------
# Baswana–Sen randomized (2k-1)-spanner: O(k*m) expected, heap-free.
# ---------------------------------------------------------------------------
"""
    spanner_baswana_sen(n, edges; k=nothing, rng=Random.default_rng())

Baswana–Sen randomized `(2k-1)`-spanner in the resistance metric (edge length
`1/w`). `k` defaults to `max(2, ceil(log2 n))`. Returns kept spanner edges
`(u, v, w)` (`u < v`) at their original conductance — a connected subgraph for
connected input, so a drop-in for `greedy_spanner`.
"""
function spanner_baswana_sen(n::Integer, edges::AbstractVector; k=nothing,
                             rng::AbstractRNG = Random.default_rng())
    isempty(edges) && return SpanEdge[]
    if k === nothing
        k = n > 2 ? max(2, ceil(Int, log2(n))) : 1
    end
    k = Int(k)
    m = length(edges)
    u = Vector{Int64}(undef, m); v = similar(u); w = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        u[i], v[i], w[i] = edges[i][1], edges[i][2], float(edges[i][3])
    end
    # symmetric CSC of conductances (each undirected edge stored both ways)
    A = sparse(vcat(u, v), vcat(v, u), vcat(w, w), n, n)
    econd = A.nzval
    elen = 1.0 ./ econd
    p = k >= 1 ? float(n)^(-1.0 / k) : 1.0
    samp = rand(rng, max(k - 1, 0), n)              # samp[phase, center] in [0,1)
    su, sv, sw = _baswana_sen_impl(Int64(n), A.colptr, A.rowval, elen, econd, k, p, samp)
    # dedup undirected (same conductance from both endpoints' contributions)
    kept = Dict{Tuple{Int64,Int64},Float64}()
    @inbounds for i in 1:length(su)
        a, b = su[i], sv[i]
        kept[a < b ? (a, b) : (b, a)] = sw[i]
    end
    out = SpanEdge[]
    for (key, val) in kept
        push!(out, (key[1], key[2], val))
    end
    return out
end

# One full Baswana–Sen run over the symmetric CSC (colptr, rowval); `elen` and
# `econd` are aligned with rowval. `samp[i, c]` decides if center `c` is sampled
# in phase `i` (sampled iff < p). Sentinel `0` marks a finalized vertex.
# Returns (su, sv, sw): spanner edges at original conductance (with duplicates).
function _baswana_sen_impl(n::Int64, colptr::Vector{<:Integer},
                           rowval::Vector{<:Integer}, elen::Vector{Float64},
                           econd::Vector{Float64}, k::Int, p::Float64,
                           samp::Matrix{Float64})
    center = collect(Int64, 1:n)                    # each vertex its own cluster
    new_center = zeros(Int64, n)
    best_len = zeros(Float64, n)                     # per-cluster scratch (epoch-stamped)
    best_u = zeros(Int64, n)
    best_cond = zeros(Float64, n)
    seen = zeros(Int64, n)
    clusters = zeros(Int64, n)
    epoch = 0
    su = Int64[]; sv = Int64[]; sw = Float64[]

    for i in 1:(k - 1)                              # phases 1..k-1: form clusters
        @inbounds for vtx in 1:n
            cv = center[vtx]
            if cv == 0                              # already finalized
                new_center[vtx] = 0
                continue
            end
            if samp[i, cv] < p                     # vtx's cluster sampled -> stays
                new_center[vtx] = cv
                continue
            end
            epoch += 1                             # least-length edge per incident cluster
            ncl = 0
            for r in colptr[vtx]:(colptr[vtx + 1] - 1)
                uu = rowval[r]
                cu = center[uu]
                (cu == 0 || cu == cv) && continue  # finalized / intra-cluster -> skip
                L = elen[r]
                if seen[cu] != epoch
                    seen[cu] = epoch
                    best_len[cu] = L; best_u[cu] = uu; best_cond[cu] = econd[r]
                    ncl += 1; clusters[ncl] = cu
                elseif L < best_len[cu]
                    best_len[cu] = L; best_u[cu] = uu; best_cond[cu] = econd[r]
                end
            end
            cstar = 0                              # nearest SAMPLED cluster
            lstar = Inf
            for j in 1:ncl
                c = clusters[j]
                if samp[i, c] < p && best_len[c] < lstar
                    lstar = best_len[c]; cstar = c
                end
            end
            if cstar == 0                          # no sampled neighbor: finalize vtx,
                for j in 1:ncl                     # edge to every incident cluster
                    c = clusters[j]
                    push!(su, vtx); push!(sv, best_u[c]); push!(sw, best_cond[c])
                end
                new_center[vtx] = 0
            else                                   # join nearest sampled cluster,
                push!(su, vtx); push!(sv, best_u[cstar]); push!(sw, best_cond[cstar])
                new_center[vtx] = cstar
                for j in 1:ncl                     # + edge to each strictly-closer cluster
                    c = clusters[j]
                    if best_len[c] < lstar
                        push!(su, vtx); push!(sv, best_u[c]); push!(sw, best_cond[c])
                    end
                end
            end
        end
        @inbounds for vtx in 1:n
            center[vtx] = new_center[vtx]
        end
    end

    @inbounds for vtx in 1:n                        # final phase: connect survivors
        cv = center[vtx]
        cv == 0 && continue
        epoch += 1
        ncl = 0
        for r in colptr[vtx]:(colptr[vtx + 1] - 1)
            uu = rowval[r]
            cu = center[uu]
            (cu == 0 || cu == cv) && continue
            L = elen[r]
            if seen[cu] != epoch
                seen[cu] = epoch
                best_len[cu] = L; best_u[cu] = uu; best_cond[cu] = econd[r]
                ncl += 1; clusters[ncl] = cu
            elseif L < best_len[cu]
                best_len[cu] = L; best_u[cu] = uu; best_cond[cu] = econd[r]
            end
        end
        for j in 1:ncl
            c = clusters[j]
            push!(su, vtx); push!(sv, best_u[c]); push!(sw, best_cond[c])
        end
    end

    return su, sv, sw
end
