# Shared test-graph builders + spectral helpers for the sparsify-on-stall
# benchmarks (port of the CMG-python experiments' graphs.py). Included by the
# bench_*.jl scripts, which each define `const CMG = CombinatorialMultigrid`
# BEFORE including this file (edge_ratio/n_inject reach the package internals
# through that alias). Returns ADJACENCY matrices — apply `lap` to get L.

using SparseArrays, LinearAlgebra, Random

lap(adj::SparseMatrixCSC) = spdiagm(0 => vec(sum(adj, dims = 2))) - adj

# near-Laplacian SPD operator (Laplacian + tiny slack): well posed for kappa
spd_op(adj; slack::Float64 = 1e-8) =
    lap(adj) + slack * sparse(1.0I, size(adj, 1), size(adj, 1))

relres(A, x, b) = norm(A * x - b) / norm(b)
_mean(v) = sum(v) / length(v)

# connected Erdos-Renyi blob (spanning tree + random edges to avgdeg), unit
# weights — dense enough (avgdeg >= 32) to stall the real aggregation
function dense_blob_adj(n::Int; avgdeg::Int = 32, seed::Int = 0)
    local rng = MersenneTwister(seed)
    local es = Set{Tuple{Int64,Int64}}()
    for v = 2:n
        local p = rand(rng, 1:v-1)
        push!(es, (min(v, p), max(v, p)))
    end
    while length(es) < n * avgdeg ÷ 2
        local u = rand(rng, 1:n)
        local v = rand(rng, 1:n)
        u != v && push!(es, (min(u, v), max(u, v)))
    end
    local I = Int64[]; local J = Int64[]; local V = Float64[]
    for (u, v) in es
        push!(I, u); push!(J, v); push!(V, 1.0)
        push!(I, v); push!(J, u); push!(V, 1.0)
    end
    return sparse(I, J, V, n, n)
end

# chain of dense blobs joined by single weak bridges
function blob_chain_adj(nblobs::Int, blobn::Int; avgdeg::Int = 40, seed::Int = 7,
                        wbridge::Float64 = 1e-2)
    local rng = MersenneTwister(seed)
    local I = Int64[]; local J = Int64[]; local V = Float64[]
    for k = 0:nblobs-1
        local off = k * blobn
        local B = dense_blob_adj(blobn; avgdeg = avgdeg, seed = seed + k)
        local rv = rowvals(B); local nz = nonzeros(B)
        for j = 1:blobn, p in nzrange(B, j)
            push!(I, rv[p] + off); push!(J, j + off); push!(V, nz[p])
        end
        if k > 0
            local a = (k - 1) * blobn + rand(rng, 1:blobn)
            local b = off + rand(rng, 1:blobn)
            push!(I, a); push!(J, b); push!(V, wbridge)
            push!(I, b); push!(J, a); push!(V, wbridge)
        end
    end
    return sparse(I, J, V, nblobs * blobn, nblobs * blobn)
end

# two dense blobs joined by ONE weak (high-resistance) bridge
function dense_blob_pair_bridge_adj(n::Int; avgdeg::Int = 32, seed::Int = 2,
                                    wbridge::Float64 = 1e-3)
    local h = n ÷ 2
    local Ab = blockdiag(dense_blob_adj(h; avgdeg = avgdeg, seed = seed),
                         dense_blob_adj(n - h; avgdeg = avgdeg, seed = seed + 100))
    return Ab + sparse([1, h + 1], [h + 1, 1], [wbridge, wbridge], n, n)
end

# generalized condition number kappa(M^-1 A) over the shared range
function gkappa(A, M)
    local ev = eigvals(Symmetric(Matrix(A)), Symmetric(Matrix(M)))
    ev = filter(x -> x > 1e-9, ev)
    return maximum(ev) / minimum(ev)
end

# edge-reduction ratio m_c/m of one aggregate+contract step (the stall criterion)
function edge_ratio(A)
    local n = size(A, 1)
    local cI, nc = CMG.steiner_group(A, Array(diag(A)))
    nc == 1 && return 0.0
    local Ac = CMG.contract_coo(A, cI, nc)
    local m = CMG.nnz_lower(A) - n
    return m == 0 ? 0.0 : (CMG.nnz_lower(Ac) - nc) / m
end

# count injected same-size (identity-transfer) levels
n_inject(H) = count(h -> !h.islast && h.nc == h.n, H)
