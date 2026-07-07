## Exact symmetric elimination of degree-1 and degree-2 nodes (partial Cholesky).
##
## Many Laplacians arising in practice (e.g. Spielman IPM graphs) are close to
## trees: a spanning tree plus a small number of off-tree edges. For such
## matrices, iteratively eliminating every degree-1 (leaf) and degree-2 (chain)
## node until none remain is an exact Schur-complement factorization that is
## cheap and numerically exact. What survives is the degree->=3 "core"
## (essentially the off-tree structure), which is then handed to the ordinary
## CMG solver (legacy V-cycle or K-cycle) as a much smaller iterative problem.
##
## This is opt-in via `eliminate = true` on `cmg_preconditioner_lap` /
## `cmg_preconditioner_adj`; every existing code path is untouched when it is
## unset. The exact forward/back-substitution mirrors the degree-1/2 special
## case of Laplacians.jl `elimDeg12`/`forwardSolve`/`backSolve`, generalized
## from a spanning tree to an arbitrary graph.

"""
    ElimRecord

One eliminated node `node`, its neighbors `nbrs` with edge weights `ws` (matrix
entry `A[node, nbrs[k]] = -ws[k]`), and the pivot `piv = A[node, node]` at the
moment of elimination. Records are stored in elimination order.
"""
struct ElimRecord
    node::Int64
    nbrs::Vector{Int64}
    ws::Vector{Float64}
    piv::Float64
end

"""
    EliminatedHierarchy

Bundles the exact degree-1/2 elimination of an input Laplacian/SDD matrix with
the CMG hierarchy built on the reduced (surviving) matrix. Produced by
`cmg_preconditioner_lap(A; eliminate = true)` and consumed by
`cmg_solve(::EliminatedHierarchy, b; ...)`.

- `elims`: elimination records, in order.
- `ind`  : original indices of the surviving nodes (sorted increasing).
- `n`    : original problem size.
- `H`    : CMG hierarchy on the reduced matrix; empty when the reduced system is
           tiny (`length(ind) <= 1`) and solved directly.
- `A_red`: the reduced (Schur-complement) matrix on `ind`.
- `is_lap`: `true` when the input is a pure (row-sum-zero) Laplacian, `false`
            for a strictly diagonally dominant (SDD) matrix.
"""
struct EliminatedHierarchy
    elims::Vector{ElimRecord}
    ind::Vector{Int64}
    n::Int64
    H::Vector{HierarchyLevel}
    A_red::SparseMatrixCSC{Float64,Int64}
    is_lap::Bool
end

"""
    elims, ind, A_red, is_lap = eliminate_deg12(A)

Exactly eliminate all degree-1 and degree-2 nodes of the Laplacian/SDD matrix
`A` by repeated Schur complementation. Returns the elimination records, the
surviving node indices, the reduced matrix `A_red = Schur(A)[ind, ind]`, and
whether `A` is a pure Laplacian.

A node `v` of degree `deg in {1, 2}` is eliminated on pivot `p = A[v,v]`; for
every ordered neighbor pair `(i, j)` (including `i == j`) the update is
`A[i,j] -= A[i,v] * A[v,j] / p`. With `deg <= 2` this touches only the two
diagonals and, for `deg == 2`, a single fill edge of weight `w_i*w_j/p` between
the neighbors (the classical series/harmonic rule). Degree-0 nodes are never
eligible, so the Laplacian null-space node (pivot 0) always survives.

Implementation: array-based with amortized-linear total work (no per-node
hash maps). The base adjacency is read in place from the CSC structure; edge
deletion is lazy (`alive` checks at scan time); fill edges go to per-node
spill vectors. `deg` is a *lower bound* on the live-distinct degree — it is
decremented when a neighbor is eliminated but never incremented by fill, so a
popped candidate is first *compacted* (dead entries dropped, duplicate
neighbors summed via an epoch-stamped sparse accumulator) and its exact degree
re-checked; a stale pop is a cheap skip, never a wrong elimination. Compaction
replaces the node's adjacency with the deduped form, so every adjacency cell
is scanned O(1) times overall. The elimination *order* can differ from other
strategies, but the eliminated set, the survivors, and the Schur complement
are order-independent.
"""
function eliminate_deg12(A::SparseMatrixCSC)
    local n = size(A, 1)
    local d = Vector{Float64}(diag(A))

    # pure Laplacian (all row sums ~ 0) vs strictly dominant (SDD)
    local rowsum = vec(sum(A, dims = 1))
    local maxdiag = n == 0 ? 1.0 : maximum(d)
    local is_lap = n == 0 ? true : maximum(abs, rowsum) <= 1e-13 * max(1.0, maxdiag)

    local rows = rowvals(A)
    local vals = nonzeros(A)

    local alive = fill(true, n)
    local base_live = fill(true, n)                    # base CSC slice not yet consumed
    local spill_nbr = Vector{Vector{Int64}}(undef, n)  # fill edges / compacted adjacency
    local spill_w = Vector{Vector{Float64}}(undef, n)

    # lower-bound live-distinct degree = stored off-diagonal count per column
    local deg = zeros(Int64, n)
    @inbounds for j = 1:n
        local c = 0
        for r in nzrange(A, j)
            if rows[r] != j
                c += 1
            end
        end
        deg[j] = c
    end

    # sparse-accumulator scratch for compaction
    local mark = zeros(Int64, n)
    local slot = zeros(Int64, n)
    local epoch = Ref(zero(Int64))

    local elims = ElimRecord[]

    # LIFO worklist of candidate degree-1/2 nodes (order does not affect
    # exactness; forward/back use the recorded elimination order)
    local stack = Int64[]
    local inq = fill(false, n)
    @inbounds for i = 1:n
        if deg[i] == 1 || deg[i] == 2
            push!(stack, i)
            inq[i] = true
        end
    end

    while !isempty(stack)
        local v = pop!(stack)
        inq[v] = false
        alive[v] || continue
        local nbrs, ws = compact_adjacency!(
            v, A, rows, vals, base_live, spill_nbr, spill_w, alive, deg, mark, slot, epoch)
        local dv = length(nbrs)
        # exact-degree check: deg-0 (ground) survives; a stale lower bound is a skip
        (dv == 1 || dv == 2) || continue

        local p = d[v]
        push!(elims, ElimRecord(v, nbrs, ws, p))
        alive[v] = false

        # Schur diagonal corrections on the neighbors
        for a = 1:dv
            d[nbrs[a]] -= ws[a]^2 / p
        end
        # single fill / series edge between the two neighbors of a degree-2 node
        if dv == 2
            local f = ws[1] * ws[2] / p
            push_fill!(spill_nbr, spill_w, nbrs[1], nbrs[2], f)
            push_fill!(spill_nbr, spill_w, nbrs[2], nbrs[1], f)
        end

        # neighbors lost a distinct live neighbor; enqueue new candidates
        for k in nbrs
            deg[k] -= 1
            if alive[k] && deg[k] <= 2 && !inq[k]
                push!(stack, k)
                inq[k] = true
            end
        end
    end

    local ind = findall(alive)
    local m = length(ind)
    local pos = zeros(Int64, n)
    @inbounds for r = 1:m
        pos[ind[r]] = r
    end

    # rebuild the reduced matrix from compacted survivors
    local I = Int64[]
    local J = Int64[]
    local V = Float64[]
    for orig in ind
        local nbrs, ws = compact_adjacency!(
            orig, A, rows, vals, base_live, spill_nbr, spill_w, alive, deg, mark, slot, epoch)
        local r = pos[orig]
        push!(I, r)
        push!(J, r)
        push!(V, d[orig])
        for a = 1:length(nbrs)
            push!(I, r)
            push!(J, pos[nbrs[a]])
            push!(V, -ws[a])
        end
    end
    local A_red = sparse(I, J, V, m, m)

    return elims, ind, A_red, is_lap
end

# Dedupe the live adjacency of `v` (base CSC slice + spill), summing duplicate
# neighbors with the epoch-stamped sparse accumulator, and store the compacted
# form back (base slice consumed, spill := compacted). Returns `(nbrs, ws)` —
# the same vectors that now back the node's adjacency; an eliminated node hands
# them to its ElimRecord (it is dead, so the aliasing is harmless), and fills
# arriving later are appended after them.
function compact_adjacency!(
    v::Int64,
    A::SparseMatrixCSC,
    rows::AbstractVector,
    vals::AbstractVector,
    base_live::Vector{Bool},
    spill_nbr::Vector{Vector{Int64}},
    spill_w::Vector{Vector{Float64}},
    alive::Vector{Bool},
    deg::Vector{Int64},
    mark::Vector{Int64},
    slot::Vector{Int64},
    epoch::Base.RefValue{Int64},
)
    epoch[] += 1
    local e = epoch[]
    local nbrs = Int64[]
    local ws = Float64[]
    @inbounds if base_live[v]
        for r in nzrange(A, v)
            local k = Int64(rows[r])
            (k == v || !alive[k]) && continue
            if mark[k] != e
                mark[k] = e
                push!(nbrs, k)
                push!(ws, -Float64(vals[r]))
                slot[k] = length(nbrs)
            else
                ws[slot[k]] -= Float64(vals[r])
            end
        end
    end
    @inbounds if isassigned(spill_nbr, v)
        local sn = spill_nbr[v]
        local sw = spill_w[v]
        for a = 1:length(sn)
            local k = sn[a]
            alive[k] || continue
            if mark[k] != e
                mark[k] = e
                push!(nbrs, k)
                push!(ws, sw[a])
                slot[k] = length(nbrs)
            else
                ws[slot[k]] += sw[a]
            end
        end
    end
    base_live[v] = false
    spill_nbr[v] = nbrs
    spill_w[v] = ws
    deg[v] = length(nbrs)
    return nbrs, ws
end

@inline function push_fill!(
    spill_nbr::Vector{Vector{Int64}},
    spill_w::Vector{Vector{Float64}},
    x::Int64,
    y::Int64,
    f::Float64,
)
    if !isassigned(spill_nbr, x)
        spill_nbr[x] = Int64[]
        spill_w[x] = Float64[]
    end
    push!(spill_nbr[x], y)
    push!(spill_w[x], f)
    return nothing
end

"""
    EH = build_eliminated_hierarchy(A_lap)

Run `eliminate_deg12` on `A_lap` and build a CMG hierarchy on the reduced matrix
(when it has at least two nodes). Returns an `EliminatedHierarchy`.
"""
function build_eliminated_hierarchy(A_lap::SparseMatrixCSC)::EliminatedHierarchy
    local n = size(A_lap, 1)
    local elims, ind, A_red, is_lap = eliminate_deg12(A_lap)
    local H = HierarchyLevel[]
    if length(ind) >= 2
        local A_red_ = validateInput!(A_red)   # reuses SDD augmentation, throws on positive off-diagonals
        H = build_hierarchy(A_red, A_red_)
    end
    return EliminatedHierarchy(elims, ind, n, H, A_red, is_lap)
end

"""
    y = forward_elim(b, elims)

Forward substitution of the partial Cholesky: fold each eliminated variable's
right-hand side into its neighbors, in elimination order.
"""
function forward_elim(b::AbstractVector, elims::Vector{ElimRecord})
    local y = Vector{Float64}(b)
    @inbounds for e in elims
        local yv = y[e.node]
        local p = e.piv
        for k = 1:length(e.nbrs)
            y[e.nbrs[k]] += (e.ws[k] / p) * yv
        end
    end
    return y
end

"""
    back_elim!(x, y, elims)

Back substitution of the partial Cholesky: recover each eliminated variable from
its neighbors (already solved), in reverse elimination order.
"""
function back_elim!(x::Vector{Float64}, y::Vector{Float64}, elims::Vector{ElimRecord})
    @inbounds for idx = length(elims):-1:1
        local e = elims[idx]
        local s = y[e.node]
        for k = 1:length(e.nbrs)
            s += e.ws[k] * x[e.nbrs[k]]
        end
        x[e.node] = s / e.piv
    end
    return x
end

"""
    pfunc = make_eliminated_preconditioner(EH, cycle, theta, inner_tol)

Wrap the reduced-system preconditioner with the exact forward/back substitution
so the returned closure applies one full compressed CMG cycle on the original
coordinates. With `cycle = :vcycle` the closure is a fixed linear operator (safe
in a standard PCG); with `cycle = :kcycle` it is nonlinear and must be driven by
`cmg_solve`.
"""
function make_eliminated_preconditioner(
    EH::EliminatedHierarchy,
    cycle::Symbol,
    theta::Float64,
    inner_tol::Float64,
)::Function
    if cycle !== :vcycle && cycle !== :kcycle
        throw(ArgumentError("unknown cycle $(repr(cycle)); use :vcycle or :kcycle"))
    end

    local inner::Function
    if isempty(EH.H)
        if isempty(EH.ind)
            inner = b_red -> Float64[]
        elseif EH.is_lap
            inner = b_red -> [0.0]                       # Laplacian null-space reference
        else
            local dred = EH.A_red[1, 1]
            inner = b_red -> [b_red[1] / dred]
        end
    elseif cycle === :vcycle
        local X = init_LevelAux(EH.H)
        local W = init_Workspace(EH.H)
        local M = init_Hierarchy(EH.H)
        inner = make_preconditioner(M, W, X)
    else
        inner = make_kcycle_preconditioner(EH.H, theta, inner_tol)
    end

    return function (b)
        local y = forward_elim(b, EH.elims)
        local x = zeros(Float64, EH.n)
        if !isempty(EH.ind)
            local x_red = inner(Vector{Float64}(y[EH.ind]))
            @inbounds for r = 1:length(EH.ind)
                x[EH.ind[r]] = x_red[r]
            end
        end
        back_elim!(x, y, EH.elims)
        return x
    end
end
