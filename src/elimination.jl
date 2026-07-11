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
    ElimSequence

The elimination records in pooled (CSR-style) storage: entry `t` eliminated
`node[t]` on pivot `piv[t]`, with neighbors `nbrs[ptr[t]:ptr[t+1]-1]` and edge
weights `ws[...]` (matrix entry `A[node[t], nbrs[a]] = -ws[a]`) at elimination
time, in elimination order. Flat pools avoid per-record allocation and make the
forward/back substitution walk contiguous memory.
"""
struct ElimSequence
    node::Vector{Int64}
    piv::Vector{Float64}
    ptr::Vector{Int64}      # length(node) + 1 offsets into nbrs/ws
    nbrs::Vector{Int64}
    ws::Vector{Float64}
end

ElimSequence() = ElimSequence(Int64[], Float64[], Int64[1], Int64[], Float64[])

Base.length(seq::ElimSequence) = length(seq.node)

"""
    EliminatedHierarchy

Bundles the exact degree-1/2 elimination of an input Laplacian/SDD matrix with
the CMG hierarchy built on the reduced (surviving) matrix. Produced by
`cmg_preconditioner_lap(A; eliminate = true)` and consumed by
`cmg_solve(::EliminatedHierarchy, b; ...)`.

- `elims`: pooled elimination records, in order.
- `ind`  : original indices of the surviving nodes (sorted increasing).
- `n`    : original problem size.
- `H`    : CMG hierarchy on the reduced matrix; empty when the reduced system is
           tiny (`length(ind) <= 1`) and solved directly.
- `A_red`: the reduced (Schur-complement) matrix on `ind`.
- `is_lap`: `true` when the input is a pure (row-sum-zero) Laplacian, `false`
            for a strictly diagonally dominant (SDD) matrix.
"""
struct EliminatedHierarchy
    elims::ElimSequence
    ind::Vector{Int64}
    n::Int64
    H::Vector{HierarchyLevel}
    A_red::SparseMatrixCSC{Float64,Int64}
    is_lap::Bool
end

"""
    elims, ind, A_red, is_lap = eliminate_deg12(A)

Exactly eliminate all degree-1 and degree-2 nodes of the Laplacian/SDD matrix
`A` by repeated Schur complementation. Returns the pooled elimination records,
the surviving node indices, the reduced matrix `A_red = Schur(A)[ind, ind]`,
and whether `A` is a pure Laplacian. Requires a symmetric matrix with
non-positive off-diagonals; a positive off-diagonal throws the same
`ArgumentError` as `validateInput!` (symmetry is checked by the caller,
`build_eliminated_hierarchy`).

A node `v` of degree `deg in {1, 2}` is eliminated on pivot `p = A[v,v]`; for
every ordered neighbor pair `(i, j)` (including `i == j`) the update is
`A[i,j] -= A[i,v] * A[v,j] / p`. With `deg <= 2` this touches only the two
diagonals and, for `deg == 2`, a single fill edge of weight `w_i*w_j/p` between
the neighbors (the classical series/harmonic rule). Degree-0 nodes are never
eligible, so the Laplacian null-space node (pivot 0) always survives.

Implementation: array-based, no per-node hash maps and no per-record
allocation. The base adjacency is read in place from the CSC structure; edge
deletion is lazy (`alive` checks at scan time); fill edges go to per-node spill
vectors. `deg` is a *lower bound* on the live-distinct degree — decremented
when a neighbor is eliminated, never incremented by fill — so a popped
candidate is first *compacted* into reused scratch buffers (dead entries
dropped, duplicate neighbors summed via an epoch-stamped sparse accumulator)
and its exact degree re-checked; a stale pop is a cheap skip, never a wrong
elimination. Eliminated nodes append their scratch to the record pools and
store no adjacency; surviving candidates keep the compacted form.

Total scanning work is amortized `O(n + m)`: an individual live spill entry can
be rescanned across repeated compactions of the same node, but per pop the
scanned cells split into dead cells (each discarded forever at the compaction
that sees it), at most `2 + deaths-since-last-compaction` live cells (the bound
is reset to the exact degree at each compaction and only neighbor deaths
decrement it, and a pop requires the bound to be at most 2), and
fills-since-last-compaction — and deaths, fills, and pops are each globally
`O(n + m)`, so the sum telescopes. The elimination *order* can differ from
other strategies, but the eliminated set, the survivors, and the Schur
complement are order-independent.
"""
function eliminate_deg12(A::SparseMatrixCSC)
    local n = size(A, 1)
    local rows = rowvals(A)
    local vals = nonzeros(A)

    # single-pass setup: diagonal, row sums (Laplacian detection), lower-bound
    # degrees, and the positive-off-diagonal check (same error as validateInput!)
    local d = zeros(Float64, n)
    local deg = zeros(Int64, n)
    local maxabsrowsum = 0.0
    local maxdiag = 0.0
    @inbounds for j = 1:n
        local s = 0.0
        local c = 0
        for r in nzrange(A, j)
            local v = vals[r]
            s += v
            if rows[r] == j
                d[j] = v
            else
                if v > 0
                    throw(
                        ArgumentError(
                            "Current Version of CMG Does Not Support Positive Off-Diagonals!",
                        ),
                    )
                end
                c += 1
            end
        end
        deg[j] = c
        maxabsrowsum = max(maxabsrowsum, abs(s))
        maxdiag = max(maxdiag, d[j])
    end
    local is_lap = n == 0 ? true : maxabsrowsum <= 1e-13 * max(1.0, maxdiag)

    local alive = fill(true, n)
    local base_live = fill(true, n)                    # base CSC slice not yet consumed
    local spill_nbr = Vector{Vector{Int64}}(undef, n)  # fill edges / compacted adjacency
    local spill_w = Vector{Vector{Float64}}(undef, n)

    # sparse-accumulator scratch for compaction, plus the reused output buffers
    local mark = zeros(Int64, n)
    local slot = zeros(Int64, n)
    local epoch = Ref(zero(Int64))
    local sc_nbrs = Int64[]
    local sc_ws = Float64[]

    # pooled elimination records
    local seq = ElimSequence()
    sizehint!(seq.node, n)
    sizehint!(seq.piv, n)
    sizehint!(seq.ptr, n + 1)
    sizehint!(seq.nbrs, 2 * n)
    sizehint!(seq.ws, 2 * n)

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
        compact_into_scratch!(
            sc_nbrs, sc_ws, v, A, rows, vals, base_live, spill_nbr, spill_w,
            alive, mark, slot, epoch)
        local dv = length(sc_nbrs)
        if dv == 0 || dv > 2
            # ground node or stale lower bound: keep the compacted adjacency
            spill_nbr[v] = copy(sc_nbrs)
            spill_w[v] = copy(sc_ws)
            deg[v] = dv
            continue
        end

        local p = d[v]
        push!(seq.node, v)
        push!(seq.piv, p)
        append!(seq.nbrs, sc_nbrs)
        append!(seq.ws, sc_ws)
        push!(seq.ptr, length(seq.nbrs) + 1)
        alive[v] = false
        # eliminated nodes are never scanned again; store no adjacency
        deg[v] = 0

        # Schur diagonal corrections on the neighbors
        for a = 1:dv
            d[sc_nbrs[a]] -= sc_ws[a]^2 / p
        end
        # single fill / series edge between the two neighbors of a degree-2 node
        if dv == 2
            local f = sc_ws[1] * sc_ws[2] / p
            push_fill!(spill_nbr, spill_w, sc_nbrs[1], sc_nbrs[2], f)
            push_fill!(spill_nbr, spill_w, sc_nbrs[2], sc_nbrs[1], f)
        end

        # neighbors lost a distinct live neighbor; enqueue new candidates
        for a = 1:dv
            local k = sc_nbrs[a]
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

    # rebuild the reduced matrix: compact all survivors first so the triplet
    # arrays can be filled at their exact final size
    local surv_nbr = Vector{Vector{Int64}}(undef, m)
    local surv_w = Vector{Vector{Float64}}(undef, m)
    local nnz_red = m
    for r = 1:m
        compact_into_scratch!(
            sc_nbrs, sc_ws, ind[r], A, rows, vals, base_live, spill_nbr, spill_w,
            alive, mark, slot, epoch)
        surv_nbr[r] = copy(sc_nbrs)
        surv_w[r] = copy(sc_ws)
        nnz_red += length(sc_nbrs)
    end
    local I = Vector{Int64}(undef, nnz_red)
    local J = Vector{Int64}(undef, nnz_red)
    local V = Vector{Float64}(undef, nnz_red)
    local t = 0
    @inbounds for r = 1:m
        t += 1
        I[t] = r
        J[t] = r
        V[t] = d[ind[r]]
        local nb = surv_nbr[r]
        local wv = surv_w[r]
        for a = 1:length(nb)
            t += 1
            I[t] = r
            J[t] = pos[nb[a]]
            V[t] = -wv[a]
        end
    end
    local A_red = sparse(I, J, V, m, m)

    return seq, ind, A_red, is_lap
end

# Dedupe the live adjacency of `v` (base CSC slice + spill) into the reused
# scratch buffers, summing duplicate neighbors with the epoch-stamped sparse
# accumulator, and consume the base slice. The caller decides what to do with
# the scratch: append it to the record pools (elimination) or copy it into the
# node's spill vectors (survivor).
function compact_into_scratch!(
    sc_nbrs::Vector{Int64},
    sc_ws::Vector{Float64},
    v::Int64,
    A::SparseMatrixCSC,
    rows::AbstractVector,
    vals::AbstractVector,
    base_live::Vector{Bool},
    spill_nbr::Vector{Vector{Int64}},
    spill_w::Vector{Vector{Float64}},
    alive::Vector{Bool},
    mark::Vector{Int64},
    slot::Vector{Int64},
    epoch::Base.RefValue{Int64},
)
    epoch[] += 1
    local e = epoch[]
    empty!(sc_nbrs)
    empty!(sc_ws)
    @inbounds if base_live[v]
        for r in nzrange(A, v)
            local k = Int64(rows[r])
            (k == v || !alive[k]) && continue
            if mark[k] != e
                mark[k] = e
                push!(sc_nbrs, k)
                push!(sc_ws, -Float64(vals[r]))
                slot[k] = length(sc_nbrs)
            else
                sc_ws[slot[k]] -= Float64(vals[r])
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
                push!(sc_nbrs, k)
                push!(sc_ws, sw[a])
                slot[k] = length(sc_nbrs)
            else
                sc_ws[slot[k]] += sw[a]
            end
        end
    end
    base_live[v] = false
    return nothing
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
    EH = build_eliminated_hierarchy(A_lap; min_frac = ELIM_MIN_FRAC)

Run `eliminate_deg12` on `A_lap` and build a CMG hierarchy on the reduced matrix
(when it has at least two nodes). Returns an `EliminatedHierarchy`. Throws the
same `ArgumentError`s as the non-elimination path for an asymmetric matrix or a
positive off-diagonal.

**Adaptive skip:** when fewer than `min_frac` of the nodes (floor: 2) are
degree-1/2 candidates — decided by one allocation-free scan — the elimination
machinery is skipped and the hierarchy is built directly on the input, with
`ind = 1:n`, empty `elims`, and `A_red` aliasing the input (no rebuild). The
result is exact either way; the skip only avoids paying several full passes
and a matrix rebuild on graphs with no low-degree structure.
"""
# Allocation-free pre-scan: count the degree-1/2 elimination candidates and
# detect the Laplacian property (same criterion as eliminate_deg12's setup).
# One pass over the CSC arrays, no allocation — the cheap gate that decides
# whether the (much more expensive) elimination machinery is worth running.
function scan_deg12(A::SparseMatrixCSC)
    local n = size(A, 1)
    local rows = rowvals(A)
    local vals = nonzeros(A)
    local cand = 0
    local maxabsrowsum = 0.0
    local maxdiag = 0.0
    @inbounds for j = 1:n
        local s = 0.0
        local c = 0
        for r in nzrange(A, j)
            local v = vals[r]
            s += v
            if rows[r] == j
                maxdiag = max(maxdiag, v)
            else
                c += 1
            end
        end
        (c == 1 || c == 2) && (cand += 1)
        maxabsrowsum = max(maxabsrowsum, abs(s))
    end
    local is_lap = n == 0 ? true : maxabsrowsum <= 1e-13 * max(1.0, maxdiag)
    return cand, is_lap
end

# Skip elimination when fewer than this fraction of the nodes are degree-1/2
# candidates (with an absolute floor of 2): the elimination machinery costs
# several full passes plus a matrix rebuild, which buys nothing on graphs
# without low-degree structure. The initial candidate count is a lower bound
# on what a full cascade could remove, so near-trees always clear the gate.
const ELIM_MIN_FRAC = 0.01

function build_eliminated_hierarchy(
    A_lap::SparseMatrixCSC;
    min_frac::Float64 = ELIM_MIN_FRAC,
    sparsify_on_stall::Bool = false,
    sparsify_opts::SparsifyOptions = SparsifyOptions(),
)::EliminatedHierarchy
    local n = size(A_lap, 1)

    # Adaptive skip: with (almost) no degree-1/2 candidates, elimination would
    # rebuild the matrix for nothing — build plain CMG on the input instead.
    # Exactness is unaffected (this skips work, it approximates nothing), and
    # validateInput! performs the symmetry/off-diagonal checks on this path.
    local cand, is_lap = scan_deg12(A_lap)
    if cand < max(2, ceil(Int, min_frac * n))
        local A_ = validateInput!(A_lap)
        local H = build_hierarchy(A_lap, A_; sparsify_on_stall = sparsify_on_stall,
            sparsify_opts = sparsify_opts)
        return EliminatedHierarchy(ElimSequence(), collect(Int64, 1:n), n, H, A_lap, is_lap)
    end

    if !issymmetric(A_lap)
        throw(ArgumentError("Input Matrix Must Be Symmetric!"))
    end
    local elims, ind, A_red, is_lap_e = eliminate_deg12(A_lap)
    local H = HierarchyLevel[]
    if length(ind) >= 2
        local A_red_ = validateInput!(A_red)   # reuses SDD augmentation
        H = build_hierarchy(A_red, A_red_; sparsify_on_stall = sparsify_on_stall,
            sparsify_opts = sparsify_opts)
    end
    return EliminatedHierarchy(elims, ind, n, H, A_red, is_lap_e)
end

"""
    forward_elim!(y, b, elims)

Forward substitution of the partial Cholesky into the preallocated buffer `y`
(overwritten with a copy of `b`, then updated in place): fold each eliminated
variable's right-hand side into its neighbors, in elimination order.
"""
function forward_elim!(y::Vector{Float64}, b::AbstractVector, elims::ElimSequence)
    copyto!(y, b)
    local ptr = elims.ptr
    local nbrs = elims.nbrs
    local ws = elims.ws
    @inbounds for t = 1:length(elims)
        local yv = y[elims.node[t]]
        local p = elims.piv[t]
        for a = ptr[t]:(ptr[t+1]-1)
            y[nbrs[a]] += (ws[a] / p) * yv
        end
    end
    return y
end

forward_elim(b::AbstractVector, elims::ElimSequence) =
    forward_elim!(Vector{Float64}(undef, length(b)), b, elims)

"""
    back_elim!(x, y, elims)

Back substitution of the partial Cholesky: recover each eliminated variable from
its neighbors (already solved), in reverse elimination order.
"""
function back_elim!(x::Vector{Float64}, y::Vector{Float64}, elims::ElimSequence)
    local ptr = elims.ptr
    local nbrs = elims.nbrs
    local ws = elims.ws
    @inbounds for t = length(elims):-1:1
        local s = y[elims.node[t]]
        for a = ptr[t]:(ptr[t+1]-1)
            s += ws[a] * x[nbrs[a]]
        end
        x[elims.node[t]] = s / elims.piv[t]
    end
    return x
end

"""
    pfunc = make_eliminated_preconditioner(EH, cycle, theta, inner_tol)

Wrap the reduced-system preconditioner with the exact forward/back substitution
so the returned closure applies one full compressed CMG cycle on the original
coordinates. With `cycle = :legacy` the closure is a fixed linear operator (safe
in a standard PCG); with `cycle = :kcycle` it is nonlinear and must be driven by
`cmg_solve`. The closure shares internal workspace across calls (the returned
vector is a reused buffer, like CMG's other preconditioner closures) and is not
reentrant or thread-safe.
"""
function make_eliminated_preconditioner(
    EH::EliminatedHierarchy,
    cycle::Symbol,
    theta::Float64,
    inner_tol::Float64,
)::Function
    local c = _canonical_cycle(cycle)

    local m = length(EH.ind)
    local inner::Function
    if isempty(EH.H)
        if m == 0
            inner = b_red -> Float64[]
        elseif EH.is_lap
            local zref = [0.0]
            inner = b_red -> zref                        # Laplacian null-space reference
        else
            local dred = EH.A_red[1, 1]
            local xred1 = [0.0]
            inner = b_red -> begin
                xred1[1] = b_red[1] / dred
                xred1
            end
        end
    elseif c === :vcycle
        local X = init_LevelAux(EH.H)
        local W = init_Workspace(EH.H)
        local M = init_Hierarchy(EH.H)
        inner = make_preconditioner(M, W, X)
    else
        inner = make_kcycle_preconditioner(EH.H, theta, inner_tol)
    end

    # closure-held workspace: forward buffer, solution buffer, reduced rhs
    local y = zeros(Float64, EH.n)
    local x = zeros(Float64, EH.n)
    local b_red = zeros(Float64, m)

    return function (b)
        forward_elim!(y, b, EH.elims)
        @inbounds for r = 1:m
            b_red[r] = y[EH.ind[r]]
        end
        if m > 0
            local x_red = inner(b_red)
            @inbounds for r = 1:m
                x[EH.ind[r]] = x_red[r]
            end
        end
        back_elim!(x, y, EH.elims)
        return x
    end
end
