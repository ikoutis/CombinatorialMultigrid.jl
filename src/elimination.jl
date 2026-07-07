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
"""
function eliminate_deg12(A::SparseMatrixCSC)
    local n = size(A, 1)
    local d = Vector{Float64}(diag(A))

    # pure Laplacian (all row sums ~ 0) vs strictly dominant (SDD)
    local rowsum = vec(sum(A, dims = 1))
    local maxdiag = n == 0 ? 1.0 : maximum(d)
    local is_lap = n == 0 ? true : maximum(abs, rowsum) <= 1e-13 * max(1.0, maxdiag)

    # adjacency of positive edge weights: A[i,j] = -w_ij for i != j
    local adj = [Dict{Int64,Float64}() for _ = 1:n]
    local rows = rowvals(A)
    local vals = nonzeros(A)
    @inbounds for j = 1:n
        for r in nzrange(A, j)
            local i = rows[r]
            if i != j
                adj[i][j] = -vals[r]
            end
        end
    end

    local alive = trues(n)
    local elims = ElimRecord[]

    # LIFO worklist of currently degree-1/2 nodes (order does not affect
    # exactness; forward/back use the recorded elimination order)
    local stack = Int64[]
    local inq = falses(n)
    @inbounds for i = 1:n
        local deg = length(adj[i])
        if deg == 1 || deg == 2
            push!(stack, i)
            inq[i] = true
        end
    end

    while !isempty(stack)
        local v = pop!(stack)
        inq[v] = false
        alive[v] || continue
        local deg = length(adj[v])
        (deg == 1 || deg == 2) || continue

        local p = d[v]
        local nbrs = collect(keys(adj[v]))
        local ws = Float64[adj[v][k] for k in nbrs]
        push!(elims, ElimRecord(v, nbrs, ws, p))

        # detach v from the graph
        for k in nbrs
            delete!(adj[k], v)
        end
        alive[v] = false
        empty!(adj[v])

        # Schur diagonal corrections on the neighbors
        for a = 1:deg
            d[nbrs[a]] -= ws[a]^2 / p
        end
        # single fill / series edge between the two neighbors of a degree-2 node
        if deg == 2
            local i = nbrs[1]
            local j = nbrs[2]
            local f = ws[1] * ws[2] / p
            adj[i][j] = get(adj[i], j, 0.0) + f
            adj[j][i] = get(adj[j], i, 0.0) + f
        end

        # re-enqueue neighbors that just became degree-1/2
        for k in nbrs
            if alive[k] && !inq[k]
                local dk = length(adj[k])
                if dk == 1 || dk == 2
                    push!(stack, k)
                    inq[k] = true
                end
            end
        end
    end

    local ind = findall(alive)
    local m = length(ind)
    local pos = Dict{Int64,Int64}()
    for (r, orig) in enumerate(ind)
        pos[orig] = r
    end

    # rebuild the reduced matrix from surviving diagonals and (symmetric) edges
    local I = Int64[]
    local J = Int64[]
    local V = Float64[]
    for orig in ind
        local r = pos[orig]
        push!(I, r)
        push!(J, r)
        push!(V, d[orig])
        for (k, w) in adj[orig]
            local rk = get(pos, k, 0)
            if rk != 0
                push!(I, r)
                push!(J, rk)
                push!(V, -w)
            end
        end
    end
    local A_red = sparse(I, J, V, m, m)

    return elims, ind, A_red, is_lap
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
