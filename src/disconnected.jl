## Disconnected-graph handling: solve each connected component independently.
##
## CMG's hierarchy assumes a connected graph. On a disconnected Laplacian/SDD
## input the build either errors (an isolated vertex makes `findMinSparse`
## return a 0 parent, tripping `split_forest_!`) or, with several components,
## leaves the preconditioner an uncovered null-space dimension per extra
## component, so the outer iteration cannot converge. Splitting the input into
## its connected components and solving each block with the ordinary CMG solver
## fixes both: every block is connected, so all the existing build/grounding
## logic applies unchanged. Components are detected on the ORIGINAL matrix,
## before `validateInput!`'s n+1 augmentation (which would otherwise wire every
## strictly-dominant node to a shared ground and merge the components).
##
## Opt-in/out via `split_components` (default `true`) on `cmg_solve(A, b; ...)`
## and `cmg_preconditioner_lap`. A single-component input takes the existing
## path unchanged (one connected-components pass of overhead).
##
## `components` / `vecToComps` are adapted from Laplacians.jl (src/graphAlgs.jl):
##   Copyright (c) 2015-2016 Daniel A. Spielman and other contributors.
##   MIT "Expat" License. (Started by Dan Spielman; contributor xiao.shi@yale.edu.)
## Vendored here so this package does not depend on Laplacians.jl.

"""
    components(mat::SparseMatrixCSC) -> Vector{Ti}

Connected components of the graph with adjacency pattern `mat`, as a per-vertex
label vector numbered `1..k`. Uses the structural sparsity pattern only (values
are ignored); a stored diagonal is a harmless self-loop. Adapted from
Laplacians.jl `components` (MIT, © 2015-2016 Daniel A. Spielman and contributors).
"""
function components(mat::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n = mat.n
    order = Array{Ti}(undef, n)
    comp = zeros(Ti, n)
    c::Ti = 0
    colptr = mat.colptr
    rowval = mat.rowval
    @inbounds for x = 1:n
        if comp[x] == 0
            c += 1
            comp[x] = c
            if colptr[x+1] > colptr[x]
                ptr::Ti = 1
                orderLen::Ti = 2
                order[ptr] = x
                while ptr < orderLen
                    curNode = order[ptr]
                    for ind = colptr[curNode]:(colptr[curNode+1]-1)
                        nbr = rowval[ind]
                        if comp[nbr] == 0
                            comp[nbr] = c
                            order[orderLen] = nbr
                            orderLen += 1
                        end
                    end
                    ptr += 1
                end
            end
        end
    end
    return comp
end

"""
    vecToComps(compvec::Vector{Ti}) -> Vector{Vector{Ti}}

Turn a component-label vector (from `components`) into a list of the vertex
indices in each component. Adapted from Laplacians.jl `vecToComps` (MIT,
© 2015-2016 Daniel A. Spielman and contributors).
"""
function vecToComps(compvec::Vector{Ti}) where {Ti}
    nc = maximum(compvec)
    comps = Vector{Vector{Ti}}(undef, nc)
    sizes = zeros(Ti, nc)
    for i in compvec
        sizes[i] += 1
    end
    for i = 1:nc
        comps[i] = zeros(Ti, sizes[i])
    end
    ptrs = zeros(Ti, nc)
    for i = 1:length(compvec)
        c = compvec[i]
        ptrs[c] += 1
        comps[c][ptrs[c]] = i
    end
    return comps
end

"""
    _lap(A::SparseMatrixCSC) -> SparseMatrixCSC

Graph Laplacian `D - A` of an adjacency matrix `A` (`D` = weighted-degree
diagonal). Internal replacement for `Laplacians.lap`, used by
`cmg_preconditioner_adj`.
"""
_lap(A::SparseMatrixCSC) = spdiagm(0 => vec(sum(A, dims = 2))) - A

"""
    DisconnectedHierarchy

Bundles the per-connected-component solvers for a disconnected input. Produced by
`build_disconnected_hierarchy` and consumed by
`cmg_solve(::DisconnectedHierarchy, b; ...)`.

- `ind`   : original vertex indices of each component (sorted increasing).
- `blocks`: per component, either the CMG hierarchy for that block (an
            `EliminatedHierarchy` or `Vector{HierarchyLevel}`) when the component
            has ≥ 2 nodes, or the scalar diagonal value (`Float64`) when the
            component is a single vertex (solved directly).
- `n`     : original problem size.
"""
struct DisconnectedHierarchy
    ind::Vector{Vector{Int64}}
    blocks::Vector{Any}
    n::Int64
end

"""
    build_disconnected_hierarchy(A; eliminate = true) -> DisconnectedHierarchy | nothing

Detect the connected components of `A` (on the original matrix, before any
augmentation). Return `nothing` when `A` is connected — the caller then uses the
ordinary single-hierarchy path. Otherwise build a per-component hierarchy: a
single-vertex component stores its diagonal for a direct solve; a larger
component builds the normal (optionally degree-1/2-eliminated) hierarchy on its
principal submatrix `A[idx, idx]`.
"""
function build_disconnected_hierarchy(A::SparseMatrixCSC; eliminate::Bool = true)
    local comp = components(dropzeros(A))    # pattern-only; drop stored zeros so
                                             # a zero entry cannot bridge components
    local nc = Int(maximum(comp))
    nc == 1 && return nothing
    local idxsets = vecToComps(comp)
    local blocks = Vector{Any}(undef, nc)
    @inbounds for k = 1:nc
        local idx = idxsets[k]
        if length(idx) == 1
            blocks[k] = Float64(A[idx[1], idx[1]])
        else
            local Ab = A[idx, idx]
            blocks[k] = eliminate ? build_eliminated_hierarchy(Ab) :
                        build_hierarchy(Ab, validateInput!(Ab))
        end
    end
    return DisconnectedHierarchy([Vector{Int64}(s) for s in idxsets], blocks, size(A, 1))
end

"""
    (x, stats) = cmg_solve(DH::DisconnectedHierarchy, b; tol, maxit, cycle, theta, inner_tol, collect_stats)

Solve `A x = b` for a disconnected `A` by solving each connected component's
block independently and scattering the results back. `stats` aggregates the
per-block solves: `iterations` and `relres` are the worst (max) over blocks and
`converged` is the conjunction. A single-vertex Laplacian component (zero
diagonal) is pinned to `0`; a single-vertex SDD component is solved directly.
"""
function cmg_solve(
    DH::DisconnectedHierarchy,
    b::AbstractVector{<:Real};
    tol::Float64 = 1e-8,
    maxit::Int64 = 500,
    cycle::Symbol = :kcycle,
    theta::Float64 = 0.75,
    inner_tol::Float64 = 0.25,
    collect_stats::Bool = false,
)
    if length(b) != DH.n
        throw(DimensionMismatch("length(b) = $(length(b)), expected $(DH.n)"))
    end
    local x = zeros(Float64, DH.n)
    local iters = 0
    local relres = 0.0
    local converged = true
    for k = 1:length(DH.ind)
        local idx = DH.ind[k]
        if length(idx) == 1
            local v = idx[1]
            local d = DH.blocks[k]::Float64
            x[v] = abs(d) < 1e-14 ? 0.0 : Float64(b[v]) / d   # isolated Laplacian node -> 0
        else
            local xb, st = cmg_solve(
                DH.blocks[k],
                Vector{Float64}(b[idx]);
                tol = tol,
                maxit = maxit,
                cycle = cycle,
                theta = theta,
                inner_tol = inner_tol,
                collect_stats = collect_stats,
            )
            @inbounds for j = 1:length(idx)
                x[idx[j]] = xb[j]
            end
            iters = max(iters, st.iterations)
            relres = max(relres, st.relres)
            converged &= st.converged
        end
    end
    return (x, CMGStats(iters, relres, converged, Int64[]))
end
