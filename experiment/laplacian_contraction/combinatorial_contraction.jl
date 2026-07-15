# Experimental, standalone Laplacian contraction routines.
#
# This file is intentionally not wired into the solver. It provides a sequential
# combinatorial contraction implementation to compare against the algebraic
# sparse matrix product used in the hierarchy builder:
#
#     Rt = sparse(cI, 1:n, 1.0, nc, n)
#     Lc = Rt * L * Rt'
#
# where cI[i] is the coarse/cluster id of fine vertex i.

using LinearAlgebra
using SparseArrays

"""
    contract_laplacian_combinatorial(L, cI[, nc]; check = true)

Sequentially contract a fine-level Laplacian `L` according to cluster labels
`cI`, returning the coarse Laplacian whose cluster-to-cluster edge weights are
sums of fine edge weights crossing those clusters.

This routine assumes `L` is symmetric Laplacian-like with non-positive
off-diagonal entries. Internal edges inside a cluster vanish under contraction;
each crossing fine edge of weight `w = -L[i,j]` contributes:

- `+w` to the two coarse diagonals, and
- `-w` to both coarse off-diagonal entries.

The result should match `sparse(cI, 1:n, 1.0, nc, n) * L * sparse(cI, 1:n, 1.0, nc, n)'`
for Laplacians, up to floating-point roundoff.
"""
function contract_laplacian_combinatorial(
    L::SparseMatrixCSC{Tv,Ti},
    cI::AbstractVector{<:Integer},
    nc::Integer = maximum(cI);
    check::Bool = true,
) where {Tv<:Real,Ti<:Integer}
    n, m = size(L)
    if check
        n == m || throw(DimensionMismatch("L must be square, got $(size(L))"))
        length(cI) == n || throw(DimensionMismatch("length(cI) = $(length(cI)), expected $n"))
        nc >= 0 || throw(ArgumentError("nc must be nonnegative"))
        for (i, c) in pairs(cI)
            1 <= c <= nc || throw(ArgumentError("cI[$i] = $c is outside 1:$nc"))
        end
    end

    local diag = zeros(Float64, nc)
    local I = Int[]
    local J = Int[]
    local V = Float64[]

    # At most one undirected contribution per stored off-diagonal pair. The
    # final sparse construction sums duplicate coarse edges between clusters.
    sizehint!(I, nnz(L))
    sizehint!(J, nnz(L))
    sizehint!(V, nnz(L))

    local rows = rowvals(L)
    local vals = nonzeros(L)
    @inbounds for j = 1:n
        local cj = Int(cI[j])
        for p in nzrange(L, j)
            local i = rows[p]
            i < j || continue
            local ci = Int(cI[i])
            ci == cj && continue

            local w = -Float64(vals[p])
            if check && w < 0.0
                throw(ArgumentError("positive off-diagonal at ($i, $j): L[i,j] = $(vals[p])"))
            end

            diag[ci] += w
            diag[cj] += w
            push!(I, ci); push!(J, cj); push!(V, -w)
            push!(I, cj); push!(J, ci); push!(V, -w)
        end
    end

    @inbounds for c = 1:nc
        push!(I, c); push!(J, c); push!(V, diag[c])
    end

    return sparse(I, J, V, Int(nc), Int(nc))
end

"""
    contract_laplacian_matmul(L, cI[, nc])

Baseline contraction using the sparse matrix product currently used by the
hierarchy builder. `Rt` has shape `nc × n` and maps fine vertices to clusters.
"""
function contract_laplacian_matmul(
    L::SparseMatrixCSC,
    cI::AbstractVector{<:Integer},
    nc::Integer = maximum(cI),
)
    local n = size(L, 1)
    local Rt = sparse(Int.(cI), collect(1:n), ones(Float64, n), Int(nc), n)
    return Rt * L * Rt'
end

"""
    max_abs_diff(A, B)

Maximum absolute entrywise difference between sparse matrices `A` and `B`.
"""
max_abs_diff(A::SparseMatrixCSC, B::SparseMatrixCSC) =
    isempty(nonzeros(A - B)) ? 0.0 : maximum(abs, nonzeros(A - B))
