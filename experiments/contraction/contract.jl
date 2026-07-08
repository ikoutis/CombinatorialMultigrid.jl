# Laplacian contraction routines — standalone experiment, NOT wired into the
# solver. See README.md in this directory.
#
# The operation: given the fine Laplacian/SDD matrix `A` (n x n) and a cluster
# map `cI : [n] -> [nc]`, compute the contracted matrix
#
#     A'[C, D] = sum_{i in C, j in D} A[i, j]
#
# which the CMG build currently obtains as the sparse triple product
# `Rt * A * Rt'` with `Rt = sparse(cI, 1:n, 1, nc, n)`
# (src/cmgAlg.jl, build_hierarchy). Combinatorially: every fine edge (i, j, w)
# with cI[i] != cI[j] adds w to the coarse edge (cI[i], cI[j]) — inter-cluster
# weights sum; intra-cluster edges cancel off-diagonally and fold into the
# coarse diagonal together with any SDD excess.
#
# All three routines share the signature
#     contract_*(A::SparseMatrixCSC, cI::Vector{<:Integer}, nc::Integer)
# and return the nc x nc contracted SparseMatrixCSC. Sequential by design;
# parallelization is a possible follow-up.

using SparseArrays
using LinearAlgebra

"""
    contract_matmul(A, cI, nc)

The baseline: the verbatim expression from `build_hierarchy`, including the
integer-valued restriction matrix, so benchmarks measure exactly what the CMG
build pays today. Two sparse matrix-matrix products.
"""
function contract_matmul(A::SparseMatrixCSC, cI::Vector{<:Integer}, nc::Integer)
    local n = size(A, 2)
    local Rt = sparse(cI, 1:n, 1, nc, n)
    return Rt * A * Rt'
end

"""
    contract_coo(A, cI, nc)

Mapped triplets: one pass over the CSC emitting `(cI[row], cI[col], val)`
arrays, then a single `sparse(I, J, V, nc, nc)` call, whose constructor sums
duplicate entries in linear time. Minimal code; allocates three nnz-length
arrays for the intermediate COO form.
"""
function contract_coo(A::SparseMatrixCSC, cI::Vector{<:Integer}, nc::Integer)
    local rows = rowvals(A)
    local vals = nonzeros(A)
    local nz = nnz(A)
    local I = Vector{Int64}(undef, nz)
    local J = Vector{Int64}(undef, nz)
    local V = Vector{Float64}(undef, nz)
    local t = 0
    @inbounds for j = 1:size(A, 2)
        local cj = cI[j]
        for r in nzrange(A, j)
            t += 1
            I[t] = cI[rows[r]]
            J[t] = cj
            V[t] = vals[r]
        end
    end
    return sparse(I, J, V, nc, nc)
end

"""
    contract_spa(A, cI, nc)

Fully combinatorial contraction — no intermediate COO, no matmul:

1. counting-sort the fine columns into coarse buckets (`O(n + nc)`);
2. for each coarse column `D`, scan the CSC entries of its fine columns once,
   map rows through `cI`, and sum duplicates with a mark/slot sparse
   accumulator stamped by `D` (each coarse column is visited exactly once, so
   the stamp needs no epoch counter);
3. append the deduped entries into growing `rowval`/`nzval` with `colptr`
   tracking, sorting each column's (row, value) slice through a reusable pair
   buffer so `rowval` is sorted within columns as `SparseMatrixCSC` requires.

Total work `O(n + nc + nnz(A))` plus the per-column sorts of the coarse
columns (whose total size is `nnz(A')`).
"""
function contract_spa(A::SparseMatrixCSC, cI::Vector{<:Integer}, nc::Integer)
    local n = size(A, 2)
    local rows = rowvals(A)
    local vals = nonzeros(A)

    # 1. bucket fine columns by coarse column (counting sort)
    local cptr = zeros(Int64, nc + 1)
    @inbounds for j = 1:n
        cptr[cI[j]+1] += 1
    end
    cptr[1] = 1
    @inbounds for c = 1:nc
        cptr[c+1] += cptr[c]
    end
    local ccols = Vector{Int64}(undef, n)
    local fill_ = copy(cptr)
    @inbounds for j = 1:n
        local c = cI[j]
        ccols[fill_[c]] = j
        fill_[c] += 1
    end

    # 2 + 3. per-coarse-column sparse accumulation into a growing CSC
    local mark = zeros(Int64, nc)
    local slot = zeros(Int64, nc)
    local colptr = Vector{Int64}(undef, nc + 1)
    local rowval = Int64[]
    local nzval = Float64[]
    sizehint!(rowval, nnz(A))
    sizehint!(nzval, nnz(A))
    local pairbuf = Vector{Tuple{Int64,Float64}}()

    colptr[1] = 1
    @inbounds for D = 1:nc
        local start = length(rowval) + 1
        for t = cptr[D]:(cptr[D+1]-1)
            local j = ccols[t]
            for r in nzrange(A, j)
                local ci = Int64(cI[rows[r]])
                if mark[ci] != D
                    mark[ci] = D
                    push!(rowval, ci)
                    push!(nzval, Float64(vals[r]))
                    slot[ci] = length(rowval)
                else
                    nzval[slot[ci]] += Float64(vals[r])
                end
            end
        end
        # sort this column's slice by row index (SparseMatrixCSC requires it)
        local len = length(rowval) - start + 1
        if len > 1
            resize!(pairbuf, len)
            for a = 1:len
                pairbuf[a] = (rowval[start+a-1], nzval[start+a-1])
            end
            sort!(pairbuf; by = first)
            for a = 1:len
                rowval[start+a-1] = pairbuf[a][1]
                nzval[start+a-1] = pairbuf[a][2]
            end
        end
        colptr[D+1] = length(rowval) + 1
    end

    return SparseMatrixCSC(Int64(nc), Int64(nc), colptr, rowval, nzval)
end
