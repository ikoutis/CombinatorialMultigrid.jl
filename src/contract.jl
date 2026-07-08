# Combinatorial Laplacian contraction used by build_hierarchy.
#
# `contract_coo(A, cI, nc)` computes the level contraction
#     A'[C, D] = sum_{i in C, j in D} A[i, j]
# i.e. the sparse triple product `Rt * A * Rt'` with the restriction
# `Rt = sparse(cI, 1:n, 1, nc, n)` — but combinatorially, in one pass over the
# stored entries: every entry A[i,j] is relabeled to (cI[i], cI[j]) and the
# `sparse` constructor sums the entries that land on the same coarse pair
# (inter-cluster weights add; intra-cluster entries fold onto the coarse
# diagonal). This avoids the two SpGEMMs.
#
# Chosen over the sparse-accumulator variant because it is the robust
# all-rounder across clustering index-locality (see experiments/contraction/
# for the three-way study and benchmark). Sequential; the entry loop is
# trivially parallelizable if needed later.

"""
    contract_coo(A, cI, nc) -> SparseMatrixCSC

Contract the `n × n` matrix `A` under the cluster map `cI : [n] -> [nc]`,
returning the `nc × nc` matrix `A'[C,D] = sum_{i∈C, j∈D} A[i,j]`. Equivalent to
`Rt * A * Rt'` with `Rt = sparse(cI, 1:n, 1, nc, n)`, computed as a single
mapped-triplet pass plus a duplicate-summing `sparse` build.
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
