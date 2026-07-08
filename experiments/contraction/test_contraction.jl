# Correctness tests for the contraction experiment. Standalone:
#     julia --project=. experiments/contraction/test_contraction.jl
# (from the repo root; the project provides SparseArrays/Laplacians/CMG)

using Test
using Random
using SparseArrays
using LinearAlgebra
using CombinatorialMultigrid
const CMG = CombinatorialMultigrid

include(joinpath(@__DIR__, "contract.jl"))

## builders (local copies — the experiment stays standalone)

tridiag_sdd(n::Int; w::Float64 = 1.0) =
    spdiagm(-1 => fill(-w, n - 1), 0 => fill(2w, n), 1 => fill(-w, n - 1))

function grid2_sdd(nx::Int, ny::Int; shift::Float64 = 0.01)
    local A =
        kron(sparse(1.0I, ny, ny), tridiag_sdd(nx)) +
        kron(tridiag_sdd(ny), sparse(1.0I, nx, nx))
    return A + shift * sparse(1.0I, nx * ny, nx * ny)
end

function grid3_sdd(nx::Int, ny::Int, nz::Int; shift::Float64 = 0.01)
    local Ixy = sparse(1.0I, nx * ny, nx * ny)
    local A =
        kron(sparse(1.0I, nz, nz), grid2_sdd(nx, ny; shift = 0.0)) +
        kron(tridiag_sdd(nz), Ixy)
    return A + shift * sparse(1.0I, nx * ny * nz, nx * ny * nz)
end

function grid2_lap(nx::Int, ny::Int)
    local A = grid2_sdd(nx, ny; shift = 0.0)
    return A - spdiagm(0 => vec(sum(A, dims = 2)))
end

function near_tree_lap(n::Int, k::Int)
    local I = Int64[]; local J = Int64[]; local V = Float64[]
    for v = 2:n
        local p = rand(1:v-1); local w = rand() + 0.5
        push!(I, v); push!(J, p); push!(V, w)
        push!(I, p); push!(J, v); push!(V, w)
    end
    for _ = 1:k
        local i = rand(1:n); local j = rand(1:n)
        if i != j
            local w = rand() + 0.5
            push!(I, i); push!(J, j); push!(V, w)
            push!(I, j); push!(J, i); push!(V, w)
        end
    end
    local A = sparse(I, J, V, n, n)
    return spdiagm(0 => vec(sum(A, dims = 2))) - A
end

function random_sdd(n::Int, p::Float64)
    local A = sprand(n, n, p)
    A = -abs.(A + A')
    A = A - spdiagm(0 => diag(A))
    return A - spdiagm(0 => vec(sum(A, dims = 2))) +
           spdiagm(0 => rand(n) .* 0.5 .+ 0.01)
end

# random surjective clustering with coarsening ratio ~r
function random_clustering(n::Int, r::Int)
    local nc = max(1, n ÷ r)
    local cI = rand(1:nc, n)
    cI[1:nc] .= 1:nc            # every cluster nonempty
    shuffle!(cI)
    return cI, nc
end

l1(M) = sum(abs, M)
agree(M1, M2) = l1(M1 - M2) <= 1e-12 * max(l1(M1), 1.0)

function check_all(A, cI, nc, tag)
    local M0 = contract_matmul(A, cI, nc)
    local M1 = contract_coo(A, cI, nc)
    local M2 = contract_spa(A, cI, nc)
    @test agree(M0, M1)
    @test agree(M0, M2)
    @test agree(M2, M2')                    # symmetric
    # contraction preserves per-cluster row sums (Laplacian in -> Laplacian out)
    local rs_fine = vec(sum(A, dims = 2))
    local rs_grouped = zeros(nc)
    for i = 1:size(A, 1)
        rs_grouped[cI[i]] += rs_fine[i]
    end
    @test maximum(abs.(rs_grouped .- vec(sum(M2, dims = 2)))) <=
          1e-10 * max(l1(M0), 1.0)
    return M2
end

Random.seed!(1)

@testset "contraction correctness" begin

    local cases = [
        ("grid2 sdd 40x40", grid2_sdd(40, 40)),
        ("grid2 lap 40x40", grid2_lap(40, 40)),
        ("grid3 sdd 12x12x12", grid3_sdd(12, 12, 12)),
        ("near-tree lap n=5000 k=50", near_tree_lap(5000, 50)),
        ("random sdd n=2000", random_sdd(2000, 0.004)),
    ]

    for (tag, A) in cases
        local n = size(A, 1)
        @testset "$tag" begin
            for r in (2, 8, 32)
                local cI, nc = random_clustering(n, r)
                check_all(A, cI, nc, "$tag r=$r")
            end
            # identity clustering: contraction must equal A exactly
            local Mid = contract_spa(A, collect(1:n), n)
            @test l1(Mid - A) == 0.0
            # all-in-one: 1x1 matrix holding the total excess
            local M1x1 = contract_spa(A, fill(1, n), 1)
            @test abs(M1x1[1, 1] - sum(A)) <= 1e-10 * max(abs(sum(A)), 1.0)
        end
    end

    @testset "real CMG clustering (steiner_group)" begin
        local A = grid2_sdd(120, 120)
        local cI, nc = CMG.steiner_group(A, Array(diag(A)))
        @test nc < size(A, 1)
        check_all(A, cI, nc, "steiner grid2 120x120")

        local L = near_tree_lap(20_000, 200)
        local cIt, nct = CMG.steiner_group(L, Array(diag(L)))
        @test nct < size(L, 1)
        check_all(L, cIt, nct, "steiner near-tree")
    end

end

println("contraction correctness: all tests passed")
