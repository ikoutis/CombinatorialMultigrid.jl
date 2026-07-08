# Runtime comparison of the contraction variants. Standalone:
#     julia --project=. experiments/contraction/bench_contraction.jl
#     julia --project=. experiments/contraction/bench_contraction.jl --quick
#
# For each (matrix, clustering) configuration, verifies that all three
# variants agree, then reports the minimum of `REPS` timed runs each
# (one untimed warmup; GC.gc() before every timed run).

using Random
using SparseArrays
using LinearAlgebra
using Printf
using CombinatorialMultigrid
const CMG = CombinatorialMultigrid

include(joinpath(@__DIR__, "contract.jl"))

const QUICK = "--quick" in ARGS
const REPS = 3

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

function near_tree_lap(n::Int, k::Int)
    local I = Int64[]; local J = Int64[]; local V = Float64[]
    sizehint!(I, 2 * (n + k)); sizehint!(J, 2 * (n + k)); sizehint!(V, 2 * (n + k))
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

function random_clustering(n::Int, r::Int)
    local nc = max(1, n ÷ r)
    local cI = rand(1:nc, n)
    cI[1:nc] .= 1:nc
    shuffle!(cI)
    return cI, nc
end

l1(M) = sum(abs, M)
agree(M1, M2) = l1(M1 - M2) <= 1e-12 * max(l1(M1), 1.0)

function time_min(f, args...)
    f(args...)                        # warmup (also JIT on first config)
    local best = Inf
    for _ = 1:REPS
        GC.gc()
        local t = @elapsed f(args...)
        best = min(best, t)
    end
    return best
end

function run_config(tag, A, cI, nc)
    local M0 = contract_matmul(A, cI, nc)
    local M1 = contract_coo(A, cI, nc)
    local M2 = contract_spa(A, cI, nc)
    (agree(M0, M1) && agree(M0, M2)) ||
        error("variants disagree on $tag — fix correctness before timing")
    local tm = time_min(contract_matmul, A, cI, nc)
    local tc = time_min(contract_coo, A, cI, nc)
    local ts = time_min(contract_spa, A, cI, nc)
    @printf("%-26s %9d %10d %8d | %9.4f %9.4f %9.4f | %6.1fx %6.1fx\n",
            tag, size(A, 1), nnz(A), nc, tm, tc, ts, tm / tc, tm / ts)
end

Random.seed!(1)

const SIZES2D = QUICK ? [(300, 300)] : [(1000, 1000), (2000, 2000)]
const SIZE3D = QUICK ? (30, 30, 30) : (120, 120, 120)
const NTREE = QUICK ? 100_000 : 1_000_000

println("contraction benchmark: min of $REPS runs (seconds); speedup vs matmul")
@printf("%-26s %9s %10s %8s | %9s %9s %9s | %6s %6s\n",
        "config", "n", "nnz", "nc", "matmul", "coo", "spa", "coo↑", "spa↑")
println(repeat("-", 112))

for (nx, ny) in SIZES2D
    local A = grid2_sdd(nx, ny)
    for r in (2, 8, 32)
        local cI, nc = random_clustering(size(A, 1), r)
        run_config("grid2 $(nx)x$(ny) r=$r", A, cI, nc)
    end
    local cI, nc = CMG.steiner_group(A, Array(diag(A)))
    run_config("grid2 $(nx)x$(ny) steiner", A, cI, nc)
end

let (nx, ny, nz) = SIZE3D
    local A = grid3_sdd(nx, ny, nz)
    for r in (2, 8, 32)
        local cI, nc = random_clustering(size(A, 1), r)
        run_config("grid3 $(nx)^3 r=$r", A, cI, nc)
    end
    local cI, nc = CMG.steiner_group(A, Array(diag(A)))
    run_config("grid3 $(nx)^3 steiner", A, cI, nc)
end

let n = NTREE
    local A = near_tree_lap(n, n ÷ 1000)
    for r in (2, 8, 32)
        local cI, nc = random_clustering(n, r)
        run_config("near-tree n=$(n) r=$r", A, cI, nc)
    end
    local cI, nc = CMG.steiner_group(A, Array(diag(A)))
    run_config("near-tree n=$(n) steiner", A, cI, nc)
end

println("\ncoo↑ / spa↑ = matmul time over that variant's time (higher = faster than matmul).")
println("The steiner rows use CMG's real clustering (CMG.steiner_group), i.e. the exact")
println("cI shape the build_hierarchy contraction sees.")
