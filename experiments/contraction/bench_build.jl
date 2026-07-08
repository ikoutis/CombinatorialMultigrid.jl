# A/B build-timing for the contraction swap. Standalone:
#     julia --project=. experiments/contraction/bench_build.jl
#     julia --project=. experiments/contraction/bench_build.jl --quick
#
# Times the full CMG hierarchy build (cmg_preconditioner_lap, :kcycle), which
# drives the level contraction once per level. Compare the reported build times
# against a run where src/cmgAlg.jl is toggled back to the `Rt * A * Rt'` matmul
# (comment A_ = contract_coo(...), uncomment the two Rt/product lines, restart
# Julia). A quick solve is run each time as a correctness sanity check.

using Random
using SparseArrays
using LinearAlgebra
using Printf
using CombinatorialMultigrid

const QUICK = "--quick" in ARGS
const REPS = 3

tridiag_sdd(n::Int; w::Float64 = 1.0) =
    spdiagm(-1 => fill(-w, n - 1), 0 => fill(2w, n), 1 => fill(-w, n - 1))

function grid2_sdd(nx::Int, ny::Int; shift::Float64 = 0.01)
    local A =
        kron(sparse(1.0I, ny, ny), tridiag_sdd(nx)) +
        kron(tridiag_sdd(ny), sparse(1.0I, nx, nx))
    return A + shift * sparse(1.0I, nx * ny, nx * ny)
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

relres(A, x, b) = norm(A * x - b) / norm(b)

function build_min(L)
    cmg_preconditioner_lap(L; cycle = :kcycle)      # warmup / JIT
    local best = Inf
    for _ = 1:REPS
        GC.gc()
        local t = @elapsed cmg_preconditioner_lap(L; cycle = :kcycle)
        best = min(best, t)
    end
    return best
end

function run_case(tag, L)
    local n = size(L, 1)
    local tb = build_min(L)
    # correctness sanity: solve once
    local b = randn(n); b .-= sum(b) / n
    local (_, H) = cmg_preconditioner_lap(L; cycle = :kcycle)
    local (x, stats) = cmg_solve(H, b; cycle = :kcycle)
    @printf("%-26s %9d %10d | build %8.4f s | solve its=%3d relres=%.1e conv=%s\n",
            tag, n, nnz(L), tb, stats.iterations, relres(L, x, b), stats.converged)
end

Random.seed!(1)

println("CMG hierarchy build timing (min of $REPS, :kcycle) — contraction is in the build path")
println("Toggle src/cmgAlg.jl (contract_coo vs Rt*A*Rt') + restart Julia to A/B.\n")

let g = QUICK ? (300, 300) : (1000, 1000)
    run_case("grid2 $(g[1])x$(g[2])", grid2_sdd(g[1], g[2]))
end
let n = QUICK ? 100_000 : 1_000_000
    run_case("near-tree n=$(n)", near_tree_lap(n, n ÷ 1000))
end
