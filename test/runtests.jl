using Test
using LinearAlgebra
using SparseArrays
using Random
using Laplacians
using CombinatorialMultigrid

## grid builders (sized to cross the n >= 500 coarsening threshold)

# 1D SDD: tridiagonal with constant diagonal 2w (boundary rows strictly dominant)
tridiag_sdd(n::Int; w::Float64 = 1.0) =
    spdiagm(-1 => fill(-w, n - 1), 0 => fill(2w, n), 1 => fill(-w, n - 1))

# 2D (optionally anisotropic) SDD grid via kron sums, plus a diagonal shift
function grid2_sdd(nx::Int, ny::Int; wx::Float64 = 1.0, wy::Float64 = 1.0, shift::Float64 = 0.01)
    local A =
        kron(sparse(1.0I, ny, ny), tridiag_sdd(nx; w = wx)) +
        kron(tridiag_sdd(ny; w = wy), sparse(1.0I, nx, nx))
    return A + shift * sparse(1.0I, nx * ny, nx * ny)
end

function grid3_sdd(
    nx::Int,
    ny::Int,
    nz::Int;
    wx::Float64 = 1.0,
    wy::Float64 = 1.0,
    wz::Float64 = 1.0,
    shift::Float64 = 0.01,
)
    local Ixy = sparse(1.0I, nx * ny, nx * ny)
    local A =
        kron(sparse(1.0I, nz, nz), grid2_sdd(nx, ny; wx = wx, wy = wy, shift = 0.0)) +
        kron(tridiag_sdd(nz; w = wz), Ixy)
    return A + shift * sparse(1.0I, nx * ny * nz, nx * ny * nz)
end

# graph Laplacian of the uniform 2D grid (exact zero row sums)
function grid2_lap(nx::Int, ny::Int)
    local A = grid2_sdd(nx, ny; shift = 0.0)
    return A - spdiagm(0 => vec(sum(A, dims = 2)))
end

relres(A, x, b) = norm(A * x - b) / norm(b)

const CMG = CombinatorialMultigrid

@testset "CombinatorialMultigrid" begin

    @testset "legacy path regression" begin
        Random.seed!(0)
        local A = grid2_sdd(40, 40)
        local b = randn(size(A, 1))

        local (pfunc, H) = cmg_preconditioner_lap(A)
        @test H isa Vector{CMG.HierarchyLevel}
        local f = pcgSolver(A, pfunc)
        local x = f(b, maxits = 200, tol = 1e-8)
        @test relres(A, x, b) < 1e-6

        # adjacency entry point
        local Adj = sprand(600, 600, 0.01)
        Adj = Adj + Adj'
        Adj = Adj - spdiagm(0 => diag(Adj))
        local (pfunc2, _) = cmg_preconditioner_adj(Adj)
        local L = lap(Adj)
        local b2 = randn(600)
        b2 .-= sum(b2) / length(b2)
        local x2 = pfunc2(b2)
        @test all(isfinite, x2)
    end

    @testset "budget rule invariants" begin
        for A in (grid2_sdd(60, 60), grid3_sdd(14, 14, 14))
            local A_ = CMG.validateInput!(A)
            local H = CMG.build_hierarchy(A, A_)
            local L = length(H)
            local theta = 0.75

            # masses in the budget rule's convention
            local m = [Float64(CMG.nnz_lower(h.A)) for h in H]
            if !H[L].iterative
                m[L] = Float64(nnz(H[L].chol.L) + (H[L].n - 1))
            end

            local krepeat = CMG.compute_kcycle_repeats(H, theta)
            @test all(krepeat .>= 1)
            for k = 1:L-1
                if H[k].nc >= 1
                    @test krepeat[k] <= max(H[k].nc, 1)
                end
            end
            # work-cap invariant: N_k * m_k <= theta^(k-1) * m_1 (up to the
            # clamp-to-1 floor, which the max() below accounts for)
            local N = 1.0
            for k = 2:L
                N *= krepeat[k-1]
                @test N * m[k] <= max(theta^(k - 1) * m[1], m[k]) * 1.001
            end

            # theta = 0 reproduces the local repeat rule
            local kr0 = CMG.compute_kcycle_repeats(H, 0.0)
            for k = 1:L-1
                @test kr0[k] == max(floor(Int64, m[k] / max(m[k+1], 1.0) - 1), 1)
            end
        end
    end

    @testset "kcycle correctness 1D/2D/3D" begin
        Random.seed!(1)
        for A in (
            tridiag_sdd(800) + 0.01 * sparse(1.0I, 800, 800),
            grid2_sdd(30, 30),
            grid3_sdd(12, 12, 12; wy = 0.3, wz = 2.5),
        )
            local n = size(A, 1)
            local x_true = randn(n)
            local b = A * x_true
            local (x, stats) = cmg_solve(A, b; cycle = :kcycle)
            @test stats.converged
            @test relres(A, x, b) < 1e-6
        end
    end

    @testset "kcycle vs vcycle iterations" begin
        Random.seed!(2)
        local A = grid3_sdd(16, 16, 16)
        local b = randn(size(A, 1))
        local (xk, sk) = cmg_solve(A, b; cycle = :kcycle)
        local (xv, sv) = cmg_solve(A, b; cycle = :vcycle)
        @test sk.converged && sv.converged
        @test relres(A, xk, b) < 1e-6
        @test relres(A, xv, b) < 1e-6
        # Krylov acceleration should not lose to stationary repetition
        @test sk.iterations <= sv.iterations + 5
    end

    @testset "opt-out (theta = 0, inner_tol = 0)" begin
        Random.seed!(3)
        local A = grid2_sdd(40, 40; wy = 50.0)
        local b = randn(size(A, 1))
        local (x, stats) = cmg_solve(A, b; cycle = :kcycle, theta = 0.0, inner_tol = 0.0)
        @test stats.converged
        @test relres(A, x, b) < 1e-6
    end

    @testset "visit stats" begin
        Random.seed!(4)
        local A = grid2_sdd(60, 60)
        local b = randn(size(A, 1))

        local (x, stats) = cmg_solve(A, b; cycle = :kcycle, collect_stats = true)
        @test stats.converged
        @test length(stats.level_visits) >= 1
        @test stats.level_visits[1] == stats.iterations

        # budget mode does at least as many coarse visits per outer iteration
        # as the local-repeat opt-out at the second level
        local (_, s_local) =
            cmg_solve(A, b; cycle = :kcycle, theta = 0.0, inner_tol = 0.0, collect_stats = true)
        if length(stats.level_visits) >= 2 && s_local.iterations > 0 && stats.iterations > 0
            @test stats.level_visits[2] >= stats.iterations  # at least one inner step each
        end

        local (_, sv) = cmg_solve(A, b; cycle = :vcycle, collect_stats = true)
        @test isempty(sv.level_visits)
    end

    @testset "SDD strong-shift path (augment/extract)" begin
        Random.seed!(5)
        local A = grid2_sdd(25, 25; shift = 0.5)
        local n = size(A, 1)
        local b = randn(n)
        local x_ref = Matrix(A) \ b
        for cycle in (:kcycle, :vcycle)
            local (x, stats) = cmg_solve(A, b; cycle = cycle, tol = 1e-10)
            @test stats.converged
            @test norm(x - x_ref) / norm(x_ref) < 1e-6
        end
    end

    @testset "pure Laplacian path" begin
        Random.seed!(6)
        local L = grid2_lap(30, 30)
        @test maximum(abs.(sum(L, dims = 2))) < 1e-12
        local n = size(L, 1)
        local b = randn(n)
        b .-= sum(b) / n  # solvable rhs (orthogonal to the null space)
        local (x, stats) = cmg_solve(L, b; cycle = :kcycle)
        @test stats.converged
        @test relres(L, x, b) < 1e-6
    end

    @testset "preconditioner knob API" begin
        Random.seed!(7)
        local A = grid2_sdd(30, 30)
        local n = size(A, 1)
        local b = randn(n)

        # default is unchanged and PCG-safe
        local (pfunc_v, H) = cmg_preconditioner_lap(A)
        local f = pcgSolver(A, pfunc_v)
        local x = f(b, maxits = 200, tol = 1e-8)
        @test relres(A, x, b) < 1e-6

        # :kcycle knob returns a (nonlinear) single-apply closure
        local (pfunc_k, Hk) = cmg_preconditioner_lap(A; cycle = :kcycle)
        local x1 = pfunc_k(b)
        @test length(x1) == n
        @test all(isfinite, x1)
        @test norm(A * x1 - b) < norm(b)  # one sweep reduces the residual

        @test_throws ArgumentError cmg_preconditioner_lap(A; cycle = :wcycle)
        @test_throws ArgumentError cmg_solve(Hk, b; cycle = :wcycle)
        @test_throws DimensionMismatch cmg_solve(Hk, randn(n + 7))

        # hierarchy reuse: solve directly on the prebuilt hierarchy
        local (x2, stats2) = cmg_solve(Hk, b)
        @test stats2.converged
        @test relres(A, x2, b) < 1e-6
    end

end

# optional timing script on the large example matrix (requires MAT.jl and
# example/X.mat; see the header of xmat_timing.jl)
if get(ENV, "CMG_TEST_XMAT", "0") == "1"
    include("xmat_timing.jl")
end
