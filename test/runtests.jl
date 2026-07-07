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

# random weighted spanning-tree adjacency on n nodes (connected)
function random_tree_adj(n::Int)
    local I = Int64[]
    local J = Int64[]
    local V = Float64[]
    for v = 2:n
        local p = rand(1:v-1)
        local w = rand() + 0.5
        push!(I, v); push!(J, p); push!(V, w)
        push!(I, p); push!(J, v); push!(V, w)
    end
    return sparse(I, J, V, n, n)
end

# spanning tree plus `k` extra random (off-tree) edges
function tree_plus_offtree_adj(n::Int, k::Int)
    local A = random_tree_adj(n)
    local extra = Set{Tuple{Int64,Int64}}()
    while length(extra) < k
        local i = rand(1:n)
        local j = rand(1:n)
        if i != j
            push!(extra, (min(i, j), max(i, j)))
        end
    end
    local I = Int64[]
    local J = Int64[]
    local V = Float64[]
    for (i, j) in extra
        local w = rand() + 0.5
        push!(I, i); push!(J, j); push!(V, w)
        push!(I, j); push!(J, i); push!(V, w)
    end
    return A + sparse(I, J, V, n, n)
end

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

    @testset "degree-1/2 elimination: pure tree (exact)" begin
        Random.seed!(10)
        for n in (16, 300)
            local L = lap(random_tree_adj(n))
            @test maximum(abs.(sum(L, dims = 2))) < 1e-10
            # a pure tree collapses to the single null-space node
            local elims, ind, A_red, is_lap = CMG.eliminate_deg12(L)
            @test is_lap
            @test length(ind) == 1
            @test length(elims) == n - 1

            local b = randn(n)
            b .-= sum(b) / n
            for cycle in (:vcycle, :kcycle)
                local (pfunc, EH) = cmg_preconditioner_lap(L; cycle = cycle, eliminate = true)
                @test EH isa EliminatedHierarchy
                @test isempty(EH.H)                       # no CMG needed
                local (x, stats) = cmg_solve(EH, b; cycle = cycle)
                @test stats.converged
                @test relres(L, x, b) < 1e-8              # elimination alone is exact
            end
        end
    end

    @testset "degree-1/2 elimination: near-tree Laplacian (both cycles)" begin
        Random.seed!(11)
        for (n, k) in ((200, 10), (700, 30))
            local A = tree_plus_offtree_adj(n, k)
            local L = lap(A)
            local elims, ind, A_red, is_lap = CMG.eliminate_deg12(L)
            @test length(ind) < n                          # meaningful reduction
            @test size(A_red, 1) == length(ind)
            @test issymmetric(A_red)

            local b = randn(n)
            b .-= sum(b) / n
            for cycle in (:vcycle, :kcycle)
                # eliminate vs plain must agree
                local (_, EH) = cmg_preconditioner_lap(L; cycle = cycle, eliminate = true)
                local (xe, se) = cmg_solve(EH, b; cycle = cycle, tol = 1e-10)
                local (xp, sp) = cmg_solve(L, b; cycle = cycle, tol = 1e-10)
                @test se.converged
                @test relres(L, xe, b) < 1e-6
                # both are valid solutions of a singular system; compare residuals
                @test relres(L, xe, b) <= relres(L, xp, b) * 10 + 1e-8
            end
        end
    end

    @testset "degree-1/2 elimination: SDD near-tree" begin
        Random.seed!(12)
        local n = 300
        local L = lap(tree_plus_offtree_adj(n, 12))
        local A = L + 0.1 * sparse(1.0I, n, n)             # strictly dominant (SDD)
        local elims, ind, A_red, is_lap = CMG.eliminate_deg12(A)
        @test !is_lap
        local b = randn(n)
        local x_ref = Matrix(A) \ b
        for cycle in (:vcycle, :kcycle)
            local (pfunc, EH) = cmg_preconditioner_lap(A; cycle = cycle, eliminate = true)
            local (x, stats) = cmg_solve(EH, b; cycle = cycle, tol = 1e-10)
            @test stats.converged
            @test norm(x - x_ref) / norm(x_ref) < 1e-6
            # returned preconditioner closure is usable and finite
            local xp = pfunc(b)
            @test length(xp) == n
            @test all(isfinite, xp)
        end
    end

    @testset "degree-1/2 elimination: grid core + API guards" begin
        Random.seed!(13)
        local A = grid2_sdd(30, 30)                        # interior nodes stay degree >= 3
        local n = size(A, 1)
        local b = randn(n)
        local x_ref = Matrix(A) \ b
        local elims, ind, A_red, is_lap = CMG.eliminate_deg12(A)
        @test length(ind) >= 1
        local (_, EH) = cmg_preconditioner_lap(A; cycle = :kcycle, eliminate = true)
        local (x, stats) = cmg_solve(EH, b; cycle = :kcycle, tol = 1e-10)
        @test stats.converged
        @test norm(x - x_ref) / norm(x_ref) < 1e-6

        @test_throws ArgumentError cmg_solve(EH, b; cycle = :wcycle)
        @test_throws DimensionMismatch cmg_solve(EH, randn(n + 3))
    end

    @testset "degree-1/2 elimination: fill-merge structures" begin
        # structures that exercise fill edges landing on existing or previously
        # created edges (duplicate-neighbor summation in the elimination): pure
        # cycles, theta graphs (two hubs joined by several disjoint paths), and
        # a square with one diagonal. All collapse fully -> exact solve.
        Random.seed!(14)

        # pure cycles: every node degree 2; fills cascade into parallel edges
        for n in (3, 4, 10, 101)
            local I = Int64[]; local J = Int64[]; local V = Float64[]
            for i = 1:n
                local j = i == n ? 1 : i + 1
                local w = rand() + 0.5
                push!(I, i); push!(J, j); push!(V, w)
                push!(I, j); push!(J, i); push!(V, w)
            end
            local L = lap(sparse(I, J, V, n, n))
            local elims, ind, A_red, is_lap = CMG.eliminate_deg12(L)
            @test is_lap
            @test length(ind) == 1                 # cycle collapses fully
            @test length(elims) == n - 1
            local b = randn(n); b .-= sum(b) / n
            local (_, EH) = cmg_preconditioner_lap(L; eliminate = true)
            local (x, stats) = cmg_solve(EH, b)
            @test stats.converged
            @test relres(L, x, b) < 1e-8           # exact by elimination alone
        end

        # theta graph: hubs 1 and 2 joined by `npath` disjoint paths of length
        # `plen` (path collapse fills merge into parallel hub-hub edges)
        for (npath, plen) in ((3, 5), (5, 1), (2, 30))
            local I = Int64[]; local J = Int64[]; local V = Float64[]
            local nid = 2
            for _ = 1:npath
                local prev = 1
                for _ = 1:plen
                    nid += 1
                    local w = rand() + 0.5
                    push!(I, prev); push!(J, nid); push!(V, w)
                    push!(I, nid); push!(J, prev); push!(V, w)
                    prev = nid
                end
                local w = rand() + 0.5
                push!(I, prev); push!(J, 2); push!(V, w)
                push!(I, 2); push!(J, prev); push!(V, w)
            end
            local n = nid
            local L = lap(sparse(I, J, V, n, n))
            local elims, ind, _, _ = CMG.eliminate_deg12(L)
            @test length(ind) == 1                 # theta collapses fully
            local b = randn(n); b .-= sum(b) / n
            local (_, EH) = cmg_preconditioner_lap(L; eliminate = true)
            local (x, stats) = cmg_solve(EH, b)
            @test stats.converged
            @test relres(L, x, b) < 1e-8
        end

        # square + one diagonal: eliminating a corner fills exactly onto the
        # existing diagonal edge (duplicate summation on the base slice)
        local I = [1, 2, 2, 3, 3, 4, 4, 1, 1, 3]
        local J = [2, 1, 3, 2, 4, 3, 1, 4, 3, 1]
        local V = [1.3, 1.3, 0.7, 0.7, 1.1, 1.1, 0.9, 0.9, 2.0, 2.0]
        local L = lap(sparse(I, J, V, 4, 4))
        local b = randn(4); b .-= sum(b) / 4
        local (_, EH) = cmg_preconditioner_lap(L; eliminate = true)
        local (x, stats) = cmg_solve(EH, b)
        @test stats.converged
        @test relres(L, x, b) < 1e-10
    end

    @testset "degree-1/2 elimination: hub-fill stress + invalid input" begin
        Random.seed!(16)

        # broom: one hub with many chains hanging off it (pure leaf cascades)
        local I = Int64[]; local J = Int64[]; local V = Float64[]
        local nid = 1
        for _ = 1:50
            local prev = 1
            for _ = 1:8
                nid += 1
                local w = rand() + 0.5
                push!(I, prev); push!(J, nid); push!(V, w)
                push!(I, nid); push!(J, prev); push!(V, w)
                prev = nid
            end
        end
        local Lb = lap(sparse(I, J, V, nid, nid))
        local bb = randn(nid); bb .-= sum(bb) / nid
        local (_, EHb) = cmg_preconditioner_lap(Lb; eliminate = true)
        local (xb, sb) = cmg_solve(EHb, bb)
        @test sb.converged
        @test relres(Lb, xb, bb) < 1e-8

        # subdivided clique: K5 whose every edge is a 7-node path — chain
        # collapses repeatedly fill onto the same five hubs (stale-candidate
        # compactions); the survivors must be exactly the clique core
        I = Int64[]; J = Int64[]; V = Float64[]
        nid = 5
        for a = 1:5, c = (a+1):5
            local prev = a
            for _ = 1:7
                nid += 1
                local w = rand() + 0.5
                push!(I, prev); push!(J, nid); push!(V, w)
                push!(I, nid); push!(J, prev); push!(V, w)
                prev = nid
            end
            local w = rand() + 0.5
            push!(I, prev); push!(J, c); push!(V, w)
            push!(I, c); push!(J, prev); push!(V, w)
        end
        local Lk = lap(sparse(I, J, V, nid, nid))
        local elims, ind, A_red, is_lap = CMG.eliminate_deg12(Lk)
        @test ind == [1, 2, 3, 4, 5]
        @test length(elims) == nid - 5
        local bk = randn(nid); bk .-= sum(bk) / nid
        for cycle in (:vcycle, :kcycle)
            local (_, EHk) = cmg_preconditioner_lap(Lk; cycle = cycle, eliminate = true)
            local (xk, sk) = cmg_solve(EHk, bk; cycle = cycle)
            @test sk.converged
            @test relres(Lk, xk, bk) < 1e-8
        end

        # invalid input errors match the non-elimination path
        local Asym = sparse([1, 2], [2, 1], [1.0, 2.0], 2, 2) + 3.0 * sparse(1.0I, 2, 2)
        @test_throws ArgumentError cmg_preconditioner_lap(Asym; eliminate = true)
        local Apos = sparse([1, 2, 1, 2], [1, 2, 2, 1], [2.0, 2.0, 0.5, 0.5], 2, 2)
        @test_throws ArgumentError cmg_preconditioner_lap(Apos; eliminate = true)
    end

    @testset "eliminated preconditioner closure reuse" begin
        # the closure shares workspace across calls (documented non-reentrant);
        # repeated applies must keep working and the V-cycle variant must be
        # PCG-safe end to end
        Random.seed!(17)
        local A = lap(tree_plus_offtree_adj(3000, 20))
        local n = size(A, 1)
        local (pfunc, EH) = cmg_preconditioner_lap(A; cycle = :vcycle, eliminate = true)
        local b1 = randn(n); b1 .-= sum(b1) / n
        local b2 = randn(n); b2 .-= sum(b2) / n
        local x1 = copy(pfunc(b1))          # copy: the closure reuses its buffer
        local x2 = copy(pfunc(b2))
        local x1_again = copy(pfunc(b1))
        @test x1 == x1_again                # deterministic under buffer reuse
        @test any(x1 .!= x2)
        local f = pcgSolver(A, pfunc)
        local xs = f(b1, maxits = 300, tol = 1e-8)
        @test relres(A, xs, b1) < 1e-6
    end

end

# optional timing script on the large example matrix (requires MAT.jl and
# example/X.mat; see the header of xmat_timing.jl)
if get(ENV, "CMG_TEST_XMAT", "0") == "1"
    include("xmat_timing.jl")
end
