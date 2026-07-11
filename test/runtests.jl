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

## dense / expander-like builders for sparsify-on-stall (the existing grid and
## near-tree helpers coarsen fine and never stall aggregation). Ported from the
## CMG-python experiments' graphs.py; return ADJACENCY (apply lap() to get L).

# connected Erdos-Renyi blob (spanning tree + random edges to avgdeg), unit
# weights -- dense enough (avgdeg >= 32) that the real aggregation stalls on it
function dense_blob_adj(n::Int; avgdeg::Int = 32, seed::Int = 0)
    local rng = MersenneTwister(seed)
    local es = Set{Tuple{Int64,Int64}}()
    for v = 2:n
        local p = rand(rng, 1:v-1)
        push!(es, (min(v, p), max(v, p)))
    end
    while length(es) < n * avgdeg ÷ 2
        local u = rand(rng, 1:n)
        local v = rand(rng, 1:n)
        u != v && push!(es, (min(u, v), max(u, v)))
    end
    local I = Int64[]; local J = Int64[]; local V = Float64[]
    for (u, v) in es
        push!(I, u); push!(J, v); push!(V, 1.0)
        push!(I, v); push!(J, u); push!(V, 1.0)
    end
    return sparse(I, J, V, n, n)
end

# chain of dense blobs joined by single weak bridges (nblobs-1 small cuts give
# an ill-conditioned system that genuinely needs a multilevel preconditioner)
function blob_chain_adj(nblobs::Int, blobn::Int; avgdeg::Int = 40, seed::Int = 7,
                        wbridge::Float64 = 1e-2)
    local rng = MersenneTwister(seed)
    local I = Int64[]; local J = Int64[]; local V = Float64[]
    for k = 0:nblobs-1
        local off = k * blobn
        local B = dense_blob_adj(blobn; avgdeg = avgdeg, seed = seed + k)
        local rv = rowvals(B); local nz = nonzeros(B)
        for j = 1:blobn, p in nzrange(B, j)
            push!(I, rv[p] + off); push!(J, j + off); push!(V, nz[p])
        end
        if k > 0
            local a = (k - 1) * blobn + rand(rng, 1:blobn)
            local b = off + rand(rng, 1:blobn)
            push!(I, a); push!(J, b); push!(V, wbridge)
            push!(I, b); push!(J, a); push!(V, wbridge)
        end
    end
    return sparse(I, J, V, nblobs * blobn, nblobs * blobn)
end

# two dense blobs joined by ONE weak (high-resistance) bridge
function dense_blob_pair_bridge_adj(n::Int; avgdeg::Int = 32, seed::Int = 2,
                                    wbridge::Float64 = 1e-3)
    local h = n ÷ 2
    local Ab = blockdiag(dense_blob_adj(h; avgdeg = avgdeg, seed = seed),
                         dense_blob_adj(n - h; avgdeg = avgdeg, seed = seed + 100))
    return Ab + sparse([1, h + 1], [h + 1, 1], [wbridge, wbridge], n, n)
end

const CMG = CombinatorialMultigrid

# --- sparsify-on-stall analysis helpers ---
# near-Laplacian SPD operator (Laplacian + tiny slack): makes the generalized
# eigenproblem well posed for the kappa checks
_spd(adj; slack::Float64 = 1e-8) =
    lap(adj) + slack * sparse(1.0I, size(adj, 1), size(adj, 1))

# generalized condition number kappa(M^-1 A) over the shared range
function _gkappa(A, M)
    local ev = eigvals(Symmetric(Matrix(A)), Symmetric(Matrix(M)))
    ev = filter(x -> x > 1e-9, ev)
    return maximum(ev) / minimum(ev)
end

# edge-reduction ratio m_c/m of one aggregate+contract step. (A densification
# DIAGNOSTIC: ~1 means contraction barely thins the edges. NB: this is no longer
# the build's stall trigger -- sparsification now fires at the stock node/nnz
# stagnation point, see build_hierarchy.)
function _edge_ratio(A)
    local n = size(A, 1)
    local cI, nc = CMG.steiner_group(A, Array(diag(A)))
    nc == 1 && return 0.0
    local Ac = CMG.contract_coo(A, cI, nc)
    local m = CMG.nnz_lower(A) - n
    return m == 0 ? 0.0 : (CMG.nnz_lower(Ac) - nc) / m
end

# count injected same-size (identity-transfer) levels in a hierarchy
_n_inject(H) = count(h -> !h.islast && h.nc == h.n, H)

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
            local (x, stats) = cmg_solve(A, b; cycle = :kcycle, eliminate = false)
            @test stats.converged
            @test relres(A, x, b) < 1e-6
        end
    end

    @testset "kcycle vs vcycle iterations" begin
        Random.seed!(2)
        local A = grid3_sdd(16, 16, 16)
        local b = randn(size(A, 1))
        local (xk, sk) = cmg_solve(A, b; cycle = :kcycle, eliminate = false)
        local (xv, sv) = cmg_solve(A, b; cycle = :vcycle, eliminate = false)
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
        local (x, stats) = cmg_solve(A, b; cycle = :kcycle, theta = 0.0, inner_tol = 0.0, eliminate = false)
        @test stats.converged
        @test relres(A, x, b) < 1e-6
    end

    @testset "visit stats" begin
        Random.seed!(4)
        local A = grid2_sdd(60, 60)
        local b = randn(size(A, 1))

        local (x, stats) = cmg_solve(A, b; cycle = :kcycle, collect_stats = true, eliminate = false)
        @test stats.converged
        @test length(stats.level_visits) >= 1
        @test stats.level_visits[1] == stats.iterations

        # budget mode does at least as many coarse visits per outer iteration
        # as the local-repeat opt-out at the second level
        local (_, s_local) =
            cmg_solve(A, b; cycle = :kcycle, theta = 0.0, inner_tol = 0.0, collect_stats = true, eliminate = false)
        if length(stats.level_visits) >= 2 && s_local.iterations > 0 && stats.iterations > 0
            @test stats.level_visits[2] >= stats.iterations  # at least one inner step each
        end

        local (_, sv) = cmg_solve(A, b; cycle = :vcycle, collect_stats = true, eliminate = false)
        @test isempty(sv.level_visits)
    end

    @testset "SDD strong-shift path (augment/extract)" begin
        Random.seed!(5)
        local A = grid2_sdd(25, 25; shift = 0.5)
        local n = size(A, 1)
        local b = randn(n)
        local x_ref = Matrix(A) \ b
        for cycle in (:kcycle, :vcycle)
            local (x, stats) = cmg_solve(A, b; cycle = cycle, tol = 1e-10, eliminate = false)
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
        local (x, stats) = cmg_solve(L, b; cycle = :kcycle, eliminate = false)
        @test stats.converged
        @test relres(L, x, b) < 1e-6
    end

    @testset "preconditioner knob API" begin
        Random.seed!(7)
        local A = grid2_sdd(30, 30)
        local n = size(A, 1)
        local b = randn(n)

        # cmg_preconditioner_lap default is the legacy linear cycle (PCG-safe)
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

    @testset "solve default (k-cycle + elim) and :legacy alias" begin
        Random.seed!(11)
        local A = grid2_sdd(30, 30)
        local n = size(A, 1)
        local b = randn(n)

        # cmg_solve(A, b) with no options: k-cycle + elimination
        local (xd, sd_) = cmg_solve(A, b)
        @test sd_.converged
        @test relres(A, xd, b) < 1e-6

        # dropping elimination still solves the same system
        local (xf, sf) = cmg_solve(A, b; eliminate = false)
        @test sf.converged
        @test relres(A, xf, b) < 1e-6
        @test norm(xd - xf) / norm(xf) < 1e-4          # same solution either way

        # :legacy names the classic cycle; :vcycle is a deprecated alias
        local (xl, sl) = cmg_solve(A, b; cycle = :legacy, eliminate = false)
        @test sl.converged && relres(A, xl, b) < 1e-6
        @test_throws ArgumentError cmg_solve(A, b; cycle = :bogus)
        @test_throws ArgumentError cmg_preconditioner_lap(A; cycle = :bogus)
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
                local (xp, sp) = cmg_solve(L, b; cycle = cycle, tol = 1e-10, eliminate = false)
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
        # (explicit triplets: `I` is shadowed by the local triplet vector in
        # this testset, so `1.0I` would not be the identity here)
        local Asym = sparse([1, 1, 2, 2], [1, 2, 1, 2], [3.0, -1.0, -2.0, 3.0], 2, 2)
        @test_throws ArgumentError cmg_preconditioner_lap(Asym; eliminate = true)
        local Apos = sparse([1, 2, 1, 2], [1, 2, 2, 1], [2.0, 2.0, 0.5, 0.5], 2, 2)
        @test_throws ArgumentError cmg_preconditioner_lap(Apos; eliminate = true)
    end

    @testset "adaptive elimination skip" begin
        Random.seed!(18)

        # 3D grid: every node has degree >= 3, so there is nothing to
        # eliminate; the pre-scan must skip the machinery and alias the input
        local A3 = grid3_sdd(8, 8, 8)
        local n3 = size(A3, 1)
        local (_, EH3) = cmg_preconditioner_lap(A3; cycle = :kcycle, eliminate = true)
        @test length(EH3.elims) == 0
        @test EH3.ind == collect(1:n3)
        @test EH3.A_red === A3                    # aliased, not rebuilt
        @test !isempty(EH3.H)
        local b3 = randn(n3)
        local (xe, se) = cmg_solve(EH3, b3; tol = 1e-10)
        local (xf, sf) = cmg_solve(A3, b3; tol = 1e-10, eliminate = false)
        @test se.converged && sf.converged
        @test norm(xe - xf) / norm(xf) < 1e-6     # same system, same solution

        # a handful of candidates below the 1% threshold (grid corners) skips too
        local A2 = grid2_sdd(30, 30)
        local n2 = size(A2, 1)
        local (_, EH2) = cmg_preconditioner_lap(A2; eliminate = true)
        @test length(EH2.elims) == 0
        @test EH2.A_red === A2
        # ...but the explicit knob can force those corners out, exactly
        local EH2f = CMG.build_eliminated_hierarchy(A2; min_frac = 0.0)
        @test 0 < length(EH2f.elims)
        @test length(EH2f.ind) < n2
        local bg = randn(n2)
        local (xg, sg) = cmg_solve(EH2f, bg; tol = 1e-10)
        local xg_ref = Matrix(A2) \ bg
        @test sg.converged
        @test norm(xg - xg_ref) / norm(xg_ref) < 1e-6

        # near-trees stay far above the threshold and still eliminate
        local L = lap(tree_plus_offtree_adj(400, 15))
        local (_, EHt) = cmg_preconditioner_lap(L; eliminate = true)
        @test length(EHt.ind) < 400
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

    @testset "disconnected components" begin
        # A disconnected Laplacian: two connected blocks + an isolated vertex.
        # The whole matrix is singular (one null-space dim per component); the
        # per-component solve grounds each block, so the residual is ~0 for a
        # range-consistent b. This input makes the plain (split=false) build
        # error/stall, which is exactly what split_components fixes.
        Random.seed!(101)
        local L1 = grid2_lap(20, 20)                      # connected, 400 nodes
        local L2 = lap(tree_plus_offtree_adj(300, 10))    # connected, 300 nodes
        local A = blockdiag(L1, L2, spzeros(1, 1))        # + 1 isolated vertex
        local n = size(A, 1)
        @test maximum(CMG.components(A)) == 3
        local xref = randn(n)
        local b = A * xref                                # in range() by construction
        for elim in (true, false)
            local x, st = cmg_solve(A, b; split_components = true, eliminate = elim,
                cycle = :kcycle, maxit = 200, tol = 1e-8)
            @test relres(A, x, b) < 1e-6
            @test st.converged
        end
        # the factory returns a DisconnectedHierarchy for a disconnected input
        local (_, DH) = cmg_preconditioner_lap(A; cycle = :kcycle, split_components = true)
        @test DH isa DisconnectedHierarchy

        # A disconnected but nonsingular SDD system (two SDD blocks): the
        # per-component solve must match the direct dense solve.
        local S = blockdiag(grid2_sdd(15, 15), grid2_sdd(10, 12))
        local bs = randn(size(S, 1))
        local xs, _ = cmg_solve(S, bs; split_components = true, eliminate = true,
            maxit = 200, tol = 1e-8)
        @test norm(xs - Array(S) \ bs) / norm(Array(S) \ bs) < 1e-5

        # On a connected graph, split_components is a no-op: identical code path,
        # identical result whether the knob is on or off.
        local Ac = grid2_sdd(24, 24)
        local bc = randn(size(Ac, 1))
        local x_on, _ = cmg_solve(Ac, bc; split_components = true, maxit = 200, tol = 1e-8)
        local x_off, _ = cmg_solve(Ac, bc; split_components = false, maxit = 200, tol = 1e-8)
        @test x_on == x_off
    end

    @testset "sparsify-on-stall" begin
        Random.seed!(202)

        # 1. sparsifier quality: a dense blob densifies (contraction barely thins
        # its edges); one adaptive sparsify at keep_frac=0.5 halves the edges with
        # a spectrally-close sparsifier. (Tests the sparsify FUNCTION at a pinned
        # keep_frac, independent of the SparsifyOptions default.)
        local A1 = _spd(dense_blob_adj(400; avgdeg = 32, seed = 1))
        @test _edge_ratio(A1) >= 0.9                       # densifies (diagnostic)
        local e0 = CMG.edges_of(A1)
        local sp_e, _p = CMG.sparsify(400, e0; keep_frac = 0.5, bundles = 1, rng = MersenneTwister(0))
        local A1sp = CMG.sdd_from_edges(400, sp_e, CMG.slack_of(A1))
        @test _edge_ratio(A1sp) < 0.9                      # aggregation resumes
        @test length(e0) / length(sp_e) >= 1.8             # >= 1.8x fewer edges
        @test _gkappa(A1, A1sp) < 50.0                     # spectrally close

        # 2. the spanner is essential: on two blobs + one weak bridge, uniform-
        # only sampling is far worse than spanner+uniform (it drops the bridge).
        local A2 = _spd(dense_blob_pair_bridge_adj(400; avgdeg = 32, seed = 2,
            wbridge = 1e-3); slack = 1e-10)
        local e2 = CMG.edges_of(A2)
        local span_k = sum(_gkappa(A2, CMG.sdd_from_edges(400,
            CMG.sparsify(400, e2; keep_frac = 0.5, bundles = 1, rng = MersenneTwister(i))[1],
            CMG.slack_of(A2)))
            for i = 1:4) / 4
        local unif_k = begin
            local rng = MersenneTwister(7); local acc = 0.0
            for _ = 1:6
                local kept = [(u, v, w / 0.25) for (u, v, w) in e2 if rand(rng) < 0.25]
                acc += _gkappa(A2, CMG.sdd_from_edges(400, kept, CMG.slack_of(A2)))
            end
            acc / 6
        end
        @test span_k < 50.0
        @test unif_k >= 20.0 * span_k                      # uniform-only far worse

        # 3. off == stock: sparsify_on_stall=false is byte-identical to the stock
        # build (same levels, same nnz/nc/islast, no injected level).
        local Loff = lap(blob_chain_adj(6, 150; seed = 7))
        local A_off = CMG.validateInput!(Loff)
        local Hstock = CMG.build_hierarchy(Loff, A_off)
        local Hoff = CMG.build_hierarchy(Loff, A_off; sparsify_on_stall = false)
        @test length(Hstock) == length(Hoff)
        @test all(a.nnz == b.nnz && a.nc == b.nc && a.islast == b.islast
                  for (a, b) in zip(Hstock, Hoff))
        @test _n_inject(Hoff) == 0

        # 4. end-to-end, all three cycles, genuine Laplacian (b _|_ 1). A dense
        # blob-chain densifies under contraction, so the build reaches the stock
        # stagnation point (nnz budget / nc>=n-1), injects >= 1 same-size level
        # there, and every driver converges.
        local L4 = lap(blob_chain_adj(6, 150; seed = 7))
        local Hon = CMG.build_hierarchy(L4, CMG.validateInput!(L4); sparsify_on_stall = true)
        @test _n_inject(Hon) >= 1
        local b4 = randn(size(L4, 1)); b4 .-= sum(b4) / length(b4)
        for cyc in (:legacy, :kcycle, :kscycle)
            local x, st = cmg_solve(L4, b4; sparsify_on_stall = true,
                split_components = false, eliminate = false, cycle = cyc, tol = 1e-9, maxit = 1000)
            @test st.converged
            @test relres(L4, x, b4) < 1e-8
        end

        # 5. full SDD-augmented path: an SDD input (strictly-dominant rows) is
        # augmented to a Laplacian; sparsify injects a Laplacian sparsifier and
        # the ORIGINAL system is solved EXACTLY (not up to the null-space constant).
        local L5 = lap(blob_chain_adj(6, 150; seed = 7))
        local n5 = size(L5, 1)
        local A5 = L5 + spdiagm(0 => [i % 7 == 0 ? 0.5 : 0.0 for i = 1:n5])
        local b5 = randn(n5)                               # SDD: nonsingular
        local xref = Matrix(A5) \ b5
        for cyc in (:legacy, :kcycle, :kscycle)
            local x, st = cmg_solve(A5, b5; sparsify_on_stall = true,
                split_components = false, eliminate = false, cycle = cyc, tol = 1e-11, maxit = 1000)
            @test st.converged
            @test relres(A5, x, b5) < 1e-8
            @test norm(x - xref) / norm(xref) < 1e-6       # exact
        end

        # 6. both spanners are connected, bounded-kappa sparsifiers.
        local A6 = _spd(dense_blob_adj(400; avgdeg = 32, seed = 1))
        local e6 = CMG.edges_of(A6)
        local bs = CMG.spanner_baswana_sen(400, e6; rng = MersenneTwister(1))
        local Gb = sparse([e[1] for e in bs], [e[2] for e in bs], ones(length(bs)), 400, 400)
        @test maximum(CMG.components(Gb + Gb')) == 1       # Baswana-Sen connected
        local gs = CMG.greedy_spanner(400, e6, 9.0)
        local Gg = sparse([e[1] for e in gs], [e[2] for e in gs], ones(length(gs)), 400, 400)
        @test maximum(CMG.components(Gg + Gg')) == 1       # greedy connected
        local sp6, = CMG.sparsify(400, e6; spanner = :baswana_sen, rng = MersenneTwister(2))
        @test length(sp6) < 0.9 * length(e6)
        @test _gkappa(A6, CMG.sdd_from_edges(400, sp6, CMG.slack_of(A6))) < 50.0

        # 7. composes with disconnected + eliminate (each component sparsified).
        local n1 = 3 * 150
        local Ad = blockdiag(lap(blob_chain_adj(3, 150; seed = 1)),
                             lap(blob_chain_adj(3, 150; seed = 9)))
        local nd = size(Ad, 1)
        local bd = randn(nd)
        bd[1:n1] .-= sum(bd[1:n1]) / n1
        bd[n1+1:end] .-= sum(bd[n1+1:end]) / (nd - n1)
        local xd, sd = cmg_solve(Ad, bd; sparsify_on_stall = true, cycle = :legacy,
            tol = 1e-9, maxit = 1000)
        @test sd.converged
        @test relres(Ad, xd, bd) < 1e-7
    end

end

# optional timing script on the large example matrix (requires MAT.jl and
# example/X.mat; see the header of xmat_timing.jl)
if get(ENV, "CMG_TEST_XMAT", "0") == "1"
    include("xmat_timing.jl")
end
