# Quick benchmark: legacy stationary cycle vs the K-cycle option, on 2D SDD
# grids (uniform and anisotropic 100:1). Julia analog of CMG-python's
# benchmarks/bench_grids.py.
#
#   julia --project=. example/bench_kcycle.jl

using SparseArrays
using LinearAlgebra
using Random
using Laplacians
using CombinatorialMultigrid

tridiag_sdd(n; w = 1.0) =
    spdiagm(-1 => fill(-w, n - 1), 0 => fill(2w, n), 1 => fill(-w, n - 1))

# Dirichlet-style SDD grid: interior rows sum to zero, boundary rows are
# strictly dominant (same convention as CMG-python's benchmark grids)
function grid2_sdd(nx, ny; wx = 1.0, wy = 1.0)
    return kron(sparse(1.0I, ny, ny), tridiag_sdd(nx; w = wx)) +
           kron(tridiag_sdd(ny; w = wy), sparse(1.0I, nx, nx))
end

function bench(name, A)
    Random.seed!(0)
    b = randn(size(A, 1))
    tol = 1e-8

    tb = @elapsed (pfunc, H) = cmg_preconditioner_lap(A)

    # (a) legacy closure inside Laplacians PCG
    f = pcgSolver(A, pfunc)
    t_pcg = @elapsed x = f(b, maxits = 500, tol = tol)
    r_pcg = norm(A * x - b) / norm(b)

    # (b) legacy stationary cycle inside the flexible-CG outer loop (= plain PCG)
    t_v = @elapsed (xv, sv) = cmg_solve(H, b; cycle = :legacy, tol = tol)

    # (c) K-cycle (default knobs: theta = 0.75, inner_tol = 0.25)
    t_k = @elapsed (xk, sk) = cmg_solve(H, b; cycle = :kcycle, tol = tol)

    println("== $name  (n = $(size(A, 1)), build $(round(tb, digits = 3))s)")
    println("  pcgSolver + legacy pfunc : $(round(t_pcg, digits = 3))s  relres = $(round(r_pcg, sigdigits = 3))")
    println("  cmg_solve :legacy        : $(sv.iterations) iters  $(round(t_v, digits = 3))s  relres = $(round(sv.relres, sigdigits = 3))")
    println("  cmg_solve :kcycle        : $(sk.iterations) iters  $(round(t_k, digits = 3))s  relres = $(round(sk.relres, sigdigits = 3))")
end

bench("uniform 300x300", grid2_sdd(300, 300))
bench("anisotropic 300x300 (100:1)", grid2_sdd(300, 300; wy = 100.0))
