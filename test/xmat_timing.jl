# Optional timing/smoke script on the large example matrix. Not part of the
# assertion test suite: it needs MAT.jl (not a test dependency) and the 73MB
# example/X.mat. Run it with an environment that has MAT available, e.g.
#
#   julia --project=. -e 'using Pkg; Pkg.activate(temp=true);
#       Pkg.develop(path="."); Pkg.add(["MAT", "Laplacians"]);
#       include("test/xmat_timing.jl")'
#
# or set ENV["CMG_TEST_XMAT"] = "1" before running the test suite.

using Laplacians
using MAT
using LinearAlgebra
using CombinatorialMultigrid

## load example matrix

xmat_path = joinpath(@__DIR__, "..", "example", "X.mat")
file = matopen(xmat_path);
X = read(file, "X");
close(file);
LX = lap(X);
b1 = rand(Float64, size(X, 1));
b1 = b1 .- sum(b1) / length(b1);

@info "###  Running CMG ###"
t = @elapsed (pfunc, h) = cmg_preconditioner_lap(LX);
@info "Time Required to build CMG Solver: $(t) seconds"
t = @elapsed x = pfunc(b1);
@info "Time Required to find x: $(t) seconds"

@info "###  Running PCG with CMG    ### "
## solve with pcg and cmg preconditioner
f1 = pcgSolver(LX, pfunc);
t = @elapsed x = f1(b1, maxits = 40, tol = 1e-6, verbose = true);
@info "Time Required to solve system: $(t) seconds"

@info "###  Running K-cycle (cmg_solve)    ###"
t = @elapsed (xk, stats) = cmg_solve(h, b1; tol = 1e-6, maxit = 40);
@info "Time Required to solve system with K-cycle: $(t) seconds " *
      "($(stats.iterations) iterations, relres = $(stats.relres))"

## solve with approxchol_lap from laplacians
@info "###  Running approxchol_lap    ###"
t = @elapsed solver = approxchol_lap(
    X;
    tol = 1e-6,
    maxits = 1000,
    maxtime = Inf,
    verbose = true,
    pcgIts = Int[],
    params = ApproxCholParams(),
);
@info "Time Required to build Lap Solver: $(t) seconds"
t = @elapsed x = solver(b1);
@info "Time Required to find x: $(t) seconds"
