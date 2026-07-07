#!/usr/bin/env julia

# Standalone correctness and timing experiment for combinatorial Laplacian
# contraction. This script intentionally does not import or modify the solver.

using LinearAlgebra
using Printf
using Random
using SparseArrays

include("combinatorial_contraction.jl")

function random_laplacian(n::Int, m::Int; seed::Int = 1)
    rng = MersenneTwister(seed)
    I = Int[]
    J = Int[]
    V = Float64[]

    # Ensure connected-ish coverage with a path, then add random edges.
    for i = 1:n-1
        w = 0.1 + rand(rng)
        push!(I, i); push!(J, i + 1); push!(V, w)
        push!(I, i + 1); push!(J, i); push!(V, w)
    end
    for _ = 1:m
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        i == j && continue
        w = 0.1 + rand(rng)
        push!(I, i); push!(J, j); push!(V, w)
        push!(I, j); push!(J, i); push!(V, w)
    end

    A = sparse(I, J, V, n, n)
    d = vec(sum(A, dims = 2))
    return spdiagm(0 => d) - A
end

function random_clusters(n::Int, nc::Int; seed::Int = 2)
    rng = MersenneTwister(seed)
    cI = rand(rng, 1:nc, n)
    # Ensure every cluster is represented so the product and combinatorial paths
    # exercise all coarse vertices.
    for c = 1:min(n, nc)
        cI[c] = c
    end
    return cI
end

function check_case(n::Int, m::Int, nc::Int; seed::Int)
    L = random_laplacian(n, m; seed = seed)
    cI = random_clusters(n, nc; seed = seed + 10_000)
    L_comb = contract_laplacian_combinatorial(L, cI, nc)
    L_prod = contract_laplacian_matmul(L, cI, nc)
    err = max_abs_diff(L_comb, L_prod)
    if err > 1e-10
        error("contraction mismatch for n=$n m=$m nc=$nc seed=$seed: max abs diff = $err")
    end
    return err
end

function run_correctness()
    cases = (
        (8, 4, 3),
        (50, 100, 7),
        (200, 600, 20),
        (1000, 3000, 100),
    )
    for (idx, (n, m, nc)) in enumerate(cases)
        err = check_case(n, m, nc; seed = idx)
        @printf("correctness n=%d m=%d nc=%d max_abs_diff=%.3e\n", n, m, nc, err)
    end
end

function median_time_and_alloc(f; reps::Int = 5)
    times = Float64[]
    allocs = Int[]
    for _ = 1:reps
        GC.gc()
        allocated = @allocated begin
            t = @elapsed f()
            push!(times, t)
        end
        push!(allocs, allocated)
    end
    sort!(times)
    sort!(allocs)
    return times[cld(length(times), 2)], allocs[cld(length(allocs), 2)]
end

function run_timing(; n::Int = 20_000, m::Int = 80_000, nc::Int = 2_000, reps::Int = 5)
    L = random_laplacian(n, m; seed = 123)
    cI = random_clusters(n, nc; seed = 456)

    # Warm up compilation.
    contract_laplacian_combinatorial(L, cI, nc)
    contract_laplacian_matmul(L, cI, nc)

    t_comb, a_comb = median_time_and_alloc(() -> contract_laplacian_combinatorial(L, cI, nc; check = false); reps = reps)
    t_prod, a_prod = median_time_and_alloc(() -> contract_laplacian_matmul(L, cI, nc); reps = reps)

    L_comb = contract_laplacian_combinatorial(L, cI, nc; check = false)
    L_prod = contract_laplacian_matmul(L, cI, nc)
    err = max_abs_diff(L_comb, L_prod)

    println()
    @printf("timing n=%d input_random_edges=%d nc=%d reps=%d\n", n, m, nc, reps)
    @printf("combinatorial: median %.6f s, median allocated %.3f MiB\n", t_comb, a_comb / 2.0^20)
    @printf("matmul      : median %.6f s, median allocated %.3f MiB\n", t_prod, a_prod / 2.0^20)
    @printf("max_abs_diff: %.3e\n", err)
end

function main(args)
    run_correctness()
    n = length(args) >= 1 ? parse(Int, args[1]) : 20_000
    m = length(args) >= 2 ? parse(Int, args[2]) : 80_000
    nc = length(args) >= 3 ? parse(Int, args[3]) : 2_000
    reps = length(args) >= 4 ? parse(Int, args[4]) : 5
    run_timing(n = n, m = m, nc = nc, reps = reps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
