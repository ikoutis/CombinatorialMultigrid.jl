## K-cycle: the stationary repeats of the legacy cycle are replaced by inner
## flexible-CG (FCG) iterations at each coarse level, preconditioned by the
## recursive K-cycle below. The resulting operator is *nonlinear*, so it may
## only be driven by the flexible outer loop in `cmg_solve` — never by a
## standard PCG. Port of pycmg's _cycles.py/_solve.py (in turn a port of
## CMG-cpp-dev/src/preconditioner.cpp).

## structures

Base.@kwdef struct KWorkspace
    tmp::Vector{Float64}      # residual scratch for kcycle! at this level
    b::Vector{Float64}        # restricted rhs delivered from the level above
    z::Vector{Float64}        # inner-FCG solution when this level is the coarse target
    r::Vector{Float64}        # inner-FCG residual
    r_prev::Vector{Float64}   # previous residual (flexible PR beta)
    p::Vector{Float64}        # search direction
    q::Vector{Float64}        # A*p
    d::Vector{Float64}        # K-cycle output = FCG preconditioned direction
end

function init_KWorkspace(H::Vector{HierarchyLevel})
    # all buffers at level j have the full (untrimmed) size H[j].n
    return [
        KWorkspace(
            tmp = zeros(h.n),
            b = zeros(h.n),
            z = zeros(h.n),
            r = zeros(h.n),
            r_prev = zeros(h.n),
            p = zeros(h.n),
            q = zeros(h.n),
            d = zeros(h.n),
        ) for h in H
    ]
end

Base.@kwdef struct CMGStats
    iterations::Int64
    relres::Float64
    converged::Bool
    level_visits::Vector{Int64}   # empty unless collect_stats=true with cycle=:kcycle
end

@inline function prolongate_add!(x::Vector{Float64}, cI::Vector{Int64}, z::Vector{Float64})
    @inbounds @simd for i = 1:length(x)
        x[i] += z[cI[i]]
    end
end

"""
    nnz_lower(A)

Number of stored entries on or below the diagonal (the nnz convention of the
K-cycle work-budget rule).
"""
function nnz_lower(A::SparseMatrixCSC)
    local colptr = A.colptr
    local rowval = A.rowval
    local cnt = 0
    @inbounds for j = 1:size(A, 2)
        for p = colptr[j]:(colptr[j+1]-1)
            cnt += rowval[p] >= j
        end
    end
    return cnt
end

"""
    krepeat = compute_kcycle_repeats(H, theta)

Per-level caps on the inner FCG iterations; `krepeat[k]` bounds the FCG
launched *from* level `k` (solving level `k+1`'s system).

With `theta > 0` the budget rule enforces `N_k * m_k <= theta^(k-1) * m_1`
(total work <= m_1/(1-theta)) while carrying slack forward:
`r_k = max(floor(theta^k * m_1 / (N_k * m_{k+1})), 1)`, capped at the coarse
dimension (CG is exact in `nc` steps). `m_k` is the lower-triangle nnz; the
terminal level uses the nnz of the LDL factor when the base case is a direct
solve. With `theta == 0` the local repeat rule
`max(floor(m_k/m_{k+1} - 1), 1)` is used instead (opt-out).
"""
function compute_kcycle_repeats(H::Vector{HierarchyLevel}, theta::Float64)::Vector{Int64}
    local L = length(H)
    local krepeat = ones(Int64, L)
    if L < 2
        return krepeat
    end

    # per-level masses in the lower-triangle convention
    local m = [Float64(nnz_lower(h.A)) for h in H]
    if !H[L].iterative
        # LDL factor of the trimmed (n-1)x(n-1) block: strictly-lower entries
        # plus one diagonal per row
        m[L] = Float64(nnz(H[L].chol.L) + (H[L].n - 1))
    end

    if theta == 0.0
        for k = 1:L-1
            krepeat[k] = max(floor(Int64, m[k] / max(m[k+1], 1.0) - 1), 1)
        end
        return krepeat
    end

    local cap = m[1]
    local N = 1.0
    for k = 1:L-1
        cap *= theta
        local m_next = max(m[k+1], 1.0)
        local r = floor(cap / (N * m_next))
        r = max(r, 1.0)
        if H[k].nc >= 1
            r = min(r, Float64(H[k].nc))
        end
        krepeat[k] = Int64(r)
        N *= r
    end

    return krepeat
end

## cycle recursion

"""
    kcycle!(x, H, W, lvl, b, krepeat, inner_tol, visits)

One K-cycle sweep at level `lvl`: Jacobi pre-smooth from a zero guess,
inner-FCG coarse solve, prolongation and Jacobi post-smooth. Writes the
result into `x` (length `H[lvl].n`).
"""
function kcycle!(
    x::Vector{Float64},
    H::Vector{HierarchyLevel},
    W::Vector{KWorkspace},
    lvl::Int64,
    b::Vector{Float64},
    krepeat::Vector{Int64},
    inner_tol::Float64,
    visits::Union{Nothing,Vector{Int64}},
)
    local h = H[lvl]

    if visits !== nothing
        visits[lvl] += 1
    end

    if h.islast
        if !h.iterative
            # direct solve of the trimmed system; the last coordinate is
            # pinned to 0 (Laplacian null space)
            local nt = h.n - 1
            fill!(x, 0.0)
            @views ldiv!(x[1:nt], h.chol, b[1:nt])
        else
            x .= b .* h.invD
        end
        return x
    end

    local A = h.A
    local invD = h.invD
    local cI = h.cI
    local tmp = W[lvl].tmp

    # pre-smooth from zero initial guess
    x .= b .* invD

    # restricted residual and inner solve z ~ A_c^{-1} bc
    mul!(tmp, A, x)
    tmp .= b .- tmp
    interpolate!(W[lvl+1].b, cI, tmp)
    inner_fcg!(W[lvl+1].z, H, W, lvl, W[lvl+1].b, krepeat, inner_tol, visits)

    # interpolate and post-smooth
    prolongate_add!(x, cI, W[lvl+1].z)
    mul!(tmp, A, x)
    tmp .= b .- tmp
    tmp .*= invD
    x .+= tmp

    return x
end

"""
    inner_fcg!(z, H, W, lvl, bc, krepeat, inner_tol, visits)

Flexible CG (FCG(1), Polak-Ribiere beta) on the coarse system of level
`lvl+1`, preconditioned by the recursive K-cycle — nonlinear, so the beta
uses `(r - r_prev)`; plain Fletcher-Reeves would be invalid here. Stops at
the budget cap `krepeat[lvl]`, on sufficient residual reduction (adaptive
`inner_tol`), or on the `pq <= 0` breakdown guard (null space / roundoff
drift of the singular coarse Laplacian). Writes the solution into `z`.
"""
function inner_fcg!(
    z::Vector{Float64},
    H::Vector{HierarchyLevel},
    W::Vector{KWorkspace},
    lvl::Int64,
    bc::Vector{Float64},
    krepeat::Vector{Int64},
    inner_tol::Float64,
    visits::Union{Nothing,Vector{Int64}},
)
    local c = lvl + 1
    local hc = H[c]
    # the terminal direct level stores only the trimmed block in A; the
    # coarse-system mat-vec needs the full matrix
    local Ac = (hc.islast && !hc.iterative) ? hc.A_full : hc.A
    local Wc = W[c]
    local nu = krepeat[lvl]

    fill!(z, 0.0)
    copyto!(Wc.r, bc)

    local bnorm2 = dot(Wc.r, Wc.r)
    if bnorm2 == 0.0
        return z
    end

    # adaptive stopping threshold, floored by a roundoff guard that matters
    # when the level below is the exact LDL base case
    local stop2 = max(1e-28, inner_tol * inner_tol) * bnorm2

    local dr_prev = 0.0
    for i = 1:nu
        if dot(Wc.r, Wc.r) <= stop2
            break
        end

        kcycle!(Wc.d, H, W, c, Wc.r, krepeat, inner_tol, visits)

        if i == 1
            copyto!(Wc.p, Wc.d)
        else
            local num = dot(Wc.d, Wc.r) - dot(Wc.d, Wc.r_prev)
            local beta = dr_prev != 0.0 ? num / dr_prev : 0.0
            Wc.p .= Wc.d .+ beta .* Wc.p
        end

        mul!(Wc.q, Ac, Wc.p)
        local pq = dot(Wc.p, Wc.q)
        if pq <= 0.0
            break
        end

        local rd = dot(Wc.r, Wc.d)
        local alpha = rd / pq

        dr_prev = rd
        copyto!(Wc.r_prev, Wc.r)

        axpy!(alpha, Wc.p, z)
        axpy!(-alpha, Wc.q, Wc.r)
    end

    return z
end

## outer solver

"""
    (x, stats) = cmg_solve(H, b; kwargs...)
    (x, stats) = cmg_solve(A, b; kwargs...)

Solve the (possibly SDD) system with a flexible-CG outer loop preconditioned
by the CMG hierarchy. `H` is the hierarchy returned as the second value of
`cmg_preconditioner_lap`; passing a sparse matrix `A` builds the hierarchy
internally. Returns the solution and a `CMGStats` with `iterations`,
`relres`, `converged` and (with `collect_stats = true`) per-level
`level_visits`.

**The default configuration is the K-cycle with degree-1/2 elimination** (see
`eliminate` below) — the recommended, generally-fastest way to solve a system.

Keyword arguments:
- `eliminate = true` (matrix `A` input only): first exactly factor out degree-1
  and degree-2 nodes (a partial Cholesky / Schur complement), then solve the
  reduced core — a large win on near-tree inputs. **Adaptive:** when a cheap
  scan finds (almost) no degree-1/2 candidates, the elimination step is skipped
  automatically, so on graphs with no low-degree structure the default costs
  only one allocation-free pass. `eliminate = false` skips even the scan. The
  hierarchy methods do not accept this keyword (the elimination decision is
  baked in at build time).
- `tol = 1e-8`: relative residual tolerance.
- `maxit = 500`: maximum outer iterations.
- `cycle = :kcycle`: preconditioning cycle; `:kcycle` (inner flexible-CG
  acceleration at the coarse levels) or `:legacy` (the classic stationary CMG
  cycle, with which the outer loop reduces to plain PCG). `:vcycle` is accepted
  as a deprecated alias of `:legacy` (the legacy cycle is a stationary
  iteration, not a true geometric V-cycle).
- `theta = 0.75`: K-cycle work-budget cap (see `compute_kcycle_repeats`);
  `0.0` opts out into the fixed local repeat rule.
- `inner_tol = 0.25`: adaptive stopping for the inner FCG iterations;
  `0.0` disables early stopping.
- `collect_stats = false`: record per-level K-cycle visit counts.
"""
function cmg_solve(
    H::Vector{HierarchyLevel},
    b::AbstractVector{<:Real};
    tol::Float64 = 1e-8,
    maxit::Int64 = 500,
    cycle::Symbol = :kcycle,
    theta::Float64 = 0.75,
    inner_tol::Float64 = 0.25,
    collect_stats::Bool = false,
)
    local c = _canonical_cycle(cycle)

    local sd = H[1].sd && H[1].n > 1
    local n_orig = sd ? H[1].n - 1 : H[1].n
    if length(b) != n_orig
        throw(DimensionMismatch("length(b) = $(length(b)), expected $n_orig"))
    end

    # SDD wrapper: run the whole solve on the augmented system, extract after
    local b_sys::Vector{Float64} = sd ? [Float64.(b); -sum(b)] : Vector{Float64}(b)

    local x_sys, stats = fcg_solve!(H, b_sys, tol, maxit, c, theta, inner_tol, collect_stats)

    local x = sd ? x_sys[1:n_orig] .- x_sys[n_orig+1] : x_sys
    return (x, stats)
end

function cmg_solve(
    A::SparseMatrixCSC,
    b::AbstractVector{<:Real};
    eliminate::Bool = true,
    split_components::Bool = true,
    kwargs...,
)
    if split_components
        # Disconnected input: solve each connected component independently (each
        # is connected, so all existing build/grounding logic applies). On a
        # connected graph this returns nothing and we fall through unchanged.
        # Pass `split_components = false` to skip the check.
        local DH = build_disconnected_hierarchy(A; eliminate = eliminate)
        DH !== nothing && return cmg_solve(DH, b; kwargs...)
    end
    if eliminate
        # Default path: exactly factor out degree-1/2 nodes, then solve the
        # reduced core. On near-tree inputs this is a large win; on inputs with
        # no low-degree structure it costs one extra O(n+m) pass and otherwise
        # matches the plain solve. Pass `eliminate = false` to skip it.
        return cmg_solve(build_eliminated_hierarchy(A), b; kwargs...)
    end
    local A_ = validateInput!(A)  # throws if not valid
    cmg_solve(build_hierarchy(A, A_), b; kwargs...)
end

"""
    (x, stats) = cmg_solve(EH::EliminatedHierarchy, b; tol, maxit, cycle, theta, inner_tol, collect_stats)

Solve `A x = b` where `EH` was produced by
`cmg_preconditioner_lap(A; eliminate = true)`. Applies the exact partial-Cholesky
forward substitution, solves the reduced core system with CMG (or directly when
the core has <= 1 node — the near-tree fast path), scatters, then back-substitutes
the eliminated variables. `stats` come from the reduced solve (all-zero when the
core is solved directly).
"""
function cmg_solve(
    EH::EliminatedHierarchy,
    b::AbstractVector{<:Real};
    tol::Float64 = 1e-8,
    maxit::Int64 = 500,
    cycle::Symbol = :kcycle,
    theta::Float64 = 0.75,
    inner_tol::Float64 = 0.25,
    collect_stats::Bool = false,
)
    local c = _canonical_cycle(cycle)
    if length(b) != EH.n
        throw(DimensionMismatch("length(b) = $(length(b)), expected $(EH.n)"))
    end

    local y = forward_elim(b, EH.elims)
    local x = zeros(Float64, EH.n)
    local m = length(EH.ind)
    local stats::CMGStats

    if m == 0
        # everything was eliminated; the solution is fully determined below
        stats = CMGStats(0, 0.0, true, Int64[])
    elseif m == 1
        # tiny reduced system solved directly (near-tree fast path)
        if EH.is_lap
            x[EH.ind[1]] = 0.0                       # Laplacian null-space reference
        else
            x[EH.ind[1]] = y[EH.ind[1]] / EH.A_red[1, 1]
        end
        stats = CMGStats(0, 0.0, true, Int64[])
    else
        local b_red = Vector{Float64}(y[EH.ind])
        local x_red, st = cmg_solve(
            EH.H,
            b_red;
            tol = tol,
            maxit = maxit,
            cycle = c,
            theta = theta,
            inner_tol = inner_tol,
            collect_stats = collect_stats,
        )
        @inbounds for r = 1:m
            x[EH.ind[r]] = x_red[r]
        end
        stats = st
    end

    back_elim!(x, y, EH.elims)
    return (x, stats)
end

function fcg_solve!(
    H::Vector{HierarchyLevel},
    b_sys::Vector{Float64},
    tol::Float64,
    maxit::Int64,
    cycle::Symbol,
    theta::Float64,
    inner_tol::Float64,
    collect_stats::Bool,
)
    # on a single-level direct hierarchy H[1].A is the trimmed block; the
    # outer system is the full matrix
    local A = (H[1].islast && !H[1].iterative) ? H[1].A_full : H[1].A
    local n = size(A, 1)
    local use_kcycle = cycle === :kcycle

    local krepeat = Int64[]
    local W = KWorkspace[]
    local visits::Union{Nothing,Vector{Int64}} = nothing
    local Hs = Hierarchy[]
    local Wv = Workspace[]
    local Xv = LevelAux[]
    if use_kcycle
        krepeat = compute_kcycle_repeats(H, theta)
        W = init_KWorkspace(H)
        visits = collect_stats ? zeros(Int64, length(H)) : nothing
    else
        Hs = init_Hierarchy(H)
        Wv = init_Workspace(H)
        Xv = init_LevelAux(H)
    end

    local x = zeros(n)
    local r = copy(b_sys)
    local r_prev = zeros(n)
    local p = zeros(n)
    local q = zeros(n)
    local d = zeros(n)

    local bnorm = norm(r)
    if bnorm == 0.0
        return (x, CMGStats(0, 0.0, true, visits === nothing ? Int64[] : visits))
    end

    local iterations = 0
    local converged = false
    local dr_prev = 0.0
    local rnorm = bnorm

    for it = 1:maxit
        rnorm = norm(r)
        if rnorm <= tol * bnorm
            converged = true
            break
        end

        if use_kcycle
            kcycle!(d, H, W, 1, r, krepeat, inner_tol, visits)
        else
            # preconditioner_i returns its aliased top-level workspace buffer,
            # which the next apply overwrites — the copy is mandatory
            copyto!(d, preconditioner_i(Hs, Wv, Xv, r))
        end

        if it == 1
            copyto!(p, d)
        else
            local num = dot(d, r) - dot(d, r_prev)
            local beta = dr_prev != 0.0 ? num / dr_prev : 0.0
            p .= d .+ beta .* p
        end

        mul!(q, A, p)
        local pq = dot(p, q)
        if pq <= 0.0
            break  # breakdown guard
        end

        local rd = dot(r, d)
        local alpha = rd / pq

        dr_prev = rd
        copyto!(r_prev, r)

        axpy!(alpha, p, x)
        axpy!(-alpha, q, r)
        iterations += 1

        rnorm = norm(r)
    end

    if !converged && rnorm <= tol * bnorm
        converged = true  # converged on the final iteration
    end

    local relres = rnorm / bnorm
    return (x, CMGStats(iterations, relres, converged, visits === nothing ? Int64[] : visits))
end

"""
    pfunc = make_kcycle_preconditioner(H, theta, inner_tol)

Single-apply K-cycle closure over the hierarchy `H` (the `cycle = :kcycle`
backend of `cmg_preconditioner_lap`). The returned operator is *nonlinear*
and must only be driven by a flexible outer loop such as `cmg_solve` — never
passed as `M` to a standard PCG. The closure shares workspace across calls
and is not reentrant or thread-safe.
"""
function make_kcycle_preconditioner(
    H::Vector{HierarchyLevel},
    theta::Float64,
    inner_tol::Float64,
)::Function
    local krepeat = compute_kcycle_repeats(H, theta)
    local W = init_KWorkspace(H)
    local n = H[1].n
    local x = zeros(n)

    if H[1].sd && n > 1
        local bt = zeros(n)
        return b -> begin
            copyto!(view(bt, 1:n-1), b)
            bt[n] = -sum(b)
            kcycle!(x, H, W, 1, bt, krepeat, inner_tol, nothing)
            x[1:n-1] .- x[n]  # mirrors preconditioner_sd
        end
    else
        return b -> begin
            kcycle!(x, H, W, 1, b, krepeat, inner_tol, nothing)
            copy(x)
        end
    end
end
