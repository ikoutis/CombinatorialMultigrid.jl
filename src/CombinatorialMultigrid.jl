module CombinatorialMultigrid
using SparseArrays
using LinearAlgebra
using LDLFactorizations
using Laplacians

include("contract.jl")
include("cmgAlg.jl")
include("elimination.jl")
include("kcycle.jl")
export cmg_preconditioner_adj, cmg_preconditioner_lap, cmg_solve, CMGStats
export EliminatedHierarchy
end
