module CombinatorialMultigrid
using SparseArrays
using LinearAlgebra
using LDLFactorizations

include("contract.jl")
include("cmgAlg.jl")
include("elimination.jl")
include("kcycle.jl")
include("disconnected.jl")
export cmg_preconditioner_adj, cmg_preconditioner_lap, cmg_solve, CMGStats
export EliminatedHierarchy, DisconnectedHierarchy
end
