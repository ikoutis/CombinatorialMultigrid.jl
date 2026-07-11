module CombinatorialMultigrid
using SparseArrays
using LinearAlgebra
using LDLFactorizations
using Random
using DataStructures

include("contract.jl")
include("spanner.jl")
include("sparsify.jl")
include("cmgAlg.jl")
include("elimination.jl")
include("kcycle.jl")
include("disconnected.jl")
export cmg_preconditioner_adj, cmg_preconditioner_lap, cmg_solve, CMGStats
export EliminatedHierarchy, DisconnectedHierarchy, SparsifyOptions
end
