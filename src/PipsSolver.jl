module PipsSolver

using Plasmo
using JuMP
using Libdl
using MPI
using Distributed

export pipsnlp_solve, distribute

include("PipsNlpInterface.jl")

using .PipsNlpInterface

include("distribute.jl")

end # module
