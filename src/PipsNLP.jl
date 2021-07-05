module PipsNLP

using JuMP, Plasmo
using Libdl
using MPI
using Distributed
using DataStructures

export pipsnlp_solve, distribute_optigraph, retrieve_solution

include("pips_nlp_data.jl")
include("distribute.jl")
include("pipsnlp_interface.jl")

using .PipsNlpInterface


end # module
