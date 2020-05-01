# to import MPIManager
using MPIClusterManagers

# need to also import Distributed to use addprocs()
using Distributed

include("simple_modelgraph1.jl")

# specify, number of mpi workers, launch cmd, etc.
if !(isdefined(Main,:manager))
    manager=MPIManager(np=2)
    # start mpi workers and add them as julia workers too.
    addprocs(manager)
end

@everywhere using Pkg
@everywhere using Revise
@everywhere Pkg.activate(".")
@everywhere using ModelGraphs
@everywhere using MGPipsSolver

#Distribute the graph to workers
julia_workers = collect(values(manager.mpi2j))
remote_references = distribute(graph,julia_workers,remote_name = :pipsgraph)  #create the variable graph on each worker

@mpi_do manager begin
    using MPI
    pipsnlp_solve(getfield(Main,:pipsgraph))
end

rank_zero = manager.mpi2j[0] #julia process representing rank 0
solution = fetch(@spawnat(rank_zero, getfield(Main, :pipsgraph)))
