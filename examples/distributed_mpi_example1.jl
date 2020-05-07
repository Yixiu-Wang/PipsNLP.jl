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
@everywhere Pkg.activate((@__DIR__)*"/..")
@everywhere using Plasmo
@everywhere using PipsSolver
#
# #Distribute the graph to workers
julia_workers = collect(values(manager.mpi2j))


remote_references = PipsSolver.distribute(graph,julia_workers,remote_name = :pipsgraph)  #create the variable graph on each worker

# The remote modelgraphs can be queried if they are fetched fro mthe other workers
r1 = fetch(remote_references[1])
r2 = fetch(remote_references[2])

#
@mpi_do manager begin
    using MPI
    PipsSolver.pipsnlp_solve(pipsgraph)
end


rank_zero = manager.mpi2j[0] #julia process representing rank 0
solution = fetch(@spawnat(rank_zero, pipsgraph))
