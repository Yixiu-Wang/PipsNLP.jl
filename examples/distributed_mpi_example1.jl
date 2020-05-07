# to import MPIManager
using MPIClusterManagers

# need to also import Distributed to use addprocs()
using Distributed


#manager=MPIManager(np=2)
manager = MPIManager(;np=2, mpi_cmd=false, launch_timeout=60.0)
# start mpi workers and add them as julia workers too.
addprocs(manager)
# specify, number of mpi workers, launch cmd, etc.
# if !(isdefined(Main,:manager))
#     manager=MPIManager(np=2)
#     # start mpi workers and add them as julia workers too.
#     addprocs(manager)
# end


include("simple_modelgraph1.jl")


@everywhere begin
    using Pkg
    Pkg.activate("..")
    using Plasmo
    using PipsSolver
end

#Distribute the graph to workers
julia_workers = collect(values(manager.mpi2j))
remote_references = distribute(graph,julia_workers,remote_name = :pipsgraph)  #create the variable graph on each worker

@mpi_do manager begin
    using MPI
    pipsnlp_solve(pipsgraph)
end

rank_zero = manager.mpi2j[0] #julia process representing rank 0
solution = fetch(@spawnat(rank_zero, getfield(Main, :pipsgraph)))


# using MPI
# MPI.Init()
# pipsnlp_solve(graph)
