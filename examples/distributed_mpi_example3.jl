# to import MPIManager
using MPIClusterManagers

# need to also import Distributed to use addprocs()
using Distributed

include("simple_modelgraph3.jl")

#setup the MPIManager if it hasn't been already
if !(isdefined(Main,:manager))
    # specify, number of mpi workers, launch cmd, etc.
    manager=MPIManager(np=2)
    # start mpi workers and add them as julia workers too.
    addprocs(manager)
end

#Setup the worker environments
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using ModelGraphs
@everywhere using ModelGraphMPISolvers

#Distribute the modelgraph to the Julia workers
julia_workers = collect(values(manager.mpi2j))
remote_graphs = distribute(graph,julia_workers,remote_name = :pipsgraph)  #create the variable pipsgraph on each worker

r1 = remote_graphs[1]  #reference to the model graph on worker 1
r2 = remote_graphs[2] #reference to the model graph on worker 2

#Solve with PIPS-NLP
@mpi_do manager begin
    using MPI
    pipsnlp_solve(getfield(Main,:pipsgraph))  #this works because the vairable :pipsgraph was defined on each worker (mpirank)
end


println("Fetching solution")
rank_zero = manager.mpi2j[0] #get the julia process representing rank 0
solution = fetch(@spawnat(rank_zero, getfield(Main, :pipsgraph))) #transfer the solution from rank 0 to our local Julia process

for node in getnodes(solution)
    println(node.model.ext[:colVal])
end

# Check with julia master:
# using MPI
# MPI.Init()
# pipsnlp_solve(graph)
# MPI.Finalize()
