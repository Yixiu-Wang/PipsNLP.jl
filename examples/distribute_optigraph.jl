using MPIClusterManagers # to import MPIManager
using Distributed #provides addprocs()

manager=MPIManager(np=2)
addprocs(manager)

#Setup worker environments
@everywhere using Pkg
@everywhere Pkg.activate((@__DIR__)*"/..")
@everywhere using Plasmo
@everywhere using PipsNLP


graph = OptiGraph()

n1 = @optinode(graph)
n2 = @optinode(graph)

@variable(n1, 0 <= x <= 2)
@variable(n1, 0 <= y <= 3)
@variable(n1, z >= 0)
@constraint(n1, x+y+z >= 4)
@objective(n1, Min, y)

@variable(n2,x)
@NLconstraint(n2,ref,exp(x) >= 2)
@variable(n2,z >= 0)
@constraint(n2,z + x >= 4)
@objective(n2,Min,x)

@linkconstraint(graph,n1[:x] == n2[:x])

#distribute the graph to workers
julia_workers = collect(values(manager.mpi2j))
remote_references = PipsNLP.distribute_optigraph(graph,julia_workers,remote_name = :pipsgraph)

# The remote optigraphs can be queried if they are fetched from the other workers
# NOTE: this will copy the optigraph to the main process
# r1 = fetch(remote_references[1])
# r2 = fetch(remote_references[2])

#Solve with pips-nlp
@mpi_do manager begin
    using MPI
    PipsNLP.pipsnlp_solve(pipsgraph)
end

#fill the local solution with the result on worker 2 (worker 2 is the root MPI rank)
PipsNLP.fill_solution!(graph,:pipsgraph,2)
