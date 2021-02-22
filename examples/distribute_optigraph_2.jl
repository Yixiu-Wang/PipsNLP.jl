#NOTE: distribute does not work yet on this version


using MPIClusterManagers
using Distributed
using Plasmo

graph = OptiGraph()

#Add nodes to a GraphModel
@optinode(graph,n1)
@optinode(graph,n2)
@optinode(graph,n3)
@optinode(graph,n4)

@variable(n1,0 <= x <= 2)
@variable(n1,0 <= y <= 3)
@variable(n1, z >= 0)
@constraint(n1,x+y+z >= 4)
@objective(n1,Min,y)

@variable(n2,x >= 0)
@NLconstraint(n2,ref,exp(x) >= 2)
@variable(n2,z >= 0)
@constraint(n2,z + x >= 4)
@objective(n2,Min,x)

@variable(n3,x[1:5] >= 0)
@NLconstraint(n3,ref,exp(x[3]) >= 5)
@constraint(n3,sum(x[i] for i = 1:5) == 10)
@objective(n3,Min,x[1] + x[2] + x[3])

@variable(n4,x[1:5])
@constraint(n4,sum(x[i] for i = 1:5) >= 10)
@NLconstraint(n4,ref,exp(x[2]) >= 4)
@objective(n4,Min,x[2])

#Link constraints take the same expressions as the JuMP @constraint macro
@linkconstraint(graph,n1[:x] == n2[:x])
@linkconstraint(graph,n2[:x] == n3[:x][3])
@linkconstraint(graph,n3[:x][1] == n4[:x][1])

manager=MPIManager(np=2)
addprocs(manager)

@everywhere begin
    using Pkg
    using Revise
    Pkg.activate((@__DIR__)*"/..")
    using Plasmo
    using PipsSolver
end

#Distribute the modelgraph to the Julia workers
julia_workers = collect(values(manager.mpi2j))
remote_graphs = distribute(graph,julia_workers,remote_name = :pipsgraph)  #create the variable pipsgraph on each worker

r1 = remote_graphs[1] #reference to the model graph on worker 1
r2 = remote_graphs[2] #reference to the model graph on worker 2

#Solve with PIPS-NLP
@mpi_do manager begin
    using MPI
    pipsnlp_solve(pipsgraph)  #this works because the vairable :pipsgraph was defined on each worker (mpirank)
end

println("Fetching solution")
rank_zero = manager.mpi2j[0] #get the julia process representing rank 0
solution = fetch(@spawnat(rank_zero, getfield(Main, :pipsgraph))) #transfer the solution from rank 0 to our local Julia process

for node in getnodes(solution)
    println(node.model.ext[:colVal])
end
