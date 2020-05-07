using MPIClusterManagers
using Distributed
using Ipopt
#setup the MPIManager if it hasn't been already
if !(isdefined(Main,:manager))
    # specify, number of mpi workers, launch cmd, etc.
    manager=MPIManager(np=2)
    # start mpi workers and add them as julia workers too.
    addprocs(manager)
end

#Setup the worker environments
@everywhere begin
    using Pkg
    Pkg.activate(".")
    using Plasmo
    using PipsSolver
    using MPI
end

@everywhere function set_electricity_model(node,demand)
    @variable(m, 0<=prod<=10, start=5)     #amount of electricity produced
    @variable(m, input)             #amount of electricity purchased or sold
    @variable(m, gas_purchased)     #amount of gas purchased
    @constraint(m, gas_purchased >= prod)
    @constraint(m, prod + input == demand)
end

Ns = 10
demand = rand(Ns)*10

#Create modelgraph
graph = ModelGraph()
@node(graph,master)

@variable(master, 0 <= gas_purchased <= 8)                       #creates a linkvariable on a graph
@objective(master,Min,gas_purchased)

subgraph = ModelGraph()
add_subgraph!(graph,subgraph)
@node(subgraph,scenarios[1:Ns])
for node in scenarios
    set_electricity_model(node,demand[j])
    @linkconstraint(graph,master[:gas_purchased] == node[:gas_purchased])           #connect children and parent variables
    @objective(scenm,Min,1/Ns*scenm[:prod] + 3*scenm[:input])   #reconstruct second stage objective
end
@linkconstraint(graph,(1/Ns)*sum(s[:prod] for s in scenarios) == 8)

julia_workers = collect(values(manager.mpi2j))
remote_graphs = PipsSolver.distribute(graph,julia_workers,remote_name = :pipsgraph)  #create the variable pipsgraph on each worker

#Solve with PIPS-NLP
@mpi_do manager begin
    pipsnlp_solve(pipsgraph)  #this works because the vairable :pipsgraph was defined on each worker (mpirank)
end
