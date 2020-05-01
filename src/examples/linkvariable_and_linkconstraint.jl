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
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using ModelGraphs
@everywhere using ModelGraphMPISolvers

@everywhere function get_electricity_model(demand)
    m = Model()
    @variable(m, 0<=prod<=10, start=5)     #amount of electricity produced
    @variable(m, input)             #amount of electricity purchased or sold
    @variable(m, gas_purchased)     #amount of gas purchased
    @constraint(m, gas_purchased >= prod)
    @constraint(m, prod + input == demand)
    return m
end

Ns = 10
demand = rand(Ns)*10
#Create modelgraph
graph = ModelGraph()
@node(graph,master)
@variable(master, 0<=gas_purchased<=8)                       #creates a linkvariable on a graph
@objective(graph,Min,gas_purchased)

subproblems = ModelGraph()
add_subgraph!(master,subproblems)

for j in 1:Ns
    scenm = get_electricity_model(demand[j])                    #get scenario model and append to parent node
    n = @node(graph)
    set_model(n,scenm)
    @linkconstraint(graph, scenm[:gas_purchased])                 #connect children and parent variables
    @objective(scenm,Min,1/Ns*scenm[:prod] + 3*scenm[:input])   #reconstruct second stage objective
end
@linkconstraint(graph,(1/Ns)*sum(getnode(graph,s)[:prod] for s in 1:Ns) == 8)

julia_workers = collect(values(manager.mpi2j))
remote_graphs = distribute(graph,julia_workers,remote_name = :pipsgraph)  #create the variable pipsgraph on each worker

#Solve with PIPS-NLP
@mpi_do manager begin
    using MPI
    pipsnlp_solve(getfield(Main,:pipsgraph))  #this works because the vairable :pipsgraph was defined on each worker (mpirank)
end
