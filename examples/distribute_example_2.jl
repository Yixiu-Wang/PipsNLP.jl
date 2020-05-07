using Distributed

if nprocs() == 1
    addprocs(2)
end

#Setup the worker environments
@everywhere using Pkg
@everywhere using Revise
@everywhere Pkg.activate(".")
@everywhere using ModelGraphs


include("simple_modelgraph2.jl")

#Distribute the modelgraph to the Julia workers
remote_graphs = distribute(graph,workers(),remote_name = :graph)  #create the variable pipsgraph on each worker

r1 = remote_graphs[1] #reference to the model graph on worker 1
r2 = remote_graphs[2] #reference to the model graph on worker 2

#Fetch graph from each worker
g1 = fetch(r1)
g2 = fetch(r2)


println("Graph: ",Base.summarysize(graph)," Remote Graph: ",Base.summarysize(g1))
names = fieldnames(ModelGraph)
for name in names
    println(name," ",Base.summarysize(getfield(graph,name)), " ",Base.summarysize(getfield(g1,name)))
end
