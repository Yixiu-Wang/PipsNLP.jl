# to import MPIManager
#using MPIClusterManagers

# need to also import Distributed to use addprocs()
using Distributed

if nprocs() == 1
    addprocs(2)
end
@everywhere using Pkg
@everywhere using Revise
@everywhere Pkg.activate(".")
@everywhere using ModelGraphs

include("simple_modelgraph1.jl")

#Get references to the graph on each worker
remote_references = distribute(graph, workers(), remote_name = :graph)
r1 = remote_references[1]
r2 = remote_references[2]

#Fetch graph from each worker
g1 = fetch(r1)
g2 = fetch(r2)


println("Graph: ",Base.summarysize(graph)," Remote Graph: ",Base.summarysize(g1))
names = fieldnames(ModelGraph)
for name in names
    println(name," ",Base.summarysize(getfield(graph,name)), " ",Base.summarysize(getfield(g1,name)))
end
