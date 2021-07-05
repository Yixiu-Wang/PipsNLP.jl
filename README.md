# PipsNLP.jl

## Overview
PipsNLP.jl is a Julia interface to the [PIPS-NLP](https://github.com/Argonne-National-Laboratory/PIPS/tree/master/PIPS-NLP) nonlinear optimization solver.
Running the solver requires a working PIPS-NLP installation following the [instructions](https://github.com/Argonne-National-Laboratory/PIPS).  
The PipsNLP.jl package works with [Plasmo.jl](https://github.com/zavalab/Plasmo.jl) to model and solve optimization problems in parallel.

## Important Notes
At some point, PIPS-NLP updated and broke support for linking constraints.  If you wish to model with linking constraints, try checking out the following commit hash:
`62f664237447c7ce05a62552952c86003d90e68f`

## Julia Interface Installation
PipsNLP.jl can be installed using the following Julia Pkg command.

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/zavalab/PipsNLP.jl.git"))
```
or simply:
```julia
pkg> add https://github.com/zavalab/PipsNLP.jl.git
```


## Useage
Currently, PipsNLP.jl supports three main modes for modeling.  These are:
- 1) Model in parallel and execute with mpirun
- 2) Model in parallel using Julia's mpimanager where mpiranks correspond to Julia workers
- 3) Model in serial and then distribute to julia workers and solve with MPI.  

The examples folder has examples for each of these modeling approaches.  While the first approach is probably the most familiar, the second two approaches make it possible to interact with
the PipsNLP model and perform multiple solves in a Julia session.  The following snippet shows how one might model using the distribute functionality.

```julia
using MPIClusterManagers # to import MPIManager
using Distributed        # need to also import Distributed to use addprocs()

#setup worker environments
@everywhere using Pkg
@everywhere Pkg.activate((@__DIR__)) #change this to hit the correct environment
@everywhere using Plasmo
@everywhere using PipsNLP

#Setup MPI manager
manager=MPIManager(np=2) # specify, number of mpi workers, launch cmd, etc.
addprocs(manager)        # start mpi workers and add them as julia workers too.

#create an optigraph with Plasmo.jl
graph = OptiGraph()

#Add optinodes
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

#link constraint between nodes
@linkconstraint(graph,n1[:x] == n2[:x])

#map julia workers to MPI ranks
julia_workers = sort(collect(values(manager.mpi2j)))

#distribute the graph to workers. This creates the variable `pipsgraph` on each worker with an allocation of optinodes.
remote_references = PipsNLP.distribute_optigraph(graph,julia_workers,remote_name = :pipsgraph)

#solve with PIPS-NLP and MPI
@mpi_do manager begin
    using MPI
    PipsNLP.pipsnlp_solve(pipsgraph)
end

#fill the local solution with the result on worker 2 (worker 2 is the root MPI rank)
PipsNLP.fill_solution!(graph, :pipsgraph, 2)
```
