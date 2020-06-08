# PipsSolver.jl

[![Build Status](https://travis-ci.com/jalving/MGPipsSolver.jl.svg?branch=master)](https://travis-ci.com/jalving/PipsSolver.jl)
[![Codecov](https://codecov.io/gh/jalving/MGPipsSolver.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jalving/PipsSolver.jl)

## Overview
PipsSolver.jl is a Julia interface to the [PIPS-NLP](https://github.com/Argonne-National-Laboratory/PIPS/tree/master/PIPS-NLP) nonlinear optimization solver.
Running the solver requires a working PIPS-NLP installation following the [instructions](https://github.com/Argonne-National-Laboratory/PIPS).  
The PipsSolver.jl package works with the graph-based algebraic modeling package [Plasmo.jl](https://github.com/zavalab/Plasmo.jl).

## Installation
PipsSolver.jl can be installed using the following Julia Pkg command.

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/zavalab/PipsSolver.jl.git"))
```

## Useage
```julia
using MPIClusterManagers # to import MPIManager
using Distributed   # need to also import Distributed to use addprocs()
using Plasmo

graph = OptiGraph()

#Add nodes to a GraphModel
@optinode(graph,n1)
@optinode(graph,n2)

@variable(n1,0 <= x <= 2)
@variable(n1,0 <= y <= 3)
@variable(n1, z >= 0)
@constraint(n1,x+y+z >= 4)
@objective(n1,Min,y)

@variable(n2,x)
@NLnodeconstraint(n2,ref,exp(x) >= 2)
@variable(n2,z >= 0)
@constraint(n2,z + x >= 4)
@objective(n2,Min,x)

@linkconstraint(graph,n1[:x] == n2[:x])

#Setup MPI manager
manager=MPIManager(np=2) # specify, number of mpi workers, launch cmd, etc.
addprocs(manager)        # start mpi workers and add them as julia workers too.


#Setup worker environments
#NOTE: You will need to load your Julia environment onto each worker
@everywhere using Pkg
@everywhere Pkg.activate((@__DIR__))

#Load Plasmo and PipsSolver on every worker
@everywhere using Plasmo
@everywhere using PipsSolver


julia_workers = sort(collect(values(manager.mpi2j))) # #Distribute the graph to workers

remote_references = PipsSolver.distribute(graph,julia_workers,remote_name = :pipsgraph)  #create the variable pipsgraph on each worker

# The remote modelgraphs can be queried if they are fetched from the other workers
r1 = fetch(remote_references[1])
r2 = fetch(remote_references[2])

#Solve with PIPS-NLP
@mpi_do manager begin
    using MPI
    PipsSolver.pipsnlp_solve(pipsgraph)
end

#Retrieve Solution
rank_zero = manager.mpi2j[0] #julia process representing rank 0
solution = fetch(@spawnat(rank_zero, pipsgraph))
```
