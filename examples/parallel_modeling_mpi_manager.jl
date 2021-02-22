#Example: Using MPI Manager to do parallel modeling with PIPS-NLP
using MPIClusterManagers
using Distributed

manager = MPIManager(np = 3)
addprocs(manager)

@mpi_do manager begin
    #Run mpi on workers
    using Pkg
    Pkg.activate("./")
    using Plasmo
    using MPI
    comm=MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    println("I am $(rank) of $(MPI.Comm_size(comm))")
    MPI.Barrier(comm)
end

@mpi_do manager begin
    #All ranks have optigraph and first stage optinode
    graph = OptiGraph()

    @optinode(graph,n1)
    @variable(n1,x0 >= 0)

    subgraph = OptiGraph()
    add_subgraph!(graph,subgraph)
    @optinode(subgraph,n2)
    @optinode(subgraph,n3)
    @optinode(subgraph,n4)

    #Linked variables using linkconstraints
    @variable(n2,x >= 0)
    @variable(n3,x >= 0)
    @variable(n4,x >= 0)

    #Local models
    if rank == 0
        @variable(n2,x2 >= 2)
        @variable(n2,y2 >= 0)
        @constraint(n2,x2 + y2 == 3)
        @NLconstraint(n2,x2^3 <= 10)
        @constraint(n2,n2[:x] + n2[:x2] <= 10)
        @linkconstraint(graph,n1[:x0] == y2)
        @objective(n2,Min,x2)
    elseif rank == 1
        @variable(n3,x3 >= 2)
        @variable(n3,y3 >= 0)
        @constraint(n3,x3 + y3 == 4)
        @constraint(n3,n3[:x] + n3[:x3] <= 10)
        @linkconstraint(graph,n1[:x0] == y3)
        @objective(n3,Min,x3)
    elseif rank == 2
        @variable(n4,x4 >= 2)
        @variable(n4,y4 >= 0)
        @constraint(n4,x4 + y4 == 5)
        @constraint(n4,n4[:x] + n4[:x4] <= 10)
        @linkconstraint(graph,n1[:x0] == y4)
        @objective(n4,Min,x4)
    end

    #Link constraints between subnodes
    @linkconstraint(graph,n2[:x] == n3[:x])
    @linkconstraint(graph,n3[:x] == n4[:x])

end

#Solve with PIPS-NLP
@mpi_do manager begin
    using PipsSolver
    pipsnlp_solve(graph)
end

#Use Julia to fetch solution values from each rank
solution_values = []
for worker in workers()
    solution_worker = fetch(@spawnat(worker,value.(all_variables(graph))))
    push!(solution_values,solution_worker)
end

#Use MPI Manager to print solution values on each rank
@mpi_do manager begin
    if rank == 0
        println("first stage: ")
        @show (value(n1[:x0]))

        println("second stage: n2")
        @show (value(n2[:y2]))
        @show (value(n2[:x]))
    end
end

@mpi_do manager begin
    if rank == 1
        println("second stage: n3")
        @show (value(n3[:y3]))
        @show (value(n3[:x]))
    end
end

@mpi_do manager begin
    if rank == 2
        println("second stage: n4")
        @show (value(n4[:y4]))
        @show (value(n4[:x]))
    end
end
