using Plasmo
using Ipopt

graph = ModelGraph()
optimizer = with_optimizer(Ipopt.Optimizer)

n1 = add_node!(graph)
n2 = add_node!(graph)
n3 = add_node!(graph)
n4 = add_node!(graph)

@variable(n1,0 <= x <= 10, start = 1)
@variable(n1,0 <= y <= 3 , start = 1)
@variable(n1, z >= 0, start = 1)
@constraint(n1,x+y+z >= 1)
@objective(n1,Min,x+y+z)

@variable(n2,x >= 0, start = 1)
@NLnodeconstraint(n2,ref,exp(x) >= 2)
@variable(n2,z >= 0, start = 2)
@constraint(n2,z + x >= 4)
@objective(n2,Min,x + z)

@variable(n3,x[1:2] >= 0, start = 1)
@NLnodeconstraint(n3,ref,exp(x[1]) >= 1)
@constraint(n3,sum(x[i] for i = 1:2) <= 10)
@objective(n3,Min,x[1] + x[2])

@variable(n4,x[1:2] >= 0, start = 1)
@constraint(n4,sum(x[i] for i = 1:2) >= 0)
@NLnodeconstraint(n4,ref,exp(x[2]) >= 0)
@objective(n4,Min,x[1] + x[2])

#Link constraints take the same expressions as the JuMP @constraint macro
@linkconstraint(graph, n1[:x] == n2[:z])
@linkconstraint(graph, n1[:x] + n2[:x] + n3[:x][1] <= n4[:x][2])
@linkconstraint(graph, n3[:x][1] == n4[:x][1])
@linkconstraint(graph, n2[:x] >= n1[:x])  #I this causes issues because of how I treat upper and lower bounds

# @linkconstraint(graph, sum(n4[:x][i] for i = 1:3) <= sum(n3[:x][i] for i = 1:2))

#@graphobjective(graph,Min,n1[:y] + n2[:x])

# optimize!(graph,optimizer)
#
# println("n1[:x]= ",nodevalue(n1[:x]))
# println("n1[:y]= ",nodevalue(n1[:y]))
# println("n2[:x]= ",nodevalue(n2[:x]))
# println("objective = ", objective_value(graph))
