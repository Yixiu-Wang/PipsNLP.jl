using ModelGraphs
using Ipopt

graph = ModelGraph()
optimizer = with_optimizer(Ipopt.Optimizer)


# @linkvariable(graph,z[1:2])
# @masterconstraint(graph,z[1] + z[2] <= 2)

#Add nodes to a GraphModel
n1 = add_node!(graph)
n2 = add_node!(graph)

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

#Link constraints take the same expressions as the JuMP @constraint macro
@linkconstraint(graph,n1[:x] == n2[:x])
#@graphobjective(graph,Min,n1[:y] + n2[:x])

# optimize!(graph,optimizer)
#
# println("n1[:x]= ",nodevalue(n1[:x]))
# println("n1[:y]= ",nodevalue(n1[:y]))
# println("n2[:x]= ",nodevalue(n2[:x]))
# println("objective = ", objective_value(graph))
