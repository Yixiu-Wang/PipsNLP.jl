function distribute_optigraph(graph::OptiGraph,to_workers::Vector{Int64};remote_name = :graph)
    error("distribute() has not yet been updated for the latest version of PipsNLP.jl")
end

#TODO: Improve distribute performance.

#Distribute a modelgraph among workers.  Each worker should have the same master model.  Each worker will be allocated some of the nodes in the original modelgraph
function distribute_optigraph(graph::OptiGraph,to_workers::Vector{Int64};remote_name = :graph)
    #NOTE: Linkconstraints keep their indices in new graphs,
    #NOTE: Link constraint row index needs to match on each worker
    #NOTE: Does not yet support subgraphs.  Need to aggregate first

    #IDEA: Create a channel from the master process to each worker?
    # channel_nodes = RemoteChannel(1)    #we will allocate and send nodes to workers
    # channel_indices = RemoteChannel(1)
    to_workers = sort(to_workers)
    n_nodes = num_nodes(graph)
    n_workers = length(to_workers)
    nodes_per_worker = Int64(floor(n_nodes/n_workers))
    nodes = all_nodes(graph)
    node_indices = [getindex(graph,node) for node in nodes]

    user_pips_data = _setup_pips_nlp_data!(graph)

    #broadcast this to each worker
    @everywhere  Core.eval(Main, Expr(:(=), pips_nlp_data,user_pips_data))

    #pips_nlp_solve checks for this key to use
    graph.ext[:user_pips_data] = user_pips_data




    #link_connect_eq,link_connect_ineq,linkeqconstraints,linkineqconstraints  = _identify_linkconstraints(graph)

    #setup the data for each node here.  Then send them off to workers


    # linkeqconstraints = OrderedDict()
    # linkineqconstraints = OrderedDict()
    # for edge in getedges(mg)
    #     for (idx,link) in edge.linkeqconstraints
    #         linkeqconstraints[idx] = link
    #     end
    #     for (idx,link) in edge.linkineqconstraints
    #         linkineqconstraints[idx] = link
    #     end
    # end

    # n_linkeq_cons =  length(linkeqconstraints)
    # n_linkineq_cons = length(linkineqconstraints)
    #
    # ineqlink_lb = zeros(n_linkineq_cons)
    # ineqlink_ub = zeros(n_linkineq_cons)
    #
    # for (idx,link) in linkineqconstraints
    #     if isa(link.set,MOI.LessThan)
    #         ineqlink_lb[idx] = -Inf
    #         ineqlink_ub[idx] = link.set.upper
    #     elseif isa(link.set,MOI.GreaterThan)
    #         ineqlink_lb[idx] = link.set.lower
    #         ineqlink_ub[idx] = Inf
    #     elseif isa(link.set,MOI.Interval)
    #         ineqlink_lb[idx] = link.set.lower
    #         ineqlink_ub[idx] = link.set.upper
    #     end
    # end

    #Allocate optinodes onto workers.  Split it up evenly
    allocations = []
    node_indices = []
    j = 1
    while  j <= n_nodes
        if j + nodes_per_worker > n_nodes
            push!(allocations,nodes[j:end])
            push!(node_indices,[getindex(graph,node) for node in nodes[j:end]])
        else
            push!(allocations,nodes[j:j+nodes_per_worker - 1])
            push!(node_indices,[getindex(graph,node) for node in nodes[j:j+nodes_per_worker - 1]])
        end
        j += nodes_per_worker
    end

    println("Distributing optigraph among workers: $to_workers")
    channel = RemoteChannel(1)
    remote_references = []
    #Fill channel with sets of nodes to send
    #TODO: Make this runs in parallel with an @async?
    @sync begin
        for (i,worker) in enumerate(to_workers)
            @spawnat(1, put!(channel, allocations[i]))
            ref1 = @spawnat worker begin
                Core.eval(Main, Expr(:(=), :nodes, take!(channel)))
            end
            wait(ref1)
            ref2 = @spawnat worker Core.eval(Main, Expr(:(=), remote_name,
            PipsNLP._create_worker_optigraph(getfield(Main,:nodes),
            node_indices[i],
            n_nodes,
            n_linkeq_cons,
            n_linkineq_cons,
            ineqlink_lb,
            ineqlink_ub)))
            push!(remote_references,ref2)
        end
    end
    return remote_references
end

function _create_worker_optigraph(optinodes::Vector{OptiNode},
    node_indices::Vector{Int64},
    n_nodes::Int64,
    n_linkeq_cons::Int64,
    n_linkineq_cons::Int64,
    link_ineq_lower::Vector,
    link_ineq_upper::Vector)

    graph = OptiGraph()
    graph.node_idx_map = Dict{OptiNode,Int64}()

    #Add nodes to worker's graph.  Each worker should have the same number of nodes, but some will be empty.
    #TODO: optigraph_reference?
    for i = 1:n_nodes
        add_node!(graph)
    end
    for (i,index) in enumerate(node_indices)
        graph.optinodes[index] = optinodes[i]
    end

    #Setup user PipsNLP data.
    user_pips_data = _setup_pips_nlp_data!()
    graph.ext[:user_pips_data] = user_pips_data

    #Populate models for given nodes
    # for (i,node) in enumerate(optinodes)
    #     index = node_indices[i]  #need node index in highest level
    #     new_node = getnode(graph,index)
    #     set_model(new_node,getmodel(node))
    # end
    # We need the graph to have the partial constraints over graph nodes
    # Add link constraints
    # graph.linkeqconstraints = _add_linkeq_terms(modelnodes)
    # graph.linkineqconstraints = _add_linkineq_terms(modelnodes)

    #Use node link-constraint information to setup pipsnlp data

    #Setup new graph linkconstraints
    linkeqconstraints = _add_linkeq_terms(optinodes)
    linkineqconstraints = _add_linkineq_terms(optinodes)

    #Need to match both equality and inequality
    #Add linkconstraints, then fix indices
    for (idx,link) in linkeqconstraints
        cref = Plasmo.add_link_equality_constraint(graph,JuMP.ScalarConstraint(link.func,link.set);eq_idx = idx)
    end

    for (idx,link) in linkineqconstraints
        cref = Plasmo.add_link_inequality_constraint(graph,JuMP.ScalarConstraint(link.func,link.set);ineq_idx = idx)
    end



    return graph
end


# function _add_linkeq_terms(modelnodes::Vector{OptiNode})
#     linkeqconstraints = OrderedDict()
#     for node in modelnodes
#         partial_links = node.partial_linkeqconstraints
#         for (idx,linkconstraint) in partial_links
#             if !(haskey(linkeqconstraints,idx))   #create link constraint
#                 new_func = linkconstraint.func
#                 set = linkconstraint.set
#                 linkcon = LinkConstraint(new_func,set)
#                 linkeqconstraints[idx] = linkcon
#             else #update linkconstraint
#                 newlinkcon = linkeqconstraints[idx]
#                 nodelinkcon = node.partial_linkeqconstraints[idx]
#                 newlinkcon = LinkConstraint(newlinkcon.func + nodelinkcon.func,newlinkcon.set)
#                 linkeqconstraints[idx] = newlinkcon
#             end
#         end
#     end
#     return linkeqconstraints
# end
#
# function _add_linkineq_terms(modelnodes::Vector{OptiNode})
#     linkineqconstraints = OrderedDict()
#     for node in modelnodes
#         partial_links = node.partial_linkineqconstraints
#         for (idx,linkconstraint) in partial_links
#             if !(haskey(linkineqconstraints,idx))   #create link constraint
#                 new_func = linkconstraint.func
#                 set = linkconstraint.set
#                 linkcon = LinkConstraint(new_func,set)
#                 linkineqconstraints[idx] = linkcon
#             else #update linkconstraint
#                 newlinkcon = linkineqconstraints[idx]
#                 nodelinkcon = node.partial_linkineqconstraints[idx]
#                 newlinkcon = LinkConstraint(newlinkcon.func + nodelinkcon.func,newlinkcon.set)
#                 linkineqconstraints[idx] = newlinkcon
#             end
#         end
#     end
#     return linkineqconstraints
# end

# #need to change the constraint index to match the original linkconstraint
# linkedge = cref.linkedge
# old_idx = graph.linkeqconstraint_index
#
# #change index
# linkedge.linkeqconstraints[idx] = linkedge.linkeqconstraints[old_idx]
#
# #we don't always want to delete
# if old_idx != idx &&  linkedge.linkeqconstraints[old_idx]
#     delete!(linkedge.linkeqconstraints,old_idx)
# end

# linkedge = cref.linkedge
# old_idx = graph.linkineqconstraint_index
# linkedge.linkineqconstraints[idx] = linkedge.linkineqconstraints[old_idx] #old_idx is the wrong index
#
# if old_idx != idx
#     delete!(linkedge.linkineqconstraints,old_idx)
# end

# ref2 = @spawnat worker Core.eval(Main, Expr(:(=), remote_name, PipsNLP._create_worker_optigraph(getfield(Main,:nodes),
# getfield(Main,:node_indices),

#Core.eval(Main, Expr(:(=), :node_indices, take!(channel_indices)))
#@spawnat(1, put!(channel_indices, node_indices[i]))
_
