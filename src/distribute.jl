#Distribute a optigraph among workers.  Each worker should have the same first stage model.  Each worker will be allocated some of the nodes in the original modelgraph
function distribute_optigraph(graph::OptiGraph,to_workers::Vector{Int64};remote_name = :graph)
    #NOTE: Does not yet support subgraphs.  Need to aggregate first.
    to_workers = sort(to_workers)
    n_nodes = num_nodes(graph)
    n_workers = length(to_workers)
    nodes_per_worker = Int64(floor(n_nodes/n_workers))
    nodes = all_nodes(graph)
    node_indices = [getindex(graph,node) for node in nodes]
    user_pips_data = PipsNLP._setup_pips_nlp_data!(graph)

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
            ref2 = @spawnat worker Core.eval(Main, Expr(:(=), remote_name, PipsNLP._create_worker_optigraph(getfield(Main,:nodes),
            node_indices[i],
            user_pips_data,
            n_nodes)))
            push!(remote_references,ref2)
        end
    end
    #graph.ext[:allocations] = allocations
    return remote_references
end

function fill_solution!(graph::OptiGraph,remote_name::Symbol,worker::Int64)
    for i = 1:num_all_nodes(graph)
        x_sol = @fetchfrom worker getnode(getfield(Main,remote_name),i).ext[:pips_data].x_sol
        status = @fetchfrom worker termination_status(getnode(getfield(Main,remote_name),i))

        node = getnode(graph,i)
        Plasmo.set_node_primals(node,all_variables(node),x_sol)
        Plasmo.set_node_status(node,status)
    end
    return nothing
end

function _create_worker_optigraph(optinodes::Vector{OptiNode},
    node_indices::Vector{Int64},
    user_pips_data,
    n_nodes)

    graph = OptiGraph()
    graph.node_idx_map = Dict{OptiNode,Int64}()
    graph.ext[:user_pips_data] = user_pips_data

    #Add nodes to worker's graph.  Each worker should have the same number of nodes, but some will be empty.
    for i = 1:n_nodes
        node = add_node!(graph)
        node.ext[:pips_data] = PIPSNLPData()  #all nodes need this
    end
    for (i,index) in enumerate(node_indices)
        graph.optinodes[index] = optinodes[i]
    end

    return graph
end
