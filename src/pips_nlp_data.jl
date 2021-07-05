mutable struct PIPSNLPData
    d::Union{JuMP.NLPEvaluator,Nothing}
    n::Int
    m::Int
    local_m::Int                    #rank row dimension
    jacnnz::Int                     #Jacobian num nonzeros
    hessnnz::Int                    #Hessian num nonzeros
    firstIeq::Vector{Int}           #first stage equality constraint jacobian nonzero rows
    firstJeq::Vector{Int}           #first stage equality constraint jacobian nonzero cols
    firstVeq::Vector{Float64}       #first stage equality constraint jacobian values
    secondIeq::Vector{Int}          #second stage equality constraint jacobian nonzero rows
    secondJeq::Vector{Int}          #second stage equality constraint jacobian nonzero cols
    secondVeq::Vector{Float64}      #second stage equality constraint jacobian values
    firstIineq::Vector{Int}         #first stage inequality constraint jacobian nonzero rows
    firstJineq::Vector{Int}
    firstVineq::Vector{Float64}
    secondIineq::Vector{Int}
    secondJineq::Vector{Int}
    secondVineq::Vector{Float64}
    num_eqconnect::Int              #num equalities connecting first to second stage
    num_ineqconnect::Int            #num inequalities connecting first to second stage
    eqconnect_lb::Vector{Float64}   #connection equality lower bounds
    eqconnect_ub::Vector{Float64}   #connection equality upper bounds
    ineqconnect_lb::Vector{Float64} #connection inequality lower bounds
    ineqconnect_ub::Vector{Float64} #connection inequality upper bounds
    eq_idx::Vector{Int}             #equality connect indices
    ineq_idx::Vector{Int}           #inequality connect indices
    firstJeqmat
    secondJeqmat
    firstJineqmat
    secondJineqmat
    linkIeq::Vector{Int}            #equality link jacobian rows
    linkJeq::Vector{Int}            #equality link jacobian cols
    linkVeq::Vector{Float64}        #equality link jacobian values
    linkIineq::Vector{Int}
    linkJineq::Vector{Int}
    linkVineq::Vector{Float64}
    x_sol::Vector{Float64}          #solution on rank
    coreid::Int                     #rank
    loaded::Bool
    local_unsym_hessnnz::Int        #local hessian num nonzeros
end
PIPSNLPData() = PIPSNLPData(nothing,0,0,0,0,0,Int[],Int[], Float64[], Int[], Int[], Float64[],Int[],Int[],Float64[], Int[], Int[],
                            Float64[], 0, 0, Float64[],Float64[],Float64[],Float64[], Int[], Int[],
                            nothing, nothing, nothing, nothing, Int[],Int[], Float64[], Int[], Int[], Float64[],Float64[], 0, false, 0)

#data that each worker needs
mutable struct PipsNLPWorkerData
	first_stage::OptiNode
    n_linkeq_cons::Int64
    n_linkineq_cons::Int64
    link_ineq_lower::Vector{Float64}
    link_ineq_upper::Vector{Float64}
    link_eq_lower::Vector{Float64}
    link_eq_upper::Vector{Float64}
end

function _get_pips_data(node::OptiNode)
    return node.ext[:pips_data]
end

function _setup_pips_nlp_data!(graph::OptiGraph)

	if !(has_subgraphs(graph))
		#the first stage is empty
		first_stage = OptiNode()
		submodels = all_nodes(graph)
	elseif length(graph.subgraphs) == 1
		num_nodes(graph) == 1 || error("Incompatible optigraph structure.  Optigraph contains a single subgraph, but has multiple optinodes in the highest level graph.")
		first_stage = getnode(graph,1)
		submodels = all_nodes(graph.subgraphs[1])
	end

    model_list = [first_stage;submodels]
	for (idx,node) in enumerate(model_list)
		node.ext[:pips_data] = PIPSNLPData()
		constraint_data = get_constraint_data(node)
		node.ext[:constraint_data] = constraint_data
	end

	first_stage_data = _get_pips_data(first_stage)
	n_sub_models = length(submodels)
	model_list = [first_stage; submodels]

	#We assume every worker has the link constraint structure that connects across nodes
	#TODO: use MPI.ALLGATHER! to setup linkconstraints when they are different across workers?
	#NOTE: Each rank would need to know which links are equivalent e.g. using some sort of user-defined index.  For now,
	#we assume the user needs to model their problem correctly using ghost optinodes, or they use distribute_optigraph, which sets up
	#links correctly.  In the future, distributed optigraphs will make communicating the structure easier.

	#TODO: Check whether workers can use zeros for bounds on linkconstraints they don't need.

	#IDENTIFY LINK CONSTRAINTS
	link_connect_eq,link_connect_ineq,linkeqconstraints,linkineqconstraints  = _identify_linkconstraints(graph)

	#FIRST STAGE -> SECOND STAGE LINK CONSTRAINTS
	for link in link_connect_eq
		node = _get_subnode(first_stage,link)::OptiNode
		local_data = _get_pips_data(node)
		firstIeq = local_data.firstIeq
		firstJeq = local_data.firstJeq
		firstVeq = local_data.firstVeq
		secondIeq = local_data.secondIeq
		secondJeq = local_data.secondJeq
		secondVeq = local_data.secondVeq
		push!(local_data.eqconnect_lb, link.set.value)
		push!(local_data.eqconnect_ub, link.set.value)
		local_data.num_eqconnect += 1
		row = local_data.num_eqconnect
		for (var,coeff) in link.func.terms
			if getnode(var) == first_stage
				push!(firstIeq, row)
				push!(firstJeq, var.index.value)
				push!(firstVeq, coeff)
			else
				@assert getnode(var) == node
				push!(secondIeq, row)
				push!(secondJeq, var.index.value)
				push!(secondVeq, coeff)
			end
		end
	end

	for link in link_connect_ineq
		node = _get_subnode(first_stage,link)
		firstIineq = local_data.firstIineq
		firstJineq = local_data.firstJineq
		firstVineq = local_data.firstVineq
		secondIineq = local_data.secondIineq
		secondJineq = local_data.secondJineq
		secondVineq = local_data.secondVineq
		if isa(link.set,MOI.LessThan)
			push!(local_data.ineqconnect_lb, -Inf)
			push!(local_data.ineqconnect_ub, link.set.upper)
		elseif isa(link.set,MOI.GreaterThan)
			push!(local_data.ineqconnect_lb, link.set.lower)
			push!(local_data.ineqconnect_ub, Inf)
		elseif isa(link.set,MOI.Interval)
			push!(local_data.ineqconnect_lb, link.set.lower)
			push!(local_data.ineqconnect_ub, link.set.upper)
		end
		local_data.num_ineqconnect += 1
		row = local_data.num_ineqconnect
		for (var,coeff) in link.func.terms
			if getnode(var) == first_stage
				push!(firstIineq, row)
				push!(firstJineq, var.index.value)
				push!(firstVineq, ind)
			else
				@assert getnode(var) == node
				push!(secondIineq, row)
				push!(secondJineq, var.index.value)
				push!(secondVineq, ind)
			end
		end
	end

	#SECOND STAGE LINK CONSTRAINTS
	nlinkeq = length(linkeqconstraints)
	nlinkineq = length(linkineqconstraints)
	ineqlink_lb = zeros(nlinkineq)
	ineqlink_ub = zeros(nlinkineq)
	eqlink_lb = zeros(nlinkeq)
	eqlink_ub = zeros(nlinkeq)

	#INEQUALITY CONSTRAINTS
	#Inequality bounds
	#NOTE: linkconstraint bounds need to be on every rank.
	for (idx,link) in enumerate(linkineqconstraints)
		row = idx
		#Bounds
		if isa(link.set,MOI.LessThan)
			ineqlink_lb[row] = -Inf
			ineqlink_ub[row] = link.set.upper
		elseif isa(link.set,MOI.GreaterThan)
			ineqlink_lb[row] = link.set.lower
			ineqlink_ub[row] = Inf
		elseif isa(link.set,MOI.Interval)
			ineqlink_lb[row] = link.set.lower
			ineqlink_ub[row] = link.set.upper
		end
		#Inequality values
		for (var,coeff) in link.func.terms
			node = getnode(var)
			local_data = _get_pips_data(node)
			linkIineq = local_data.linkIineq
			linkJineq = local_data.linkJineq
			linkVineq = local_data.linkVineq
			push!(linkIineq, row)
			push!(linkJineq, var.index.value)
			push!(linkVineq, coeff)
		end
	end

	#EQUALITY CONSTRAINTS
	for (idx,link) in enumerate(linkeqconstraints)
		row = idx
		#Bounds
		eqlink_lb[row] = link.set.value
		eqlink_ub[row] = link.set.value
		#Equality values
		for (var,coeff) in link.func.terms
			node = getnode(var)
			local_data = _get_pips_data(node)
			linkIeq = local_data.linkIeq
			linkJeq = local_data.linkJeq
			linkVeq = local_data.linkVeq
			push!(linkIeq, row)              #the variable row
			push!(linkJeq, var.index.value)  #the variable column
			push!(linkVeq, coeff)            #the coefficient
		end
	end

    worker_data = PipsNLPWorkerData(
    first_stage,
    nlinkeq,
    nlinkineq,
    ineqlink_lb,
    ineqlink_ub,
    eqlink_lb,
    eqlink_ub)

	return worker_data
end

#categorize link constraints into hierarchical and distributed
function _identify_linkconstraints(graph::OptiGraph)
    links_connect_eq = LinkConstraint[]
    links_connect_ineq = LinkConstraint[]
    links_eq = LinkConstraint[]
    links_ineq = LinkConstraint[]

	if !(has_subgraphs(graph))
		for link in all_linkconstraints(graph)
			link_constraint = constraint_object(link)
			if isa(link_constraint.set,MOI.EqualTo)
				push!(links_eq,link_constraint)
			else
				push!(links_ineq,link_constraint)
			end
		end
	else
        first_stage_nodes = getnodes(graph)
	    sub_nodes = setdiff(all_nodes(graph),first_stage_nodes)
	    for link in all_linkconstraints(graph)
	        link_constraint = constraint_object(link)
	        link_nodes = getnodes(link_constraint)

	        #if first_stage_node in link_nodes
			if isempty(setdiff(first_stage_nodes,link_nodes))
	            @assert length(link_nodes) == 2
	            if isa(link_constraint.set,MOI.EqualTo)
	                push!(links_connect_eq,link_constraint)
	            else
	                push!(links_connect_ineq,link_constraint)
	            end
	        else
	            if isa(link_constraint.set,MOI.EqualTo)
	                push!(links_eq,link_constraint)
	            else
	                push!(links_ineq,link_constraint)
	            end
	        end
	    end
	end
    return links_connect_eq,links_connect_ineq,links_eq,links_ineq
end
