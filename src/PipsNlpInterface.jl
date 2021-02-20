module PipsNlpInterface

#Import Packages
using SparseArrays
using LinearAlgebra
using DataStructures
import MPI

import JuMP
import MathOptInterface
const MOI = MathOptInterface
using Plasmo

include("PipsNlpCInterface.jl")
using .PipsNlpSolver

include("pips_utils.jl")
include("nlp_evaluator.jl")

export pipsnlp_solve

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
    x_sol::Vector{Float64}          #solution
    coreid::Int                     #rank
    loaded::Bool
    local_unsym_hessnnz::Int        #local hessian num nonzeros
end
PIPSNLPData() = PIPSNLPData(nothing,0,0,0,0,0,Int[],Int[], Float64[], Int[], Int[], Float64[],Int[],Int[],Float64[], Int[], Int[],
                            Float64[], 0, 0, Float64[],Float64[],Float64[],Float64[], Int[], Int[],
                            nothing, nothing, nothing, nothing, Int[],Int[], Float64[], Int[], Int[], Float64[],Float64[], 0, false, 0)


function _get_pips_data(node::OptiNode)
    return node.ext[:pips_data]
end

function pipsnlp_solve(graph::OptiGraph) #Assume graph variables and constraints are first stage
    #Check structure: 3 possibilities
    #1.) optigraph without subgraphs (only link constraints, no first stage -> second stage constraints)
    #2.) optigraph with single subgraph (first stage and linkconstraints between optinodes)
    #3.) optigraph with multiple subgraphs (first stage and linkconstraints between subgraphs) Not yet supported

    if !(has_subgraphs(graph))
        first_stage = OptiNode() #empty first stage
        submodels = all_nodes(graph)
    elseif length(graph.subgraphs) == 1
        num_nodes(graph) == 1 || error("Incompatible optigraph structure.  Optigraph contains a single subgraph, but has multiple optinodes in highest level graph.")
        first_stage = getnode(graph,1)
        submodels = all_nodes(graph.subgraphs[1])
    end

    comm = MPI.COMM_WORLD
    if MPI.Comm_rank(comm) == 0
        println("Building Model for PIPS-NLP")
    end

    n_sub_models = length(submodels)

    model_list = [first_stage; submodels]

    for (idx,node) in enumerate(model_list)
        node.ext[:pips_data] = PIPSNLPData()
        constraint_data = get_constraint_data(node)
        node.ext[:constraint_data] = constraint_data
    end
    first_stage_data = _get_pips_data(first_stage)

    #Assume every worker has every linkconstraint that connects across nodes
    #TODO: MPI.ALLGATHER! to setup linkconstraints when they are different across workers
    #TODO: Check whether workers can use zeros for bounds they don't need.
    #TODO: Improve distribute performance.

    #IDENTIFY LINK CONSTRAINTS
    # link_constraints = all_linkconstraints(graph)
    link_connect_eq,link_connect_ineq,linkeqconstraints,linkineqconstraints  = _identify_linkconstraints(graph)

    #NOTE: linkconstraint bounds need to be on every rank.
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
    for (idx,link) in enumerate(linkineqconstraints)
        row = idx
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
        #Equality bounds
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

    #Add constraint data for each node to help with function evaluations



    #Initialize
    function str_init_x0(nodeid, x0)
        node = model_list[nodeid+1]
        jump_initval = copy(JuMP.start_value.(JuMP.all_variables(node)))
        local_initval = Vector{Float64}(undef,JuMP.num_variables(node))

        nothing_values = isa.(jump_initval,Nothing)
        float_values = .!(nothing_values)
        local_initval[nothing_values] .= 1  #set to 1 by default
        local_initval[float_values] .= jump_initval[float_values]
        original_copy(local_initval,x0)
    end

    #Problem Info
    function str_prob_info(nodeid,flag,mode,col_lb,col_ub,row_lb,row_ub)
        if flag != 1
            node = model_list[nodeid+1]
            local_data = _get_pips_data(node)
            #Do a warm start on the node
            if !(local_data.loaded)
                local_data.loaded = true
                if (nodeid > 0)
                    if haskey(node.ext, :warmStart)
    	                if node.ext[:warmStart] == true
              	      	   solve(node)
                   	   	end
                    end
            	end

    			#nlp_lb, nlp_ub = JuMP.constraintbounds(node)  #This is every constraint in the model
                nlp_lb, nlp_ub = constraintbounds(node.ext[:constraint_data])       #This is every constraint in the model
         		local_data.local_m  = length(nlp_lb)          #number of local constraints (rows)

    			newRowId = Array{Int}(undef,local_data.local_m)
    			eqId = 1
    			ineqId = 1

                #Go through all constraints, check if they are equality of inequality
                for c in 1:local_data.local_m
                    if nlp_lb[c] == nlp_ub[c]
                        push!(local_data.eq_idx, c)
                        newRowId[c] = eqId
                        eqId +=  1
                    else
        				push!(local_data.ineq_idx, c)
        				newRowId[c] = ineqId
        				ineqId += 1
                    end
         		end

                #Local node data
                local_data.m  = local_data.local_m + local_data.num_eqconnect + local_data.num_ineqconnect
                local_data.n = JuMP.num_variables(node)
                local_data.x_sol = zeros(Float64,local_data.n)
                local_data.firstJeqmat = sparse(local_data.firstIeq, local_data.firstJeq, local_data.firstVeq, local_data.num_eqconnect, first_stage_data.n)
                local_data.secondJeqmat = sparse(local_data.secondIeq, local_data.secondJeq, local_data.secondVeq, local_data.num_eqconnect, local_data.n)
                local_data.firstJineqmat = sparse(local_data.firstIineq, local_data.firstJineq, local_data.firstVineq, local_data.num_ineqconnect, first_stage_data.n)
                local_data.secondJineqmat = sparse(local_data.secondIineq, local_data.secondJineq, local_data.secondVineq, local_data.num_ineqconnect, local_data.n)

                if node.nlp_data == nothing
                    JuMP._init_NLP(node)
                end
                local_data.d = JuMP.NLPEvaluator(node)
                MOI.initialize(local_data.d, [:Grad,:Jac, :Hess])
                #Ijac, Jjac = jac_structure(local_data.d)
                jac_structure = pips_jacobian_structure(local_data.d)
                Ijac = [jac_structure[i][1] for i = 1:length(jac_structure)]
                Jjac = [jac_structure[j][2] for j = 1:length(jac_structure)]
                Ijaceq = Int[]
                Jjaceq = Int[]
                Ijacineq = Int[]
                Jjacineq = Int[]
                jac_eq_index = Int[]
                jac_ineq_index = Int[]
                for i in 1:length(Ijac)
                    c = Ijac[i]
                    if nlp_lb[c] == nlp_ub[c]
                        modifiedrow = newRowId[c]
                        push!(Ijaceq, modifiedrow)
                        push!(Jjaceq, Jjac[i])
                        push!(jac_eq_index, i)
                    else
                        modifiedrow = newRowId[c]
                        push!(Ijacineq, modifiedrow)
                        push!(Jjacineq, Jjac[i])
                        push!(jac_ineq_index,i)
                    end
                end

        		node.ext[:Ijaceq] = Ijaceq
         		node.ext[:Jjaceq] = Jjaceq
                node.ext[:Ijacineq] = Ijacineq
         		node.ext[:Jjacineq] = Jjacineq
         		node.ext[:jac_eq_index] = jac_eq_index
         		node.ext[:jac_ineq_index] = jac_ineq_index
         		#Ihess, Jhess = hesslag_structure(local_data.d)
                hess_structure = pips_hessian_lagrangian_structure(local_data.d)
                Ihess = [hess_structure[i][1] for i = 1:length(hess_structure)]
                Jhess = [hess_structure[j][2] for j = 1:length(hess_structure)]

         		Hmap = Bool[]
         		node_Hrows = Int[]
         		node_Hcols = Int[]
         		for i in 1:length(Ihess)
            	    if Jhess[i] <= Ihess[i]
               	        push!(node_Hrows, Ihess[i])
                   		push!(node_Hcols, Jhess[i])
                   		push!(Hmap, true)
            	    else
                        push!(Hmap, false)
                    end
                end
         		val = ones(Float64,length(node_Hrows))
         		mat = sparse(node_Hrows,node_Hcols,val, local_data.n, local_data.n)
         		node.ext[:Hrows] = node_Hrows
         		node.ext[:Hcols] = node_Hcols
         		node.ext[:Hmap] = Hmap
         		local_hessnnz = length(mat.rowval)
         		local_data.local_unsym_hessnnz = length(Ihess)
         		local_data.hessnnz = local_hessnnz
		    end

	 	    if mode == :Values
               	#nlp_lb, nlp_ub = JuMP.constraintbounds(node)
                nlp_lb, nlp_ub = constraintbounds(node.ext[:constraint_data])
    			eq_lb=Float64[]
    			eq_ub=Float64[]
    			ineq_lb=Float64[]
    			ineq_ub=Float64[]
    			for i in 1: length(nlp_lb)
    			    if nlp_lb[i] == nlp_ub[i]
    			       push!(eq_lb, nlp_lb[i])
    			       push!(eq_ub, nlp_ub[i])
    			    else
    			       push!(ineq_lb, nlp_lb[i])
    			       push!(ineq_ub, nlp_ub[i])
    			    end
    			end

    			if nodeid !=  0 #0 is the master
    			   eq_lb = [eq_lb; local_data.eqconnect_lb]
    			   eq_ub = [eq_ub; local_data.eqconnect_ub]
    			   ineq_lb = [ineq_lb; local_data.ineqconnect_lb]
    			   ineq_ub = [ineq_ub; local_data.ineqconnect_ub]
    			end

    			original_copy([eq_lb;ineq_lb], row_lb)
    			original_copy([eq_ub;ineq_ub], row_ub)
                colLower = variablelowerbounds(node)
                colUpper = variableupperbounds(node)
                original_copy(colLower, col_lb)
    			original_copy(colUpper, col_ub)
            end
		    return (local_data.n,local_data.m)
		else
		    if mode == :Values
                # println("row_lb: ",row_lb)
                # println("row_ub: ",row_ub)
                # println(eqlink_lb)
                # println(eqlink_ub)
                # println(ineqlink_lb)
                # println(ineqlink_ub)
                original_copy([eqlink_lb; ineqlink_lb], row_lb)
                original_copy([eqlink_ub; ineqlink_ub], row_ub)
            end
            return (0, nlinkeq + nlinkineq)
		end
    end

    #x0 is first stage variable values, x1 is local values
    function str_eval_f(nodeid,x0,x1)
    	node = model_list[nodeid+1] #Julia doesn't start index at 0
        local_data = _get_pips_data(node)
        local_d = _get_pips_data(node).d
        if nodeid ==  0
            local_x = x0
        else
            local_x = x1
        end
        #check objective sign
        #local_scl = (node.objSense == :Min) ? 1.0 : -1.0
        local_scl = (JuMP.objective_sense(node) == MOI.MAX_SENSE) ? -1.0 : 1.0
        f = local_scl*pips_eval_objective(local_d,local_x)
        return f
    end

    function str_eval_g(nodeid,x0,x1,new_eq_g, new_inq_g)

        node = model_list[nodeid+1]
        local_data = _get_pips_data(node)
        local_d = _get_pips_data(node).d
        if nodeid ==  0
            local_x = x0
        else
            local_x = x1
        end
        local_g = Array{Float64}(undef,local_data.local_m)
        pips_eval_constraint(local_d, local_g, local_x)

        #evaluates local constraints and connecting constraints with first stage
        new_eq_g[1:end] = [local_g[local_data.eq_idx]; local_data.firstJeqmat*x0+local_data.secondJeqmat*x1]
        new_inq_g[1:end] = [local_g[local_data.ineq_idx]; local_data.firstJineqmat*x0+local_data.secondJineqmat*x1]
	    return Int32(1)
    end

    function str_write_solution(id::Integer, x::Vector{Float64}, y_eq::Vector{Float64}, y_ieq::Vector{Float64})
        node = model_list[id+1]
        local_data = _get_pips_data(node)
        local_data.x_sol = copy(x)

        #TODO: Grab dual values

        #NOTE: This won't work anymore
        # if id == 0
        #     #Add objective value to node
        #     node.ext[:objective] = str_eval_f(id, x, nothing)
        # else
        #     #node.objVal = str_eval_f(id, nothing, x)
        #     node.ext[:objective] = str_eval_f(id, nothing, x)
        # end
        rank = MPI.Comm_rank(comm)
        local_data.coreid = rank
    end


    function str_eval_grad_f(rowid,colid,x0,x1,new_grad_f)
        #println("eval_grad_f")
        node = model_list[rowid+1]
        if rowid == colid
            local_data = _get_pips_data(node)
            local_d = _get_pips_data(node).d
            if colid ==  0
                local_x = x0
            else
                local_x = x1
            end
            local_grad_f = Array{Float64}(undef,local_data.n)
            pips_eval_objective_gradient(local_d, local_grad_f, local_x)
            #eval_grad_f(local_d, local_grad_f, local_x)
            #local_scl = (node.objSense == :Min) ? 1.0 : -1.0
            local_scl = (JuMP.objective_sense(node) == MOI.MAX_SENSE) ? -1.0 : 1.0
            #scale!(local_grad_f,local_scl)
            rmul!(local_grad_f,local_scl)
            original_copy(local_grad_f, new_grad_f)
        elseif colid == 0
            new_grad_f[1:end] .= 0
        else
            @assert(false)
        end
        return Int32(1)
    end

    #NOTE: Why does it subtract 1?
    function array_copy(src,dest)
        @assert(length(src)==length(dest))
        for i in 1:length(src)
            dest[i] = src[i]-1  #NOTE: Is this copying indices to a C vector?
        end
    end

    function original_copy(src,dest)
        @assert(length(src)==length(dest))
        for i in 1:length(src)
            dest[i] = src[i]
        end
    end

    function str_eval_jac_g(rowid,colid,flag, x0,x1,mode,e_rowidx,e_colptr,e_values,i_rowidx,i_colptr,i_values)
        if flag != 1 #populate parent child structure
            node = model_list[rowid+1]
            local_data = _get_pips_data(node)
            local_m_eq = length(local_data.eq_idx)
            local_m_ineq = length(local_data.ineq_idx)
            if mode == :Structure
                if (rowid == colid)
                    Ieq=[node.ext[:Ijaceq];local_data.secondIeq .+ local_m_eq]
                    Jeq=[node.ext[:Jjaceq];local_data.secondJeq]
                    Veq= ones(Float64, length(Ieq))
                    Iineq=[node.ext[:Ijacineq];local_data.secondIineq .+ local_m_ineq]
                    Jineq=[node.ext[:Jjacineq];local_data.secondJineq]
                    Vineq=ones(Float64, length(Iineq))
                    eqmat = sparse(Ieq, Jeq, Veq, local_m_eq + local_data.num_eqconnect, local_data.n)
                    ineqmat = sparse(Iineq, Jineq, Vineq, local_m_ineq + local_data.num_ineqconnect, local_data.n)
                else
                    eqmat = sparse(local_m_eq .+ local_data.firstIeq, local_data.firstJeq, local_data.firstVeq, local_m_eq .+ local_data.num_eqconnect, first_stage_data.n)
                    ineqmat = sparse(local_m_ineq .+ local_data.firstIineq, local_data.firstJineq, local_data.firstVineq, local_m_ineq .+ local_data.num_ineqconnect, first_stage_data.n)
                end
                return(length(eqmat.rowval), length(ineqmat.rowval))
            else #mode = :Values
                if rowid == colid    #evaluate Jacobian for block
                    if colid ==  0   #if it's the root node
                        local_x = x0
                    else
                        local_x = x1
                    end
                    local_values = Array{Float64}(undef,length(node.ext[:Ijaceq])+length(node.ext[:Ijacineq]))
                    #eval_jac_g(local_data.d, local_values, local_x)
                    pips_eval_constraint_jacobian(local_data.d, local_values, local_x)
                    jac_eq_index = node.ext[:jac_eq_index]
                    jac_ineq_index = node.ext[:jac_ineq_index]
                    Ieq=[node.ext[:Ijaceq];local_data.secondIeq .+ local_m_eq]
                    Jeq=[node.ext[:Jjaceq];local_data.secondJeq]
                    Veq=[local_values[jac_eq_index];local_data.secondVeq]
                    Iineq=[node.ext[:Ijacineq];local_data.secondIineq .+ local_m_ineq]
                    Jineq=[node.ext[:Jjacineq];local_data.secondJineq]
                    Vineq=[local_values[jac_ineq_index];local_data.secondVineq]
                    eqmat = sparseKeepZero(Ieq, Jeq, Veq, local_m_eq + local_data.num_eqconnect, local_data.n)
                    ineqmat = sparseKeepZero(Iineq, Jineq, Vineq, local_m_ineq + local_data.num_ineqconnect, local_data.n)
                else #evaluate
                    eqmat = sparseKeepZero(local_m_eq .+ local_data.firstIeq, local_data.firstJeq, local_data.firstVeq, local_m_eq .+ local_data.num_eqconnect, first_stage_data.n)
                    ineqmat = sparseKeepZero(local_m_ineq .+ local_data.firstIineq, local_data.firstJineq, local_data.firstVineq, local_m_ineq .+ local_data.num_ineqconnect, first_stage_data.n)
                end
                if length(eqmat.nzval) > 0
                    array_copy(eqmat.rowval,e_rowidx)
                    array_copy(eqmat.colptr,e_colptr)
                    original_copy(eqmat.nzval,e_values)
                end
                if length(ineqmat.nzval) > 0
                    array_copy(ineqmat.rowval,i_rowidx)
                    array_copy(ineqmat.colptr,i_colptr)
                    original_copy(ineqmat.nzval, i_values)
                end
            end
        else #populate linkconstraint structure
            node = model_list[rowid+1]
            local_data = _get_pips_data(node)
            linkIeq = local_data.linkIeq
            linkJeq = local_data.linkJeq
            linkVeq = local_data.linkVeq
            linkIineq = local_data.linkIineq
            linkJineq = local_data.linkJineq
            linkVineq = local_data.linkVineq

            if mode == :Structure
                return(length(linkVeq), length(linkVineq))
            else
                eqmat = sparse(linkIeq, linkJeq, linkVeq, nlinkeq, local_data.n)
                ineqmat = sparse(linkIineq, linkJineq, linkVineq, nlinkineq, local_data.n)
                if length(eqmat.nzval) > 0
                    array_copy(eqmat.rowval,e_rowidx)
                    array_copy(eqmat.colptr,e_colptr)
                    original_copy(eqmat.nzval,e_values)
                end
                if length(ineqmat.nzval) > 0
                    array_copy(ineqmat.rowval,i_rowidx)
                    array_copy(ineqmat.colptr,i_colptr)
                    original_copy(ineqmat.nzval, i_values)
                end
            end
        end
        return Int32(1)
    end

    function str_eval_h(rowid,colid,x0,x1,obj_factor,lambda,mode,rowidx,colptr,values)
        #println("eval_h")
        node = model_list[colid+1]
        local_data = _get_pips_data(node)
        if mode == :Structure
            if rowid == colid
                node_Hrows  = node.ext[:Hrows]
                node_Hcols = node.ext[:Hcols]
                node_val = ones(Float64,length(node_Hrows))
                mat = sparseKeepZero(node_Hrows,node_Hcols,node_val, local_data.n, local_data.n)
                return length(mat.rowval)
            elseif colid == 0
                node_Hrows  = node.ext[:Hrows]
                node_Hcols = node.ext[:Hcols]
                node_Hrows,node_Hcols = exchange(node_Hrows,node_Hcols)
                node_val = ones(Float64,length(node_Hrows))
                mat = sparseKeepZero(node_Hrows,node_Hcols,node_val, local_data.n, local_data.n)
                return length(mat.rowval)
            else
                return 0
            end
        else #fill in values
            if rowid == colid
                node_Hmap = node.ext[:Hmap]
                node_Hrows  = node.ext[:Hrows]
                node_Hcols = node.ext[:Hcols]
                if colid ==  0
                    local_x = x0
                else
                    local_x = x1
                end
                local_unsym_values = Array{Float64}(undef,local_data.local_unsym_hessnnz)
                node_val = ones(Float64,length(node_Hrows))
                #local_scl = (node.objSense == :Min) ? 1.0 : -1.0
                local_scl = (JuMP.objective_sense(node) == MOI.MAX_SENSE) ? -1.0 : 1.0
                local_m_eq = length(local_data.eq_idx)
                local_m_ineq = length(local_data.ineq_idx)
                local_lambda = zeros(Float64, local_data.local_m)
                for i in 1:local_m_eq
                    local_lambda[local_data.eq_idx[i]] = lambda[i]
                end
                for i in 1:local_m_ineq
                    local_lambda[local_data.ineq_idx[i]] = lambda[i+local_m_eq]
                end

                #eval_hesslag(local_data.d, local_unsym_values, local_x, obj_factor*local_scl, local_lambda)
                pips_eval_hessian_lagrangian(local_data.d, local_unsym_values, local_x, obj_factor*local_scl, local_lambda)
                local_sym_index=1
                for i in 1:local_data.local_unsym_hessnnz
                    if node_Hmap[i]
                        node_val[local_sym_index] = local_unsym_values[i]
                        local_sym_index +=1
                    end
                end
                node_Hrows,node_Hcols = exchange(node_Hrows,node_Hcols)
                mat = sparseKeepZero(node_Hrows,node_Hcols,node_val, local_data.n, local_data.n)
                array_copy(mat.rowval,rowidx)
                array_copy(mat.colptr,colptr)
                original_copy(mat.nzval,values)
            elseif colid ==	0
                node_Hrows  = node.ext[:Hrows]
                node_Hcols = node.ext[:Hcols]
                node_val = zeros(Float64,length(node_Hrows))
                node_Hrows,node_Hcols = exchange(node_Hrows,node_Hcols)
                mat = sparseKeepZero(node_Hrows,node_Hcols,node_val, local_data.n, local_data.n)
                array_copy(mat.rowval,rowidx)
                array_copy(mat.colptr,colptr)
                original_copy(mat.nzval,values)
            else
            end
        end
        return Int32(1)
    end

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    if (MPI.Comm_rank(comm) == 0)
        t1 = time()
    end

    root = 0
    rank = MPI.Comm_rank(comm)

    #Create PipsModel (The PIPS interface model) and pass all the functions it requires
    model = PipsModel(:Min,0, n_sub_models,str_init_x0, str_prob_info, str_eval_f, str_eval_g, str_eval_grad_f, str_eval_jac_g, str_eval_h,str_write_solution)

    ###################################
    #PIPS-NLP FUNCTIONS
    prob = createProblemStruct(comm, model, true)
    if rank == 0
        println("Created PIPS-NLP Problem Struct")
    end

    MPI.Barrier(comm)
    ret = solveProblemStruct(prob)
    ###################################

    if rank == 0
        println("Solved PIPS-NLP Problem Struct")
    end

    if rank == 0
        println("Timing Results:")
        println("init_x0  ",prob.t_jl_init_x0)
        println("str_init_x0  ", prob.t_jl_str_prob_info)
        println("eval_f  ", prob.t_jl_eval_f)
        println("eval_g0  ",prob.t_jl_eval_g)
        println("eval_grad_f  ", prob.t_jl_eval_grad_f)
        println("eval_jac  ", prob.t_jl_eval_jac_g)
        println("str_eval_jac  ", prob.t_jl_str_eval_jac_g)
        println("eval_h  ",  prob.t_jl_eval_h)
        println("str_eval_h ",  prob.t_jl_str_eval_h)
        println("eval_write_solution  ",  prob.t_jl_write_solution)
        println("PIPS-NLP time:   ",  time() - t1, " (s)")
    end

    #TODO.  Put solution onto optinodes
    for (idx,node) in enumerate(model_list)  #set solution values for each model
        local_data = _get_pips_data(node)
        if idx != 1 #all cores have the first stage
            coreid = zeros(Int, 1)
            sc = MPI.Reduce(local_data.coreid, MPI.SUM, root, comm)

            if rank == root
                coreid[1] = sc
            end

            MPI.Bcast!(coreid, length(coreid), root, comm)

            n = zeros(Int, 1)
            n[1] = local_data.n
            MPI.Bcast!(n,      length(n),      coreid[1], comm)

            if rank != coreid[1]
                local_data.n = n[1]
                local_data.x_sol = zeros(Float64,local_data.n)
            end
            MPI.Bcast!(local_data.x_sol, local_data.n, coreid[1], comm)
        end
        #node.ext[:colVal] = local_data.x_sol
        vars = MOI.get(node,MOI.ListOfVariableIndices())
        primals = OrderedDict(zip(vars,local_data.x_sol))
        Plasmo._set_primals(JuMP.backend(node),primals)

        #TODO set duals

        #TODO: Load results into a graph backend with termination status etc...
    end

    #TODO: Better return status.  Return MOI status
    status = :Unknown
    if ret == 0
        status = :Optimal
    elseif ret == 1
        status = :Not_Finished
    elseif	ret == 2
        status = :Maximum_Iterations_Exceeded
    elseif	ret == 3
        status = :Infeasible_Problem_Detected
    elseif ret == 4
        status = :Restoration_needed
    else
        status = :UnknownStatus
    end

    return status
end  #end pips_nlp_solve

end #end module
