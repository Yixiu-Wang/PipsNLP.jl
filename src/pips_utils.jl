#############################################
# Helpers
#############################################
function exchange(a,b)
	 temp = a
         a=b
         b=temp
	 return (a,b)
end

function sparseKeepZero(I::AbstractVector{Ti},
    J::AbstractVector{Ti},
    V::AbstractVector{Tv},
    nrow::Integer, ncol::Integer) where {Tv,Ti<:Integer}
    N = length(I)
    if N != length(J) || N != length(V)
        throw(ArgumentError("triplet I,J,V vectors must be the same length"))
    end
    if N == 0
        return spzeros(eltype(V), Ti, nrow, ncol)
    end

    # Work array
    Wj = Array{Ti}(undef,max(nrow,ncol)+1)
    # Allocate sparse matrix data structure
    # Count entries in each row
    Rnz = zeros(Ti, nrow+1)
    Rnz[1] = 1
    nz = 0
    for k=1:N
        iind = I[k]
        iind > 0 || throw(ArgumentError("all I index values must be > 0"))
        iind <= nrow || throw(ArgumentError("all I index values must be ≤ the number of rows"))
        Rnz[iind+1] += 1
        nz += 1
    end
    Rp = cumsum(Rnz)
    Ri = Array{Ti}(undef,nz)
    Rx = Array{Tv}(undef,nz)

    # Construct row form
    # place triplet (i,j,x) in column i of R
    # Use work array for temporary row pointers
    @simd for i=1:nrow; @inbounds Wj[i] = Rp[i]; end
    @inbounds for k=1:N
        iind = I[k]
        jind = J[k]
        jind > 0 || throw(ArgumentError("all J index values must be > 0"))
        jind <= ncol || throw(ArgumentError("all J index values must be ≤ the number of columns"))
        p = Wj[iind]
        Vk = V[k]
        Wj[iind] += 1
        Rx[p] = Vk
        Ri[p] = jind
    end

    # Reset work array for use in counting duplicates
    @simd for j=1:ncol; @inbounds Wj[j] = 0; end

    # Sum up duplicates and squeeze
    anz = 0
    @inbounds for i=1:nrow
        p1 = Rp[i]
        p2 = Rp[i+1] - 1
        pdest = p1
        for p = p1:p2
            j = Ri[p]
            pj = Wj[j]
            if pj >= p1
                Rx[pj] = Rx[pj] + Rx[p]
            else
                Wj[j] = pdest
                if pdest != p
                    Ri[pdest] = j
                    Rx[pdest] = Rx[p]
                end
                pdest += one(Ti)
            end
        end
        Rnz[i] = pdest - p1
        anz += (pdest - p1)
    end

    # Transpose from row format to get the CSC format
    RiT = Array{Ti}(undef,anz)
    RxT = Array{Tv}(undef,anz)

    # Reset work array to build the final colptr
    Wj[1] = 1
    @simd for i=2:(ncol+1); @inbounds Wj[i] = 0; end
    @inbounds for j = 1:nrow
        p1 = Rp[j]
        p2 = p1 + Rnz[j] - 1
        for p = p1:p2
            Wj[Ri[p]+1] += 1
        end
    end
    RpT = cumsum(Wj[1:(ncol+1)])

    # Transpose
    @simd for i=1:length(RpT); @inbounds Wj[i] = RpT[i]; end
    @inbounds for j = 1:nrow
        p1 = Rp[j]
        p2 = p1 + Rnz[j] - 1
        for p = p1:p2
            ind = Ri[p]
            q = Wj[ind]
            Wj[ind] += 1
            RiT[q] = j
            RxT[q] = Rx[p]
        end
    end

    return SparseMatrixCSC(nrow, ncol, RpT, RiT, RxT)
end

#Convert Julia indices to C indices
function convert_to_c_idx(indicies)
    for i in 1:length(indicies)
        indicies[i] = indicies[i] - 1
    end
end

#Grab the subnode in hierarchical linkconstraint link
function _get_subnode(first_stage,link)
	linked_nodes = getnodes(link)
	@assert length(linked_nodes) == 2
	sub_node = setdiff(getnodes(link),[first_stage])[1]
	return sub_node
end

#categorize link constraints into hierarchical and distributed
function _identify_linkconstraints(graph::OptiGraph)
    links_connect_eq = LinkConstraint[]
    links_connect_ineq = LinkConstraint[]
    links_eq = LinkConstraint[]
    links_ineq = LinkConstraint[]

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

    return links_connect_eq,links_connect_ineq,links_eq,links_ineq
end
