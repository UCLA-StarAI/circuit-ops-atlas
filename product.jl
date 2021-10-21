using ProbabilisticCircuits
using StatsFuns

include("utils.jl")
include("integrate.jl")


function product_circuit(m::ProbCircuit, n::ProbCircuit; cache = PCPairCache(), 
                         m_log_prob::Bool = true, n_log_prob::Bool = true, compatible::Bool = false)
    postprocess_node(node::ProbCircuit) = begin
        if is⋀gate(node) && num_children(node) == 1
            children(node)[1]
        else
            node
        end
    end
    
    @inbounds get!(cache, Pair(m, n)) do
        if is⋁gate(m) && is⋁gate(n)
            node_cache = Vector{ProbCircuit}()
            prob_cache = Vector{Float64}()
            nchm = num_children(m)
            nchn = num_children(n)
            for cm_idx in 1 : nchm
                for cn_idx in 1 : nchn
                    cmn = product_circuit(children(m)[cm_idx], children(n)[cn_idx]; cache, m_log_prob, n_log_prob, compatible)
                    if cmn !== nothing
                        push!(node_cache, postprocess_node(cmn))
                        m_prob = m_log_prob ? exp(m.log_probs[cm_idx]) : m.log_probs[cm_idx]
                        n_prob = n_log_prob ? exp(n.log_probs[cn_idx]) : n.log_probs[cn_idx]
                        push!(prob_cache, m_prob * n_prob)
                    end
                end
            end
            
            if length(node_cache) > 0
                pc = summate(node_cache...)
                pc.log_probs .= prob_cache
                pc
            else
                nothing
            end
        elseif is⋀gate(m) && is⋀gate(n)
            node_cache = Vector{ProbCircuit}()
            nchm = num_children(m)
            nchn = num_children(n)
            flag = true
            if compatible
                @assert nchm == nchn
                for c_idx = 1 : nchm
                    if !flag
                        break
                    end
                    
                    pc = product_circuit(children(m)[c_idx], children(n)[c_idx]; 
                                         cache, m_log_prob, n_log_prob, compatible)
                    if pc !== nothing
                        push!(node_cache, pc)
                    else
                        flag = false
                    end
                end
            else
                for cm_idx = 1 : nchm
                    for cn_idx = 1 : nchn
                        pc = product_circuit(children(m)[cm_idx], children(n)[cn_idx]; 
                                             cache, m_log_prob, n_log_prob, compatible)
                        if pc !== nothing
                            push!(node_cache, pc)
                        end
                    end
                end
            end
            if flag
                multiply(node_cache...)
            else
                nothing
            end
        elseif is⋁gate(m) && is⋀gate(n)
            node_cache = Vector{ProbCircuit}()
            prob_cache = Vector{Float64}()
            nchm = num_children(m)
            for cm_idx = 1 : nchm
                cmn = product_circuit(children(m)[cm_idx], n; cache, m_log_prob, n_log_prob, compatible)
                if cmn !== nothing
                    push!(node_cache, postprocess_node(cmn))
                    m_prob = m_log_prob ? exp(m.log_probs[cm_idx]) : m.log_probs[cm_idx]
                    push!(prob_cache, m_prob)
                end
            end
            if length(node_cache) > 0
                pc = summate(node_cache...)
                pc.log_probs .= prob_cache
                pc
            else
                nothing
            end
        elseif is⋀gate(m) && is⋁gate(n)
            node_cache = Vector{ProbCircuit}()
            prob_cache = Vector{Float64}()
            nchn = num_children(n)
            for cn_idx = 1 : nchn
                cmn = product_circuit(m, children(n)[cn_idx]; cache, m_log_prob, n_log_prob, compatible)
                if cmn !== nothing
                    push!(node_cache, postprocess_node(cmn))
                    n_prob = n_log_prob ? exp(n.log_probs[cn_idx]) : n.log_probs[cn_idx]
                    push!(prob_cache, n_prob)
                end
            end
            if length(node_cache) > 0
                pc = summate(node_cache...)
                pc.log_probs .= prob_cache
                pc
            else
                nothing
            end
        elseif isliteralgate(m) && isliteralgate(n)
            if m.literal == n.literal
                PlainProbLiteralNode(m.literal)
            elseif m.literal == -n.literal
                nothing
            else
                multiply(
                    PlainProbLiteralNode(m.literal),
                    PlainProbLiteralNode(n.literal)
                )
            end
        elseif is⋁gate(m) && isliteralgate(n)
            node_cache = Vector{ProbCircuit}()
            prob_cache = Vector{Float64}()
            nchm = num_children(m)
            for cm_idx in 1 : nchm
                cmn = product_circuit(children(m)[cm_idx], n; cache, m_log_prob, n_log_prob, compatible)
                if cmn !== nothing
                    push!(node_cache, postprocess_node(cmn))
                    m_prob = m_log_prob ? exp(m.log_probs[cm_idx]) : m.log_probs[cm_idx]
                    push!(prob_cache, m_prob)
                end
            end
            if length(node_cache) > 0
                pc = summate(node_cache...)
                pc.log_probs .= prob_cache
                pc
            else
                nothing
            end
        elseif isliteralgate(m) && is⋁gate(n)
            node_cache = Vector{ProbCircuit}()
            prob_cache = Vector{Float64}()
            nchn = num_children(n)
            for cn_idx in 1 : nchn
                cmn = product_circuit(m, children(n)[cn_idx]; cache, m_log_prob, n_log_prob, compatible)
                if cmn !== nothing
                    push!(node_cache, postprocess_node(cmn))
                    n_prob = n_log_prob ? exp(n.log_probs[cn_idx]) : n.log_probs[cn_idx]
                    push!(prob_cache, n_prob)
                end
            end
            if length(node_cache) > 0
                pc = summate(node_cache...)
                pc.log_probs .= prob_cache
                pc
            else
                nothing
            end
        elseif is⋀gate(m) && isliteralgate(n)
            node_cache = Vector{ProbCircuit}()
            nchm = num_children(m)
            for cm_idx in 1 : nchm
                cmn = product_circuit(children(m)[cm_idx], n; cache, m_log_prob, n_log_prob, compatible)
                if cmn !== nothing
                    push!(node_cache, postprocess_node(cmn))
                end
            end
            if length(node_cache) > 0
                multiply(node_cache...)
            else
                nothing
            end
        elseif isliteralgate(m) && is⋀gate(n)
            node_cache = Vector{ProbCircuit}()
            nchn = num_children(n)
            for cn_idx in 1 : nchn
                cmn = product_circuit(m, children(n)[cn_idx]; cache, m_log_prob, n_log_prob, compatible)
                if cmn !== nothing
                    push!(node_cache, postprocess_node(cmn))
                end
            end
            if length(node_cache) > 0
                multiply(node_cache...)
            else
                nothing
            end
        else
            @assert false "unhandled situation: ($(typeof(m)), $(typeof(n)))."
        end
    end
end