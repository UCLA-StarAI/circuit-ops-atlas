using LogicCircuits, ProbabilisticCircuits

include("support_circuit.jl")
include("utils.jl")


function log_circuit(n::ProbCircuit; cache = PCCache(), support_cache = PCCache(), log_prob::Bool = true)
    @inbounds get!(cache, n) do
        if is⋁gate(n)
            node_cache = Vector{ProbCircuit}()
            prob_cache = Vector{Float64}()
            for (c, logp) in zip(children(n), n.log_probs)
                if isliteralgate(c)
                    push!(node_cache, log_circuit(c; cache, support_cache, log_prob))
                    push!(prob_cache, 0.0)
                else
                    push!(node_cache, log_circuit(c; cache, support_cache, log_prob))
                    push!(prob_cache, 1.0)
                end
                push!(node_cache, support_circuit(c; cache = support_cache))
                push!(prob_cache, log_prob ? logp : log(logp))
            end
            pc = summate(node_cache...)
            pc.log_probs .= prob_cache # Store prob instead of log-probs
            pc
        elseif is⋀gate(n)
            supp_pcs = map(children(n)) do c
                support_circuit(c; cache = support_cache)
            end
            
            nch = num_children(n)
            node_cache = Vector{ProbCircuit}()
            prob_cache = Vector{Float64}()
            for c_idx = 1 : nch
                pcs = Vector{ProbCircuit}()
                append!(pcs, [supp_pcs[idx] for idx = 1 : nch if idx < c_idx])
                push!(pcs, log_circuit(children(n)[c_idx]; cache, support_cache, log_prob))
                append!(pcs, [supp_pcs[idx] for idx = 1 : nch if idx > c_idx])
                pc = multiply(pcs...)
                push!(node_cache, pc)
                if isliteralgate(children(n)[c_idx])
                    push!(prob_cache, 0.0)
                else
                    push!(prob_cache, 1.0)
                end
            end
            pc = summate(node_cache...)
            pc.log_probs .= prob_cache # Store prob instead of log-probs
            pc
        elseif isliteralgate(n)
            n
        end
    end
end