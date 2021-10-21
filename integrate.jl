using LogicCircuits, ProbabilisticCircuits


function evaluate_circuit(n::ProbCircuit, d::Vector{Bool}; log_prob::Bool = true)
    f_con(n)::Float64 = 0.0
    f_lit(n)::Float64 = begin
        if n.literal > Lit(0)
            d[Var(n.literal)] == true ? 1.0 : 0.0
        else
            d[Var(-n.literal)] == false ? 1.0 : 0.0
        end
    end
    f_a(n, cn)::Float64 = reduce(*, [cn...])
    f_o(n, cn)::Float64 = begin
        s = 0.0
        if log_prob
            for (logp, cp) in zip(n.log_probs, cn)
                s += cp * exp(logp)
            end
        else
            for (p, cp) in zip(n.log_probs, cn)
                s += cp * p
            end
        end
        s
    end
    
    foldup_aggregate(n, f_con, f_lit, f_a, f_o, Float64)
end

function integrate_circuit(n::ProbCircuit; log_prob::Bool = true)
    f_con(n)::Float64 = 1.0
    f_lit(n)::Float64 = 1.0
    f_a(n, cn)::Float64 = reduce(*, [cn...])
    f_o(n, cn)::Float64 = begin
        s = 0.0
        if log_prob
            for (logp, cp) in zip(n.log_probs, cn)
                s += cp * exp(logp)
            end
        else
            for (p, cp) in zip(n.log_probs, cn)
                s += cp * p
            end
        end
        s
    end
    
    foldup_aggregate(n, f_con, f_lit, f_a, f_o, Float64)
end