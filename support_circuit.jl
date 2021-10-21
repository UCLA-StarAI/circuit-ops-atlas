using LogicCircuits, ProbabilisticCircuits

include("utils.jl")


function support_circuit(n::ProbCircuit; cache = PCCache())
    @inbounds get!(cache, n) do
        if is⋁gate(n)
            chs = [support_circuit(c; cache) for c in children(n)]
            pc = summate(chs)
            pc.log_probs .= 1.0 # Store prob instead of log-probs
            pc
        elseif is⋀gate(n)
            chs = [support_circuit(c; cache) for c in children(n)]
            multiply(chs)
        elseif isliteralgate(n)
            n
        else
            @assert false "unhandled situation: $(typeof(n))."
        end
    end
end