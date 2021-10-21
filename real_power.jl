using LogicCircuits, ProbabilisticCircuits

include("utils.jl")


function circuit_real_power(n::ProbCircuit, alpha::Float64; cache = PCCache())
    @inbounds get!(cache, n) do
        if is⋁gate(n)
            chs = [circuit_real_power(c, alpha; cache) for c in children(n)]
            pc = summate(chs)
            pc.log_probs .= n.log_probs .* alpha
            pc
        elseif is⋀gate(n)
            chs = [circuit_real_power(c, alpha; cache) for c in children(n)]
            multiply(chs)
        elseif isliteralgate(n)
            n
        else
            @assert false "unhandled situation: $(typeof(n))."
        end
    end
end