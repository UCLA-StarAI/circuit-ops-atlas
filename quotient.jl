using ProbabilisticCircuits
using StatsFuns

include("utils.jl")
include("product.jl")
include("real_power.jl")


function quotient_circuit(m::ProbCircuit, n::ProbCircuit; cache = PCPairCache())
    n = circuit_real_power(n, -1.0)
    product_circuit(m, n; m_log_prob = true, n_log_prob = true, compatible = true)
end