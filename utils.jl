using ProbabilisticCircuits


PCPairCache = Dict{Pair{ProbCircuit, ProbCircuit}, Union{ProbCircuit, Nothing}}
PCCache = Dict{ProbCircuit, Union{ProbCircuit, Nothing}}