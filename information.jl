using LogicCircuits, ProbabilisticCircuits


const KLDCache = Dict{Tuple{ProbCircuit,ProbCircuit}, Float64}
const PRCache = Dict{Tuple{ProbCircuit, LogicCircuit}, Float64}


function entropy(n::ProbCircuit; log_prob::Bool = true)
    f_con(n)::Float64 = 0.0
    f_lit(n)::Float64 = 0.0
    f_a(n, cn)::Float64 = reduce(+, [cn...])
    f_o(n, cn)::Float64 = begin
        s = 0.0
        if log_prob
            for (logp, cent) in zip(n.log_probs, cn)
                s += - logp * exp(logp)
                s += exp(logp) * cent
            end
        else
            for (p, cent) in zip(n.log_probs, cn)
                s += - log(p) * p
                s += p * cent
            end
        end
        s
    end
    
    foldup_aggregate(n, f_con, f_lit, f_a, f_o, Float64)
end


"Calculate KL divergence calculation for pcs that are not necessarily identical"
function mykld(pc_node1::PlainSumNode, pc_node2::PlainSumNode,
               kld_cache::KLDCache=KLDCache(), pr_constraint_cache::PRCache=PRCache())
    # @assert !(pc_node1 isa PlainMulNode || pc_node2 isa PlainMulNode) "Prob⋀ not a valid pc node for KL-Divergence"

    # Check if both nodes are normalized for same vtree node
    # @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kld_cache) # Cache hit
        return kld_cache[(pc_node1, pc_node2)]
    elseif children(pc_node1)[1] isa PlainProbLiteralNode
        if pc_node2 isa PlainProbLiteralNode
            mykld(children(pc_node1)[1], pc_node2, kld_cache, pr_constraint_cache)
            mykld(children(pc_node1)[2], pc_node2, kld_cache, pr_constraint_cache)
            if literal(children(pc_node1)[1]) == literal(pc_node2)
                return get!(kld_cache, (pc_node1, pc_node2),
                    pc_node1.log_probs[1] * exp(pc_node1.log_probs[1])
                )
            else
                return get!(kld_cache, (pc_node1, pc_node2),
                    pc_node1.log_probs[2] * exp(pc_node1.log_probs[2])
                )
            end
        else
            # The below four lines actually assign zero, but still we need to
            # call it.
            mykld(children(pc_node1)[1], children(pc_node2)[1], kld_cache, pr_constraint_cache)
            mykld(children(pc_node1)[1], children(pc_node2)[2], kld_cache, pr_constraint_cache)
            mykld(children(pc_node1)[2], children(pc_node2)[1], kld_cache, pr_constraint_cache)
            mykld(children(pc_node1)[2], children(pc_node2)[2], kld_cache, pr_constraint_cache)
            # There are two possible matches
            if literal(children(pc_node1)[1]) == literal(children(pc_node2)[1])
                return get!(kld_cache, (pc_node1, pc_node2),
                    exp(pc_node1.log_probs[1]) * (pc_node1.log_probs[1] - pc_node2.log_probs[1]) +
                    exp(pc_node1.log_probs[2]) * (pc_node1.log_probs[2] - pc_node2.log_probs[2])
                )
            else
                return get!(kld_cache, (pc_node1, pc_node2),
                    exp(pc_node1.log_probs[1]) * (pc_node1.log_probs[1] - pc_node2.log_probs[2]) +
                    exp(pc_node1.log_probs[2]) * (pc_node1.log_probs[2] - pc_node2.log_probs[1])
                )
            end
        end
    else # the normal case
        kld = 0.0

        # loop through every combination of prim and sub
        for (prob⋀_node1, log_theta1) in zip(children(pc_node1), pc_node1.log_probs)
            for (prob⋀_node2, log_theta2) in zip(children(pc_node2), pc_node2.log_probs)
                p = children(prob⋀_node1)[1]
                s = children(prob⋀_node1)[2]

                r = children(prob⋀_node2)[1]
                t = children(prob⋀_node2)[2]

                theta1 = exp(log_theta1)

                p11 = pr_constraint(s, t, pr_constraint_cache)
                p12 = pr_constraint(p, r, pr_constraint_cache)

                p13 = theta1 * (log_theta1 - log_theta2)

                p21 = mykld(p, r, kld_cache, pr_constraint_cache)
                p31 = mykld(s, t, kld_cache, pr_constraint_cache)

                kld += p11 * p12 * p13 + theta1 * (p11 * p21 + p12 * p31)
            end
        end
        return get!(kld_cache, (pc_node1, pc_node2), kld)
    end
end
function mykld(pc_node1::PlainProbLiteralNode, pc_node2::PlainProbLiteralNode,
             kld_cache::KLDCache, pr_constraint_cache::PRCache)
    # Check if literals are over same variables in vtree
    # @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kld_cache) # Cache hit
        return kld_cache[pc_node1, pc_node2]
    else
        # In this case probability is 1, kl divergence is 0
        return get!(kld_cache, (pc_node1, pc_node2), 0.0)
    end
end
function mykld(pc_node1::PlainSumNode, pc_node2::PlainProbLiteralNode,
             kld_cache::KLDCache, pr_constraint_cache::PRCache)
    # @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kld_cache) # Cache hit
        return kld_cache[pc_node1, pc_node2]
    else
        mykld(children(pc_node1)[1], pc_node2, kld_cache, pr_constraint_cache)
        mykld(children(pc_node1)[2], pc_node2, kld_cache, pr_constraint_cache)
        if literal(children(pc_node1)[1]) == literal(pc_node2)
            return get!(kld_cache, (pc_node1, pc_node2),
                pc_node1.log_probs[1] * exp(pc_node1.log_probs[1])
            )
        else
            return get!(kld_cache, (pc_node1, pc_node2),
                pc_node1.log_probs[2] * exp(pc_node1.log_probs[2])
            )
        end
    end
end
function mykld(pc_node1::PlainProbLiteralNode, pc_node2::PlainSumNode,
             kld_cache::KLDCache, pr_constraint_cache::PRCache)
    # @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kld_cache) # Cache hit
        return kld_cache[pc_node1, pc_node2]
    else
        mykld(pc_node1, children(pc_node2)[1], kld_cache, pr_constraint_cache)
        mykld(pc_node1, children(pc_node2)[2], kld_cache, pr_constraint_cache)
        if literal(pc_node1) == literal(children(pc_node2)[1])
            return get!(kld_cache, (pc_node1, pc_node2),
                -pc_node2.log_probs[1]
            )
        else
            return get!(kld_cache, (pc_node1, pc_node2),
                -pc_node2.log_probs[2]
            )
        end
    end
end


"Calculate XENT divergence calculation for pcs that are not necessarily identical"
function myxent(pc_node1::PlainSumNode, pc_node2::PlainSumNode,
                kld_cache::KLDCache=KLDCache(), pr_constraint_cache::PRCache=PRCache())
    # @assert !(pc_node1 isa PlainMulNode || pc_node2 isa PlainMulNode) "Prob⋀ not a valid pc node for KL-Divergence"

    # Check if both nodes are normalized for same vtree node
    # @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kld_cache) # Cache hit
        return kld_cache[(pc_node1, pc_node2)]
    elseif children(pc_node1)[1] isa PlainProbLiteralNode
        if pc_node2 isa PlainProbLiteralNode
            myxent(children(pc_node1)[1], pc_node2, kld_cache, pr_constraint_cache)
            myxent(children(pc_node1)[2], pc_node2, kld_cache, pr_constraint_cache)
            if literal(children(pc_node1)[1]) == literal(pc_node2)
                return get!(kld_cache, (pc_node1, pc_node2),
                    pc_node1.log_probs[1] * exp(pc_node1.log_probs[1])
                )
            else
                return get!(kld_cache, (pc_node1, pc_node2),
                    pc_node1.log_probs[2] * exp(pc_node1.log_probs[2])
                )
            end
        else
            # The below four lines actually assign zero, but still we need to
            # call it.
            myxent(children(pc_node1)[1], children(pc_node2)[1], kld_cache, pr_constraint_cache)
            myxent(children(pc_node1)[1], children(pc_node2)[2], kld_cache, pr_constraint_cache)
            myxent(children(pc_node1)[2], children(pc_node2)[1], kld_cache, pr_constraint_cache)
            myxent(children(pc_node1)[2], children(pc_node2)[2], kld_cache, pr_constraint_cache)
            # There are two possible matches
            if literal(children(pc_node1)[1]) == literal(children(pc_node2)[1])
                return get!(kld_cache, (pc_node1, pc_node2),
                    exp(pc_node1.log_probs[1]) * (pc_node1.log_probs[1] - pc_node2.log_probs[1]) +
                    exp(pc_node1.log_probs[2]) * (pc_node1.log_probs[2] - pc_node2.log_probs[2])
                )
            else
                return get!(kld_cache, (pc_node1, pc_node2),
                    exp(pc_node1.log_probs[1]) * (pc_node1.log_probs[1] - pc_node2.log_probs[2]) +
                    exp(pc_node1.log_probs[2]) * (pc_node1.log_probs[2] - pc_node2.log_probs[1])
                )
            end
        end
    else # the normal case
        kld = 0.0

        # loop through every combination of prim and sub
        for (prob⋀_node1, log_theta1) in zip(children(pc_node1), pc_node1.log_probs)
            for (prob⋀_node2, log_theta2) in zip(children(pc_node2), pc_node2.log_probs)
                p = children(prob⋀_node1)[1]
                s = children(prob⋀_node1)[2]

                r = children(prob⋀_node2)[1]
                t = children(prob⋀_node2)[2]

                theta1 = exp(log_theta1)

                p11 = pr_constraint(s, t, pr_constraint_cache)
                p12 = pr_constraint(p, r, pr_constraint_cache)

                p13 = - theta1 * log_theta2

                p21 = myxent(p, r, kld_cache, pr_constraint_cache)
                p31 = myxent(s, t, kld_cache, pr_constraint_cache)

                kld += p11 * p12 * p13 + theta1 * (p11 * p21 + p12 * p31)
            end
        end
        return get!(kld_cache, (pc_node1, pc_node2), kld)
    end
end
function myxent(pc_node1::PlainProbLiteralNode, pc_node2::PlainProbLiteralNode,
             kld_cache::KLDCache, pr_constraint_cache::PRCache)
    # Check if literals are over same variables in vtree
    # @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kld_cache) # Cache hit
        return kld_cache[pc_node1, pc_node2]
    else
        # In this case probability is 1, kl divergence is 0
        return get!(kld_cache, (pc_node1, pc_node2), 0.0)
    end
end
function myxent(pc_node1::PlainSumNode, pc_node2::PlainProbLiteralNode,
             kld_cache::KLDCache, pr_constraint_cache::PRCache)
    # @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kld_cache) # Cache hit
        return kld_cache[pc_node1, pc_node2]
    else
        mykld(children(pc_node1)[1], pc_node2, kld_cache, pr_constraint_cache)
        mykld(children(pc_node1)[2], pc_node2, kld_cache, pr_constraint_cache)
        if literal(children(pc_node1)[1]) == literal(pc_node2)
            return get!(kld_cache, (pc_node1, pc_node2),
                pc_node1.log_probs[1] * exp(pc_node1.log_probs[1])
            )
        else
            return get!(kld_cache, (pc_node1, pc_node2),
                pc_node1.log_probs[2] * exp(pc_node1.log_probs[2])
            )
        end
    end
end
function myxent(pc_node1::PlainProbLiteralNode, pc_node2::PlainSumNode,
             kld_cache::KLDCache, pr_constraint_cache::PRCache)
    # @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kld_cache) # Cache hit
        return kld_cache[pc_node1, pc_node2]
    else
        myxent(pc_node1, children(pc_node2)[1], kld_cache, pr_constraint_cache)
        myxent(pc_node1, children(pc_node2)[2], kld_cache, pr_constraint_cache)
        if literal(pc_node1) == literal(children(pc_node2)[1])
            return get!(kld_cache, (pc_node1, pc_node2),
                -pc_node2.log_probs[1]
            )
        else
            return get!(kld_cache, (pc_node1, pc_node2),
                -pc_node2.log_probs[2]
            )
        end
    end
end


"""
Calculate the probability of the logic formula given by LC for the PC
"""
function pr_constraint(pc_node::PlainProbCircuit, lc_node, cache::PRCache=PRCache())::Float64

    # TODO require that both circuits have an equal vtree for safety. If they don't, then first convert them to have a vtree
    # @assert respects_vtree(lc_node, vtree(pc_node)) "Both circuits do not have an equal vtree"

    # Cache hit
    if (pc_node, lc_node) in keys(cache) 
        return cache[pc_node, lc_node]
    
    # Boundary cases
    elseif isliteralgate(pc_node)
        # Both are literals, just check whether they agrees with each other 
        if isliteralgate(lc_node)
            if literal(pc_node) == literal(lc_node)
                return get!(cache, (pc_node, lc_node), 1.0)
            else
                return get!(cache, (pc_node, lc_node), 0.0)
            end
        else
            pr_constraint(pc_node, children(lc_node)[1], cache)
            if length(children(lc_node)) > 1
                pr_constraint(pc_node, children(lc_node)[2], cache)
                return get!(cache, (pc_node, lc_node), 1.0)
            else
                return get!(cache, (pc_node, lc_node),
                    literal(children(lc_node)[1]) == literal(pc_node) ? 1.0 : 0.0)
            end
        end
    
    # The pc is true
    elseif isliteralgate(children(pc_node)[1])
        theta = exp(pc_node.log_probs[1])
        return get!(cache, (pc_node, lc_node),
            theta * pr_constraint(children(pc_node)[1], lc_node, cache) +
            (1.0 - theta) * pr_constraint(children(pc_node)[2], lc_node, cache))
    
    # Both pcs are not trivial
    else 
        prob = 0.0
        for (prob⋀_node, log_theta) in zip(children(pc_node), pc_node.log_probs)
            p = children(prob⋀_node)[1]
            s = children(prob⋀_node)[2]

            theta = exp(log_theta)
            for lc⋀_node in children(lc_node)
                r = children(lc⋀_node)[1]
                t = children(lc⋀_node)[2]
                prob += theta * pr_constraint(p, r, cache) * pr_constraint(s, t, cache)
            end
        end
        return get!(cache, (pc_node, lc_node), prob)
    end
end