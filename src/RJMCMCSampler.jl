module RJMCMCSampler

using Distributions, Random, SpecialFunctions
using DataStructures

include("NustarConstants.jl")
using .NustarConstants
include("TransformPSF.jl")

struct NustarModel{T}
    observed_image::T
end

@enum Move normal_move split_move merge_move birth_move death_move

function poisson_log_prob(λ, k)
    if λ == 0
        return -Inf
    end
    return -λ + k * log(λ) - loggamma(k+1)
end

function log_prior(θ)
    return sum(
        [
            log(pdf(P_SOURCE_XY, source[1])) +
            log(pdf(P_SOURCE_XY, source[2])) +
            log(pdf(P_SOURCE_B, exp(source[3])))
            for source in θ
        ]
    )
end


function (model::NustarModel)(θ::Array{Tuple{Float64,Float64,Float64},1})
    """
    Model callable to compute the conditional log likelihood of the sampled
    model parameters given the observed image.

    Params:
    - θ (model parameters): Array of Tuples of size 3 (source_x, source_y, source_b)

    Returns: likelihood
    """
    model_rate_image = TransformPSF.compose_mean_image(θ)
    log_likelihood = sum(
        [poisson_log_prob(model_rate_image[i], model.observed_image[i])
            for i in 1:length(model.observed_image)
        ]
    )
    return log_likelihood + log_prior(θ) + log(1.0/length(θ))
end


function split(head, covariance)
    point_index = rand(1:length(head))
    α = rand(Uniform(0,1))
    q_dist = MvNormal(covariance[1:2, 1:2])
    q = rand(q_dist)
    source = head[point_index]
    source_1 = (source[1] + q[1]/2, source[2] + q[2]/2, source[3] + log(α))
    source_2 = (source[1] - q[1]/2, source[2] - q[2]/2, source[3] + log(1-α))
    sample_new = [head[i] for i in 1:length(head) if i != point_index]
    push!(sample_new, source_1, source_2)
    # Compute probability of reverse move: p(merge(s1, s2))
    distance_dist = distance_distribution(sample_new)
    p_merge = distance_dist[length(distance_dist)][3]
    # Full proposal ratio: 1/p(q) * p(merge(s1, s2))/p(split(s)) * J
    return sample_new, 1.0/pdf(q_dist, q) * p_merge * length(head) * exp(source[3])
end

function distance(head, i, j)
    return ((head[i][1] - head[j][1])^2 + (head[i][2] - head[j][2])^2)^2
end

function distance_distribution(head)
    distances = [(i, j, 1.0/distance(head, i, j)) for i in 1:length(head) for j in i+1:length(head)]
    s = sum([d[3] for d in distances])
    return [(d[1], d[2], 1.0/s * d[3]) for d in distances]
end

function cumulative_distribution(dist)
    t = 0.0
    return [(i, j, t += p) for (i, j, p) in dist]
end


function merge(head, covariance)
    distance_dist = distance_distribution(head)
    cumulative_distance_dist = cumulative_distribution(distance_dist)
    v = rand(Uniform(0,1))
    point_index, point_merge_index, p_merge = -1, -1, 0
    for i in 1:length(cumulative_distance_dist)
        point_i, point_merge, p = cumulative_distance_dist[i]
        if v < p
            point_index, point_merge_index = point_i, point_merge
            p_merge = distance_dist[i][3]
            break
        end
    end
    q_dist = MvNormal(covariance[1:2, 1:2])
    q = [head[point_index][1] - head[point_merge_index][1], head[point_index][2] - head[point_merge_index][2]]
    # α = exp(head[point_index][3])/(exp(head[point_index][3]) + exp(head[point_merge_index][3]))
    x_merged = (head[point_index][1] + head[point_merge_index][1])/2
    y_merged = (head[point_index][2] + head[point_merge_index][2])/2
    b_merged = log(exp(head[point_index][3]) + exp(head[point_merge_index][3]))
    sample_new = [head[i] for i in 1:length(head) if (i != point_index) & (i != point_merge_index)]
    push!(sample_new, (x_merged, y_merged, b_merged))
    # Full proposal ratio: p(q) * p(split(s))/p(merge(s1, s2)) * 1/J
    r = pdf(q_dist, q) * 1.0/p_merge * 1.0/length(sample_new) * 1.0/exp(b_merged)
    return sample_new, pdf(q_dist, q) * 1.0/p_merge * 1.0/length(sample_new) * 1.0/exp(b_merged)
end

function birth(head)
    source = (rand(P_SOURCE_XY), rand(P_SOURCE_XY), log(rand(P_SOURCE_B)))
    sample_new = vcat([s for s in head], [source])
    p_source = pdf(P_SOURCE_XY, source[1]) * pdf(P_SOURCE_XY, source[2]) * pdf(P_SOURCE_B, exp(source[3]))
    return sample_new, 1.0/length(sample_new) * 1.0/p_source
end

function death(head)
    point_index = rand(1:length(head))
    source = head[point_index]
    sample_new = [head[i] for i in 1:length(head) if i != point_index]
    p_source = pdf(P_SOURCE_XY, source[1]) * pdf(P_SOURCE_XY, source[2]) * pdf(P_SOURCE_B, exp(source[3]))
    return sample_new, length(head) * p_source
end


function jump_proposal(head, move_type, covariance)
    if move_type == birth_move
        sample_new, proposal_acceptance_ratio = birth(head)
        # println("birth ratio: ", proposal_acceptance_ratio)
    elseif move_type == death_move
        sample_new, proposal_acceptance_ratio = death(head)
        # println("death ratio: ", proposal_acceptance_ratio)
    elseif move_type == split_move
        sample_new, proposal_acceptance_ratio = split(head, covariance)
        # println("split ratio: ", proposal_acceptance_ratio)
    elseif move_type == merge_move
        sample_new, proposal_acceptance_ratio = merge(head, covariance)
        # println("merge ratio: ", proposal_acceptance_ratio)
    end
    return sample_new, proposal_acceptance_ratio
end


function source_new(source, covariance)::Tuple{Float64, Float64, Float64}
    q = rand(MvNormal(covariance))
    source_out = (source[1] + q[1], source[2] + q[2], source[3] + q[3])
    return source_out
end


function normal_proposal(head, covariance)::Tuple{Array{Tuple{Float64,Float64,Float64},1}, Float64}
    n_sources = length(head)
    sample_new = [source_new(source, covariance) for source in head]
    return sample_new, 1.0
end

function proposal(head, move_type, covariance)
    if move_type == normal_move
        return normal_proposal(head, covariance)
    else
        return jump_proposal(head, move_type, covariance)
    end
end

function get_move_type(jump_rate)
    if rand(Uniform(0, 1)) < jump_rate
        split_merge = rand(Uniform(0, 1)) < .5
        up = rand(Uniform(0, 1)) < .5
        if split_merge
            if up
                return split_move
            else
                return merge_move
            end
        else
            if up
                return birth_move
            else
                return death_move
            end
        end
    else
        return normal_move
    end
end

function new_move_stats()
    return OrderedDict(
        normal_move => OrderedDict(
            "proposed" => 0,
            "accepted" => 0,
            "zero A moves" => 0
        ),
        split_move => OrderedDict(
            "proposed" => 0,
            "accepted" => 0,
            "zero A moves" => 0
        ),
        merge_move => OrderedDict(
            "proposed" => 0,
            "accepted" => 0,
            "zero A moves" => 0
        ),
        birth_move => OrderedDict(
            "proposed" => 0,
            "accepted" => 0,
            "zero A moves" => 0
        ),
        death_move => OrderedDict(
            "proposed" => 0,
            "accepted" => 0,
            "zero A moves" => 0
        )
    )
end

function record_move!(move_stats, move_type, A, accept)
    zero_A = (A == 0)
    move_stats[move_type]["proposed"] += 1
    move_stats[move_type]["accepted"] += accept
    move_stats[move_type]["zero A moves"] += zero_A
end

function nustar_rjmcmc(observed_image, θ_init, samples, burn_in_steps, covariance, jump_rate)
    model = NustarModel(observed_image)
    chain = Array{Tuple{Float64,Float64,Float64},1}[]
    head = θ_init
    accepted = 0
    move_stats = new_move_stats()
    for i in 1:(burn_in_steps + samples)
        if (i-1) % 50 == 0
            println("Iteration: ", i-1)
        end
        move_type = get_move_type(jump_rate)
        sample_new, proposal_acceptance_ratio = proposal(head, move_type, covariance)
	if proposal_acceptance_ratio == 0
	    println("proposal acceptance_ratio is zero ", move_type)
	end
        A = exp(model(sample_new) - model(head)) * proposal_acceptance_ratio
        accept = rand(Uniform(0, 1)) < A
        if accept
            head = sample_new
            accepted += 1
        end
        if i > burn_in_steps
            push!(chain, head)
        end
        record_move!(move_stats, move_type, A, accept)
    end
    n_sources_counts = Dict{Int, Int}()
    for sample in chain
        n_sources_counts[length(sample)] = get(n_sources_counts, length(sample), 0) + 1
    end
    stats = OrderedDict(
        "proposals" => burn_in_steps + samples,
        "accepted" => accepted,
        "acceptance rate" => accepted/(burn_in_steps + samples),
        "stats by move type" => move_stats,
        "n_sources_counts" => n_sources_counts,
    )
    return chain, stats
end


end  # module
