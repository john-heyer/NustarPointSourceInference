using Distributions, Random, SpecialFunctions
using DataStructures
using NPZ

include("NustarConstants.jl")
include("TransformPSF.jl")


@enum Move normal_move split_move merge_move birth_move death_move hyper_move

function poisson_log_prob(λ, k)
    if λ <= 0
        println("zero lambda")
        return -Inf
    end
    return -λ + k * log(λ) - loggamma(k+1)
end

function P(μ)
    if (μ < N_MIN) || (μ > N_MAX)
        return 0
    end
    return 1.0/μ * 1.0/(log(N_MAX) - log(N_MIN))
end

function log_prior(θ, μ)
    return sum(
        [
            log(pdf(P_SOURCE_XY, source[1])) +
            log(pdf(P_SOURCE_XY, source[2])) +
            log(pdf(P_SOURCE_B, exp(source[3])))
            for source in θ
        ]
    ) + log(P(μ)) + poisson_log_prob(μ, length(θ))  # P(μ) + P(N|μ)
end


function log_likelihood(θ, observed_image)
    model_rate_image = compose_mean_image(θ)
    # println("max rate count: ", maximum(model_rate_image))
    lg_likelihood = sum(
        [poisson_log_prob(model_rate_image[i], observed_image[i])
            for i in 1:length(observed_image)
        ]
    )
    return lg_likelihood
end


function split(head, covariance, rng)
    point_index = rand(rng, 1:length(head))
    α = rand(rng, Uniform(0,1))
    q_dist = MvNormal(covariance[1:2, 1:2])
    q = rand(rng, q_dist)
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


function merge(head, covariance, rng)
    distance_dist = distance_distribution(head)
    cumulative_distance_dist = cumulative_distribution(distance_dist)
    v = rand(rng, Uniform(0,1))
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

function birth(head, rng)
    source = (rand(rng, P_SOURCE_XY), rand(rng, P_SOURCE_XY), log(rand(rng, P_SOURCE_B)))
    sample_new = vcat([s for s in head], [source])
    p_source = pdf(P_SOURCE_XY, source[1]) * pdf(P_SOURCE_XY, source[2]) * pdf(P_SOURCE_B, exp(source[3]))
    return sample_new, 1.0/length(sample_new) * 1.0/p_source
end

function death(head, rng)
    point_index = rand(rng, 1:length(head))
    source = head[point_index]
    sample_new = [head[i] for i in 1:length(head) if i != point_index]
    p_source = pdf(P_SOURCE_XY, source[1]) * pdf(P_SOURCE_XY, source[2]) * pdf(P_SOURCE_B, exp(source[3]))
    return sample_new, length(head) * p_source
end


function jump_proposal(head, move_type, covariance, rng)
    if move_type == birth_move
        sample_new, proposal_ratio = birth(head, rng)
    elseif move_type == death_move
        sample_new, proposal_ratio = death(head, rng)
    elseif move_type == split_move
        sample_new, proposal_ratio = split(head, covariance, rng)
    elseif move_type == merge_move
        sample_new, proposal_ratio = merge(head, covariance, rng)
    end
    return sample_new, proposal_ratio
end


function source_new(source, covariance, rng)::Tuple{Float64, Float64, Float64}
    q = rand(rng, MvNormal(covariance))
    source_out = (source[1] + q[1], source[2] + q[2], source[3] + q[3])
    return source_out
end


function normal_proposal(head, covariance, rng)::Tuple{Array{Tuple{Float64,Float64,Float64},1}, Float64}
    n_sources = length(head)
    sample_new = [source_new(source, covariance, rng) for source in head]
    return sample_new, 1.0
end

function hyper_proposal(μ, rng)
    return μ + rand(rng, Normal(0.0, 2))
end

function proposal(head, μ, move_type, covariance, rng)
    if move_type == normal_move
        sample_new, proposal_rate = normal_proposal(head, covariance, rng)
        return sample_new, proposal_rate, μ
    elseif move_type == hyper_move
        return head, 1.0, hyper_proposal(μ, rng)
    else
        sample_new, proposal_rate = jump_proposal(head, move_type, covariance, rng)
        return sample_new, proposal_rate, μ
    end
end

function get_move_type(jump_rate, hyper_rate, rng)
    r = rand(rng, Uniform(0, 1))
    if r < hyper_rate
        return hyper_move
    elseif r < jump_rate + hyper_rate
        split_merge = rand(rng, Uniform(0, 1)) < .5
        up = rand(rng, Uniform(0, 1)) < .5
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
        ),
        hyper_move => OrderedDict(
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

function nustar_rjmcmc(observed_image, θ_init, samples, burn_in_steps, covariance, jump_rate, μ_init, hyper_rate, rng)
    chain = Array{Tuple{Float64,Float64,Float64},1}[]
    head = θ_init
    μ = μ_init
    accepted = 0
    move_stats = new_move_stats()
    mus = [μ]
    current_lg_l = log_likelihood(head, observed_image)
    current_lg_p = log_prior(head, μ)
    accepted_recent = 0
    acceptance_rates = []
    for i in 1:(burn_in_steps + samples)
        if (i) % 1000 == 0
            println("Iteration: ", i)
            push!(acceptance_rates, accepted_recent/1000)
            accepted_recent = 0
        end
        move_type = get_move_type(jump_rate, hyper_rate, rng)
        sample_new, proposal_ratio, μ_new = proposal(head, μ, move_type, covariance, rng)
        sample_lg_p = log_prior(sample_new, μ_new)
        # Don't recompute likelihood if it's a hyper_move
        if move_type == hyper_move
            sample_lg_l = current_lg_l
        else
            sample_lg_l = log_likelihood(sample_new, observed_image)
        end
        sample_lg_joint = sample_lg_l + sample_lg_p
        A = exp(sample_lg_joint - (current_lg_l + current_lg_p)) * proposal_ratio
        # if A == 0
            # if move_type == normal_move
            #     if sample_lg_p > -Inf && (i > 200)
            #         println("normal zero")
            #         zero = [sample_new[j][i] for i in 1:length(sample_new[1]), j in 1:length(sample_new)]
            #         current = [head[j][i] for i in 1:length(head[1]), j in 1:length(head)]
            #         new_map = compose_mean_image(sample_new)
            #         old_map = compose_mean_image(head)
            #         npzwrite("zero_move.npz", Dict("new" => zero, "head" => current, "n_map" => new_map, "o_map" => old_map, "sample" => observed_image))
            #         println("done writing")
            #     end
            # end
            # if move_type == death_move
            #     println(i)
            #     println("================================")
            #     println("ZERO MOVE!")
            #     println("MOVE TYPE: ", move_type)
            #     println("LOG LIKELIHOOD: ", sample_lg_l)
            #     println("LOG PRIOR: ", sample_lg_p)
            #     println("LOG JOINT: ", sample_lg_joint)
            #     println("CURRENT LOG JOINT: ", current_lg_joint)
            #     println("PROPOSAL RATIO: ", proposal_ratio)
            #     println("================================")
            #     println()
            # end
        # end
        accept = rand(rng, Uniform(0, 1)) < A
        if accept
            head = sample_new
            μ = μ_new
            current_lg_l, current_lg_p = sample_lg_l, sample_lg_p
            accepted += 1
            accepted_recent += 1
        end
        if i > burn_in_steps
            push!(chain, head)
            push!(mus, μ)
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
        "acceptance rate" => accepted / (burn_in_steps + samples),
        "stats by move type" => move_stats,
        "n_sources_counts" => n_sources_counts,
        "mus" => mus,
        "acceptance_rates" => acceptance_rates
    )
    return chain, stats
end
