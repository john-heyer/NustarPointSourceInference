module RJMCMCSampler

using Distributions, Random, SpecialFunctions
using DataStructures

include("NustarConstants.jl")
using .NustarConstants
include("TransformPSF.jl")

const P_SOURCE_XY = Uniform(-1.5 * PSF_IMAGE_LENGTH/2.0 * PSF_PIXEL_SIZE, 1.5 * PSF_IMAGE_LENGTH/2.0 * PSF_PIXEL_SIZE)
# TODO: Update this later
const P_SOURCE_B = Uniform(exp(6), exp(9))

struct NustarModel{T}
    observed_image::T
end

function poisson_log_prob(λ, k)
    if λ == 0
        return -Inf
    end
    return -λ + k * log(λ) - loggamma(k+1)
end

function outside_window(θ)
    bound = PSF_IMAGE_LENGTH * PSF_PIXEL_SIZE * 1.5/2.0
    return any([(abs(source[1]) > bound) | (abs(source[2]) > bound) for source in θ])
end

function log_prior(θ)
    # TODO: prior on b
    if outside_window(θ)
        return -Inf
    end
    return 2 * length(θ) * log(pdf(P_SOURCE_XY, 0.0))
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
    return log_likelihood + log_prior(θ)
end


function split(head, point_index, covariance)
    α = rand(Uniform(0,1))
    q_dist = MvNormal(covariance[1:2, 1:2])
    q = rand(q_dist)
    source = head[point_index]
    source_1 = (source[1] + q[1]/2, source[2] + q[2]/2, source[3] + log(α))
    source_2 = (source[1] - q[1]/2, source[2] - q[2]/2, source[3] + log(1-α))
    sample_new = [head[i] for i in 1:length(head) if i != point_index]
    push!(sample_new, source_1, source_2)
    # Compute probability of reverse move: p(merge(s1, s2))
    # TODO: FIX THIS
    distance_dist = distance_distribution(sample_new)
    p_merge = distance_dist[length(distance_dist)][3]
    # Full proposal ratio: 1/p(q) * p(merge(s1, s2))/p(split(s)) * J
    return sample_new, 1.0/pdf(q_dist, q) * p_merge * length(head) #* exp(source[3])
end

function distance(head, i, j)
    if i == j
        println("fucked up")
        return Inf
    end
    ϵ = 2 * PSF_PIXEL_SIZE
    return sqrt((head[i][1] - head[j][1])^2 + (head[i][2] - head[j][2])^2) + ϵ
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
    for (point_i, point_merge, p) in cumulative_distance_dist
        if v < p
            point_index, point_merge_index, p_merge = point_i, point_merge, p
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
    # println("pdf q")
    # println(pdf(q_dist, q))
    return sample_new, pdf(q_dist, q) * 1.0/p_merge * 1.0/length(sample_new) #* 1.0/exp(b_merged)
end

function birth(head)
    source = (rand(P_SOURCE_XY), rand(P_SOURCE_XY), log(rand(P_SOURCE_B)))
    sample_new = vcat([s for s in head], [source])
    p_source = pdf(P_SOURCE_XY, source[1]) * pdf(P_SOURCE_XY, source[2]) #* pdf(P_SOURCE_B, source[3])
    return sample_new, 1.0/length(sample_new) * 1.0/p_source
end

function death(head)
    point_index = rand(1:length(head))
    source = head[point_index]
    sample_new = [head[i] for i in 1:length(head) if i != point_index]
    p_source = pdf(P_SOURCE_XY, source[1]) * pdf(P_SOURCE_XY, source[2]) #* pdf(P_SOURCE_B, source[3])
    return sample_new, length(head) * p_source
end


function jump_proposal(head, up, birth_death, covariance)
    if up
        if birth_death
            sample_new, proposal_acceptance_ratio = birth(head)
            # println("birth ratio: ", proposal_acceptance_ratio)
        else
            point_index = rand(1:length(head))
            sample_new, proposal_acceptance_ratio = split(head, point_index, covariance)
            # println("split ratio: ", proposal_acceptance_ratio)
        end
    else
        if birth_death
            sample_new, proposal_acceptance_ratio = death(head)
            # println("death ratio: ", proposal_acceptance_ratio)
        else
            sample_new, proposal_acceptance_ratio = merge(head, covariance)
            # println("merge ratio: ", proposal_acceptance_ratio)
        end
    end
    return sample_new, proposal_acceptance_ratio
end


function source_new(source, covariance)::Tuple{Float64, Float64, Float64}
    q = rand(MvNormal(covariance))
    source_out = (source[1] + q[1], source[2] + q[2], source[3] + q[3])
    return source_out
end


function proposal(head, covariance)::Array{Tuple{Float64,Float64,Float64},1}
    n_sources = length(head)
    sample_new = [source_new(source, covariance) for source in head]
    return sample_new
end


function nustar_rjmcmc(observed_image, θ_init, samples, burn_in_steps, covariance, jump_rate)
    model = NustarModel(observed_image)
    chain = Array{Tuple{Float64,Float64,Float64},1}[]
    head = θ_init
    accepted_before = 0
    accepted_after = 0
    ratio_inf = 0
    ratio_zero = 0
    ratio_finite = 0
    split_moves = 0
    merge_moves = 0
    split_accept = 0
    merge_accept = 0
    birth_moves = 0
    death_moves = 0
    birth_accept = 0
    death_accept = 0
    for i in 1:(burn_in_steps + samples)
        if (i-1) % 50 == 0
            println("Iteration: ", i-1)
        end
        jump = rand(Uniform(0, 1)) < jump_rate
        if jump
            up = rand(Uniform(0, 1)) < 0.5
            birth_death = rand(Uniform(0, 1)) < 0.5
            if up
                if birth_death
                    birth_moves += 1
                else
                    split_moves += 1
                end
            else
                if birth_death
                    death_moves += 1
                else
                    merge_moves += 1
                end
            end
            sample_new, proposal_acceptance_ratio = jump_proposal(head, up, birth_death, covariance)
        else
            sample_new, proposal_acceptance_ratio = proposal(head, covariance), 1.0
        end
        A = exp(model(sample_new) - model(head)) * proposal_acceptance_ratio
        if A == 0
            ratio_zero += 1
        elseif A == Inf
            ratio_inf += 1
        else
            ratio_finite += 1
        end
        accept = rand(Uniform(0, 1)) < A
        if accept
            head = sample_new
            if i > burn_in_steps
                accepted_after += 1
            else
                accepted_before += 1
            end
            if jump
                if up
                    if birth_death
                        birth_accept += 1
                    else
                        split_accept += 1
                    end
                else
                    if birth_death
                        death_accept += 1
                    else
                        merge_accept += 1
                    end
                end
            end
        end
        if i > burn_in_steps
            push!(chain, head)
        end

    end
    n_sources_counts = Dict{Int, Int}()
    for sample in chain
        n_sources_counts[length(sample)] = get(n_sources_counts, length(sample), 0) + 1
    end
    stats = OrderedDict(
        "proposals" => burn_in_steps + samples,
        "accepted" => accepted_before + accepted_after,
        "acceptance rate total" => (accepted_before + accepted_after)/(burn_in_steps + samples),
        "acceptance rate burn in" => accepted_before/burn_in_steps,
        "acceptance rate after burn in" => accepted_after/samples,
        "zero A_rate" => ratio_zero/(burn_in_steps + samples),
        "infinite A_rate" => ratio_inf/(burn_in_steps + samples),
        "finite A_rate" => ratio_finite/(burn_in_steps + samples),
        "split proposals" => split_moves,
        "split accepts" => split_accept,
        "split acceptance rate" => split_accept/split_moves,
        "merge proposals" => merge_moves,
        "merge accepts" => merge_accept,
        "merge acceptance rate" => merge_accept/merge_moves,
        "birth proposals" => birth_moves,
        "birth accepts" => birth_accept,
        "birth acceptance rate" => birth_accept/birth_moves,
        "death proposals" => death_moves,
        "death accepts" => death_accept,
        "death acceptance rate" => death_accept/death_moves,
        "n_sources_counts" => n_sources_counts,
    )
    return chain, stats
end


end  # module
