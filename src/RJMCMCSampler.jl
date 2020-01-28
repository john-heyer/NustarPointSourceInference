using Distributions, Random, SpecialFunctions
using DataStructures
using NPZ

include("NustarConstants.jl")
include("TransformPSF.jl")


mutable struct WindowTree
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    is_leaf::Bool
    split_x::Bool
    above::Union{WindowTree, Nothing}
    below::Union{WindowTree, Nothing}
    source::Union{Tuple{Float64, Float64, Float64}, Nothing}
end

@enum Move normal_move split_move merge_move birth_move death_move hyper_move tree_split_move tree_merge_move

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

function construct_tree(head)
    function build_tree!(root, sources)
        if length(sources) == 1
            root.is_leaf = true
            root.source = sources[1]
        else
            if root.split_x
                center_of_bx = sum([source[1]*exp(source[3]) for source in sources])/sum([exp(source[3]) for source in sources])
                above = WindowTree(center_of_bx, root.x_max, root.y_min, root.y_max, false, !root.split_x, nothing, nothing, nothing)
                below = WindowTree(root.x_min, center_of_bx, root.y_min, root.y_max, false, !root.split_x, nothing, nothing, nothing)
                root.above = above
                root.below = below
                sources_above = [source for source in sources if source[1] > center_of_bx]
                sources_below = [source for source in sources if source[1] < center_of_bx]
                build_tree!(above, sources_above)
                build_tree!(below, sources_below)
            else
                center_of_by = sum([source[2]*exp(source[3]) for source in sources])/sum([exp(source[3]) for source in sources])
                above = WindowTree(root.x_min, root.x_max, center_of_by, root.y_max, false, !root.split_x, nothing, nothing, nothing)
                below = WindowTree(root.x_min, root.x_max, root.y_min, center_of_by, false, !root.split_x, nothing, nothing, nothing)
                root.above = above
                root.below = below
                sources_above = [source for source in sources if source[2] > center_of_by]
                sources_below = [source for source in sources if source[2] < center_of_by]
                build_tree!(above, sources_above)
                build_tree!(below, sources_below)
            end
        end
    end
    root = WindowTree(XY_MIN, XY_MAX, XY_MIN, XY_MAX, false, true, nothing, nothing, nothing)
    build_tree!(root, head)
    return root
end

function intersection_dist(tree, mid, ray_origin, ray_vec)
    # Solve ray_origin + t * ray_vec = x/y to get intersection points
    t_x_min = (tree.x_min - ray_origin[1]) / ray_vec[1]
    t_x_max = (tree.x_max - ray_origin[1]) / ray_vec[1]
    t_y_min = (tree.y_min - ray_origin[2]) / ray_vec[2]
    t_y_max = (tree.y_max - ray_origin[2]) / ray_vec[2]
    potential = [t_x_min, t_x_max, t_y_min, t_y_max]
    t_intersect = minimum([t for t in potential if t > 0])
    x_intersect = ray_origin[1] + t_intersect * ray_vec[1]
    x_min, x_max = min(mid[1], x_intersect), max(mid[1], x_intersect)
    return Uniform(x_min, x_max)
end


function tree_split(head, split_rate, rng)
    root = construct_tree(head)
    new_sources = []
    to_remove = Set()
    p_split = 1
    det_J = 1
    splt = 0
    function split(tree)
        if !isnothing(tree)
            if tree.is_leaf && rand(rng, Uniform(0, 1)) < (split_rate / 2)
                source = tree.source
                dist_x, dist_y = Uniform(tree.x_min, tree.x_max), Uniform(source[2], tree.y_max)
                qx, qy = rand(rng, dist_x), rand(rng, dist_y)
                ray_origin = [qx, qy]
                mid = [source[1], source[2]]
                ray_vec = mid - ray_origin
                source_2_dist = intersection_dist(tree, mid, ray_origin, ray_vec)
                source_2_x = rand(rng, source_2_dist)
                t_x = (source_2_x - mid[1])/ray_vec[1]
                source_2_y = mid[2] + ray_vec[2] * t_x
                b = exp(source[3])
                x_diff = abs(qx - source_2_x)
                b_1 = b * (1 - abs(qx - source[1]) / abs(qx - source_2_x))
                b_2 = b * (1 - abs(source_2_x - source[1]) / abs(qx - source_2_x))
                source_1 = (qx, qy, log(b_2))
                source_2 = (source_2_x, source_2_y, log(b_2))
                # TODO: Add to tree and try more splits?
                push!(new_sources, source_1, source_2)
                push!(to_remove, source)
                p_split *= pdf(dist_x, qx) * pdf(dist_y, qy) * pdf(source_2_dist, source_2_x) * 1/2
                det_J *= abs(b / (qx - source[1]))
                splt += 1
                # println("split coordinate check: ")
                # println("source split: ")
                # println(source[1])
                # println(source[2])
                # println(exp(source[3]))
                # println("source new 1: ")
                # println(qx)
                # println(qy)
                # println(b_1)
                # println("source new 2: ")
                # println(source_2_x)
                # println(source_2_y)
                # println(b_2)
                # println("bounds: ")
                # println(tree.x_min)
                # println(tree.x_max)
                # println(tree.y_min)
                # println(tree.y_max)
                # println("x2 bounds: ")
                # println(source_2_dist.a)
                # println(source_2_dist.b)
                # println()
            else
                split(tree.above)
                split(tree.below)
            end
        end
    end
    split(root)
    sample_new = [source for source in head if !(source in to_remove)]
    sample_new = vcat(sample_new, new_sources)
    return sample_new, 1.0/p_split * det_J
end

function tree_merge(head, split_rate, rng)
    root = construct_tree(head)
    new_sources = []
    to_remove = Set()
    p_split = 1
    det_J = 1
    merged = 0
    p_source_1 = 1
    p_source_2 = 1
    function merge(tree)
        if !tree.is_leaf
            if tree.above.is_leaf && tree.below.is_leaf && rand(rng, Uniform(0, 1)) < split_rate
                source_above, source_below = tree.above.source, tree.below.source
                b_1, b_2 = exp(source_above[3]), exp(source_below[3])
                source_new_x = (source_above[1]*b_1 + source_below[1]*b_2)/(b_1 + b_2)
                source_new_y = (source_above[2]*b_1 + source_below[2]*b_2)/(b_1 + b_2)
                source_new_b = log(b_1 + b_2)
                source_2, source_1 = sort([source_above, source_below], by=f(x)=x[2])
                p_source_1 = pdf(Uniform(tree.x_min, tree.x_max), source_1[1]) * pdf(Uniform(source_new_y, tree.y_max), source_1[2])
                ray_origin = [source_1[1], source_1[2]]
                mid = [source_new_x, source_new_y]
                ray_vec = mid - ray_origin
                source_2_dist = intersection_dist(tree, mid, ray_origin, ray_vec)
                p_source_2 = pdf(source_2_dist, source_2[1])
                # TODO: Add to tree and try more splits?
                source_new = (source_new_x, source_new_y, source_new_b)
                push!(new_sources, source_new)
                push!(to_remove, source_above, source_below)
                p_split *= p_source_1 * p_source_2 * 1/2
                det_J *= abs(exp(source_new_b) / (source_1[1] - source_new_x))
                merged += 1
                # println()
                # println("merge coordinate check: ")
                # println("source split: ")
                # println(source_new_x)
                # println(source_new_y)
                # println(exp(source_new_b))
                # println("source new 1: ")
                # println(source_1[1])
                # println(source_1[2])
                # println(exp(source_1[3]))
                # println("source new 2: ")
                # println(source_2[1])
                # println(source_2[2])
                # println(exp(source_2[3]))
                # println("bounds: ")
                # println(tree.x_min)
                # println(tree.x_max)
                # println(tree.y_min)
                # println(tree.y_max)
                # println("x2 bounds: ")
                # println(source_2_dist.a)
                # println(source_2_dist.b)
                # println()

            else
                merge(tree.above)
                merge(tree.below)
            end
        end
    end
    merge(root)
    sample_new = [source for source in head if !(source in to_remove)]
    sample_new = vcat(sample_new, new_sources)
    return sample_new, p_split * 1.0/det_J
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


function jump_proposal(head, move_type, covariance, split_rate, rng)
    if move_type == birth_move
        sample_new, proposal_ratio = birth(head, rng)
    elseif move_type == death_move
        sample_new, proposal_ratio = death(head, rng)
    elseif move_type == split_move
        sample_new, proposal_ratio = split(head, covariance, rng)
    elseif move_type == merge_move
        sample_new, proposal_ratio = merge(head, covariance, rng)
    elseif move_type == tree_split_move
        sample_new, proposal_ratio = tree_split(head, split_rate, rng)
    elseif move_type == tree_merge_move
        sample_new, proposal_ratio = tree_merge(head, split_rate, rng)
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

function proposal(head, μ, move_type, covariance, split_rate, rng)
    if move_type == normal_move
        sample_new, proposal_rate = normal_proposal(head, covariance, rng)
        return sample_new, proposal_rate, μ
    elseif move_type == hyper_move
        return head, 1.0, hyper_proposal(μ, rng)
    else
        sample_new, proposal_rate = jump_proposal(head, move_type, covariance, split_rate, rng)
        return sample_new, proposal_rate, μ
    end
end

function get_move_type_tree(jump_rate, hyper_rate, rng)
    r = rand(rng, Uniform(0, 1))
    if r < hyper_rate
        return hyper_move
    elseif r < jump_rate + hyper_rate
        up = rand(rng, Uniform(0, 1)) < .5
        if up
            return tree_split_move
        else
            return tree_merge_move
        end
    else
        return normal_move
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
                return tree_split_move
            else
                return tree_merge_move
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
    out = OrderedDict()
    for move in instances(Move)
        out[move] = OrderedDict(
            "proposed" => 0,
            "accepted" => 0,
            "zero A moves" => 0
        )
    end
    return out
end

function record_move!(move_stats, move_type, A, accept)
    zero_A = (A == 0)
    move_stats[move_type]["proposed"] += 1
    move_stats[move_type]["accepted"] += accept
    move_stats[move_type]["zero A moves"] += zero_A
end


function nustar_rjmcmc(observed_image, θ_init, samples, burn_in_steps, covariance, jump_rate, μ_init, hyper_rate, split_rate, rng)
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
        sample_new, proposal_ratio, μ_new = proposal(head, μ, move_type, covariance, split_rate, rng)
        sample_lg_p = log_prior(sample_new, μ_new)
        # Don't recompute likelihood if it's a hyper_move
        if move_type == hyper_move
            sample_lg_l = current_lg_l
        elseif (move_type == tree_split_move || move_type == tree_merge_move) && (length(head) == length(sample_new))
            # if no splits or merges proposed
            continue
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
            # if move_type != normal_move
            #     println(i)
            #     println("================================")
            #     println("ZERO MOVE!")
            #     println("MOVE TYPE: ", move_type)
            #     println("LOG LIKELIHOOD: ", sample_lg_l)
            #     println("LOG PRIOR: ", sample_lg_p)
            #     println("max b: ", maximum([source[3] for source in sample_new]))
            #     println("LOG JOINT: ", sample_lg_joint)
            #     println("CURRENT LOG JOINT: ", current_lg_l + current_lg_p)
            #     println("PROPOSAL RATIO: ", proposal_ratio)
            #     println("================================")
            #     println()
            # end
        # end
        accept = rand(rng, Uniform(0, 1)) < A
        # if move_type == tree_merge_move
        #     println("merged: ", length(head) - length(sample_new))
        #     println("proposal ratio: ", proposal_ratio)
        # elseif move_type == tree_split_move
        #     println("split: ", length(sample_new) - length(head))
        #     println("proposal ratio: ", proposal_ratio)
        # end
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
