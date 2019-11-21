module RJMCMCSampler

using Distributions
using NPZ

include("NustarConstants.jl")
using .NustarConstants
include("TransformPSF.jl")

function jump_proposal(head, up, covariance)
    n_sources = convert(Int, length(head)/3)
    point_index = rand(1:n_sources)
    if up
        x, y, b = head[point_index], head[2*point_index], head[3*point_index]
        q = rand(MvNormal(covariance))
        x_new, y_new, b_new = x + q[1], y + q[2], b + q[3]
        sample_new = vcat(
            [head[i] for i in 1:point_index], [x_new],
            [head[i] for i in point_index+1: 2*point_index], [y_new],
            [head[i] for i in 2*point_index+1: 3*point_index], [b_new],
            [head[i] for i in 3*point_index+1:length(head)]
        )
        proposal_acceptance_ratio = 1/pdf(MvNormal(covariance), q)
        # TODO: Note that this acceptance ratio will be high!
    else
        # TODO: how to handle jumps down???
        sample_new = vcat(
            [head[i] for i in 1:point_index-1],
            [head[i] for i in point_index+1: 2*point_index-1],
            [head[i] for i in 2*point_index+1: 3*point_index-1],
            [head[i] for i in 3*point_index+1:length(head)]
        )
    end

end


function source_new(source, covariance)
    q = rand(MvNormal(covariance))
    source_out = (source[1] + q[1], source[2] + q[2], source[3] + q[3])
    return source_out
end


function proposal(head, covariance)
    n_sources = length(head)
    sample_new = [source_new(source, covariance) for source in head]
    return sample_new, 1
end


function nustar_rjmcmc(model, θ_init, samples, burn_in_steps, covariance, jump_rate)
    chain = []
    head = θ_init
    accepted_before = 0
    accepted_after = 0
    ratio_zero = 0
    ratio_inf = 0
    ratio_mid = 0
    for i in 1:(burn_in_steps + samples)
        if (i-1) % 50 == 0
            println("Iteration: ", i-1)
        end
        jump = rand(Uniform(0, 1)) < jump_rate
        if jump
            # println("JUMP")
            up = rand(Uniform(0, 1)) < .5
            sample_new, proposal_acceptance_ratio = jump_proposal(head, up, covariance)
        else
            # println("PROPOSAL")
            sample_new, proposal_acceptance_ratio = proposal(head, covariance)
        end
        A = exp(model(sample_new) - model(head)) * proposal_acceptance_ratio
        if A == 0
            ratio_zero += 1
            # declined_rate_image = TransformPSF.compose_mean_image(sample_new)
            # previous_rate_image = TransformPSF.compose_mean_image(head)
            # sampled_image = model.observed_image
            # npzwrite("zero_ratio.npz", Dict(
            #     "declined_img" => declined_rate_image,
            #     "previous_img" => previous_rate_image,
            #     "sampled_img" => sampled_image
            #     )
            # )
            # print("wrote zero rate to numpy")
        elseif A == Inf
            ratio_inf += 1
            # accepted_rate_image = TransformPSF.compose_mean_image(sample_new)
            # previous_rate_image = TransformPSF.compose_mean_image(head)
            # sampled_image = model.observed_image
            # npzwrite("inf_ratio.npz", Dict(
            #     "accepted_img" => accepted_rate_image,
            #     "previous_img" => previous_rate_image,
            #     "sampled_img" => sampled_image
            #     )
            # )
            # print("wrote inf rate to numpy")
        else
            ratio_mid += 1
        end
        accept = rand(Uniform(0, 1)) < A
        if accept
            head = sample_new
            if i > burn_in_steps
                accepted_after += 1
            else
                accepted_before += 1
            end
        end
        if i > burn_in_steps
            push!(chain, head)
        end

    end
    println("Proposals: ", burn_in_steps + samples)
    println("Accepted: ", accepted)
    println("Acceptance rate burn in: ", accepted_before/burn_in_steps)
    println("Acceptance rate after burn in: ", accepted_after/samples)
    println("Infinite A ratio rate: ", ratio_inf/(burn_in_steps + samples))
    println("Zero A ratio rate: ", ratio_zero/(burn_in_steps + samples))
    println("Finite A ratio rate: ", ratio_mid/(burn_in_steps + samples))
    return chain
end


end  # module
