module RJMCMCSampler

using Distributions

include("NustarConstants.jl")
using .NustarConstants

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
    # println("q: ")
    # println(q/PSF_PIXEL_SIZE)
    # println("souce init: ")
    # println((source[1]/PSF_PIXEL_SIZE, source[2]/PSF_PIXEL_SIZE))
    source_out = (source[1] + q[1], source[2] + q[2], source[3] + q[3])
        # println("source out: ")
        # println((source_out[1]/PSF_PIXEL_SIZE, source_out[2]/PSF_PIXEL_SIZE))
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
    accepted = 0
    for i in 1:(burn_in_steps + samples)
        jump = rand(Uniform(0, 1)) < jump_rate
        if jump
            println("JUMP")
            up = rand(Uniform(0, 1)) < .5
            sample_new, proposal_acceptance_ratio = jump_proposal(head, up, covariance)
        else
            println("PROPOSAL")
            sample_new, proposal_acceptance_ratio = proposal(head, covariance)
        end
        d = model(sample_new) - model(head)
        A = exp(d) * proposal_acceptance_ratio
        println("d: ", d)
        println("Ratio: ", A)
        accept = rand(Uniform(0, 1)) < A
        if accept
            head = sample_new
            accepted += 1
        end
        if i > burn_in_steps
            push!(chain, head)
        end
    end
    println("Accepted: ")
    println(accepted)
    println("percent: ")
    println(accepted/(burn_in_steps + samples))
    return chain
end


end  # module
