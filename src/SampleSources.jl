module SampleSources

using Distributions, Random, SpecialFunctions
using Plots
using NPZ
using Random
using JSON


include("NustarConstants.jl")
using .NustarConstants

include("TransformPSF.jl")
include("RJMCMCSampler.jl")


function random_sources(n_sources)
    sources_x = rand(P_SOURCE_XY, n_sources)
    sources_y = rand(P_SOURCE_XY, n_sources)
    sources_b = rand(P_SOURCE_B, n_sources)
    return [(sources_x[i], sources_y[i], log(sources_b[i])) for i in 1:n_sources]
end

function sample_sources_main(
        n_sources, var_x, var_y, var_b, samples, burn_in_steps, jump_rate
    )
    sources_truth = random_sources(n_sources)
    mean_image =  TransformPSF.compose_mean_image(sources_truth)
    observed_image = TransformPSF.sample_image(mean_image, 1)

    covariance = [var_x 0.0 0.0; 0.0 var_y 0.0; 0.0 0.0 var_b]

    n_init_low, n_init_high = Int(floor(n_sources * .5)), Int(floor(n_sources * 1.5))
    n_init = rand(n_init_low:n_init_high)
    θ_init = random_sources(n_init)
    println("n sources: ", n_sources)
    println("n init: ", n_init)

    println("Sampling")
    posterior, stats = RJMCMCSampler.nustar_rjmcmc(
        observed_image, θ_init, samples, burn_in_steps, covariance, jump_rate)

    println("Done Sampling.  Writing ground truth and posterior to disk")
    ground_truth = [sources_truth[j][i] for i in 1:length(sources_truth[1]), j in 1:length(sources_truth)]
    sources_init = [θ_init[j][i] for i in 1:length(θ_init[1]), j in 1:length(θ_init)]
    posterior_sources = vcat(posterior...)
    posterior_array = [posterior_sources[i][j] for j in 1:length(posterior_sources[1]), i in 1:length(posterior_sources)]

    npzwrite("metropolis_data.npz", Dict("gt" => ground_truth, "posterior" => posterior_array, "init" => sources_init))
    open("acceptance_stats.json", "w") do f
        JSON.print(f, stats)
    end
    println("Done")

end


# Set up constants and configurations
n_sources = 10
var_x, var_y, var_b = (PSF_PIXEL_SIZE * 5)^2, (PSF_PIXEL_SIZE * 5)^2, .05^2
samples = 100
burn_in_steps = 0
jump_rate = 0.4

@time sample_sources_main(
    n_sources, var_x, var_y, var_b, samples, burn_in_steps, jump_rate
)

end  # module
