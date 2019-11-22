module SampleSources

using Distributions, Random, SpecialFunctions
using Parameters
using Plots
using NPZ
using Profile
using Random
using Serialization
using Traceur


include("NustarConstants.jl")
using .NustarConstants

include("TransformPSF.jl")
include("RJMCMCSampler.jl")

struct NustarModel{T}
    observed_image::T
end

function poisson_log_prob(λ, k)
    if λ == 0
        return -Inf
    end
    return -λ + k * log(λ) - loggamma(k+1)
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
    likelihood = sum(
        [poisson_log_prob(model_rate_image[i], model.observed_image[i])
            for i in 1:length(model.observed_image)
        ]
    )
    return likelihood  # TODO: currently no priors
end

function random_sources(x_y_min, x_y_max, lg_b_min, lg_b_max, n_sources)
    sources_x = rand(Uniform(x_y_min, x_y_max), n_sources)
    sources_y = rand(Uniform(x_y_min, x_y_max), n_sources)
    sources_b = rand(Uniform(lg_b_min, lg_b_max), n_sources)
    return [(sources_x[i], sources_y[i], sources_b[i]) for i in 1:n_sources]
end

function create_model(sources)
    mean_image = TransformPSF.compose_mean_image(sources)
    display(heatmap(mean_image))
    observed_image = TransformPSF.sample_image(mean_image, 1)
    return NustarModel(observed_image)
end

function sample_sources_main(
        n_sources, x_y_min, x_y_max, lg_b_min, lg_b_max,
        var_x, var_y, var_b, samples, burn_in_steps, jump_rate
    )
    sources_truth = random_sources(x_y_min, x_y_max, lg_b_min, lg_b_max, n_sources)
    model = create_model(sources_truth)

    covariance = [var_x 0.0 0.0; 0.0 var_y 0.0; 0.0 0.0 var_b]

    θ_init = random_sources(x_y_min, x_y_max, lg_b_min, lg_b_max, n_sources)

    println("Sampling")
    posterior = RJMCMCSampler.nustar_rjmcmc(
        model, θ_init, samples, burn_in_steps, covariance, jump_rate)

    println("Done Sampling.  Writing ground truth and posterior to disk")
    ground_truth = [sources_truth[j][i] for i in 1:length(sources_truth[1]), j in 1:length(sources_truth)]
    sources_init = [θ_init[j][i] for i in 1:length(θ_init[1]), j in 1:length(θ_init)]
    posterior_array = [
        posterior[x][j][i]
            for i in 1:length(posterior[1][1]),
                j in 1:length(posterior[1]),
                x in 1:length(posterior)
    ]

    npzwrite("metropolis_data.npz", Dict("gt" => ground_truth, "posterior" => posterior_array, "init" => sources_init))
    println("Done")
end


# Set up constants and configurations
n_sources = 5
X_Y_MAX = PSF_PIXEL_SIZE * PSF_IMAGE_LENGTH/2
lg_b_min, lg_b_max = 7.0, 8.0  # TODO: Unit/scale for brightness
var_x, var_y, var_b = (PSF_PIXEL_SIZE * 5)^2, (PSF_PIXEL_SIZE * 5)^2, .05^2
samples = 1000
burn_in_steps = 100
jump_rate = 0.0

@time sample_sources_main(
    n_sources, -X_Y_MAX, X_Y_MAX, lg_b_min, lg_b_max,
    var_x, var_y, var_b, samples, burn_in_steps, jump_rate
)

end  # module
