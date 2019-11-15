module SampleSources

using Distributions, Random, SpecialFunctions
using Parameters
using Plots

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
    return -λ + k * log(λ) - lgamma(k+1)
end

function (model::NustarModel)(θ)
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

function parameter_transformation(model::NustarModel)
    as((Xs = as(Array, n_sources), Ys = as(Array, n_sources), Bs = as(Array, asℝ₊, n_sources)))
end

n_sources = 3

# Sample points uniformly in square
x_y_max = PSF_PIXEL_SIZE * PSF_IMAGE_LENGTH/2
println(x_y_max/PSF_PIXEL_SIZE)
sources_x = rand(Uniform(-x_y_max, x_y_max), n_sources)
sources_y = rand(Uniform(-x_y_max, x_y_max), n_sources)

# Sample brightness uniformly TODO: What unit/scale?
brightness_max = 15
sources_b = rand(Uniform(1, brightness_max), n_sources)

sources_truth = [(sources_x[i], sources_y[i], sources_b[i]) for i in 1:n_sources]

true_mean_image = TransformPSF.compose_mean_image(sources_truth)
observed_image = TransformPSF.sample_image(true_mean_image, 1)

# plt_mean = heatmap(true_mean_image)
# plt_sample = heatmap(observed_image)
#
# display(plt_mean)

model = NustarModel(observed_image)

# println("log density of ground truth")
# println(model(sources_truth))  # log density of true parameters



var_x, var_y = (PSF_PIXEL_SIZE * 1)^2, (PSF_PIXEL_SIZE * 1)^2
var_b = .25
covariance = [var_x 0.0 0.0; 0.0 var_y 0.0; 0.0 0.0 var_b]
init_x = rand(Uniform(-x_y_max, x_y_max), n_sources)
init_y = rand(Uniform(-x_y_max, x_y_max), n_sources)
init_b = rand(Uniform(1, brightness_max), n_sources)
θ_init = [(init_x[i], init_y[i], init_b[i]) for i in 1:n_sources]
println("SOURCES INIT")
for source in θ_init
    println((source[1]/PSF_PIXEL_SIZE, source[2]/PSF_PIXEL_SIZE))
end

posterior = RJMCMCSampler.nustar_rjmcmc(model, θ_init, 5, 2, covariance, 0)

println("DONE")
println(length(posterior))
end  # module
