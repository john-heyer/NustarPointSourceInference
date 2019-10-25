module Sampler

using TransformVariables, LogDensityProblems, DynamicHMC, DynamicHMC.Diagnostics
using Distributions, Random, SpecialFunctions
using Parameters
using Plots

import ForwardDiff

include("NustarConstants.jl")
using .NustarConstants

include("TransformPSF.jl")

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
    - θ (model parameters): Array of 3*n_sources Real numbers, when unpacked
      is Arrays of [X_i, ..., X_n], [Y_i, ..., Y_n], [B_i, ..., B_n]

    Returns: likelihood
    """
    @unpack Xs, Ys, Bs = θ
    model_rate_image = TransformPSF.compose_mean_image(Xs, Ys, Bs)
    model_rate_image .+= 1
    likelihood = sum(
        [poisson_log_prob(model_rate_image[i], model.observed_image[i])
            for i in 1:length(model.observed_image)
        ]
    )
    return likelihood  # currently no priors
end

function parameter_transformation(model::NustarModel)
    as((Xs = as(Array, n_sources), Ys = as(Array, n_sources), Bs = as(Array, asℝ₊, n_sources)))
end

n_sources = 3

# Sample points uniformly in square
x_y_max = PSF_PIXEL_SIZE * PSF_IMAGE_LENGTH/2
sources_x = rand(Uniform(-x_y_max, x_y_max), n_sources)
sources_y = rand(Uniform(-x_y_max, x_y_max), n_sources)

# Sample brightness uniformly TODO: What scale?
brightness_max = 5000
sources_b = rand(Uniform(1, brightness_max), n_sources)

true_mean_image = TransformPSF.compose_mean_image(sources_x, sources_y, sources_b)
println(maximum(true_mean_image))
observed_image = TransformPSF.sample_image(true_mean_image, 1)
println(maximum(observed_image))

# plt_mean = heatmap(true_mean_image)
# plt_sample = heatmap(observed_image)
#
# display(plt_mean)

model = NustarModel(observed_image)
θs_truth = (Xs = sources_x, Ys = sources_y, Bs = sources_b)

println("log density of ground truth")
println(model(θs_truth))  # log density of true parameters

transformation = parameter_transformation(model)
transformed_model = TransformedLogDensity(transformation, model)
∇M = ADgradient(:ForwardDiff, transformed_model)
println("type of grad model")
println(typeof(∇M))
results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇M, 100)
posterior = transform.(transformation, results.chain)

posterior_Xs, posterior_Ys, posterior_Bs = (
    [params[1] for params in posterior], [params[2] for params in posterior],
    [params[3] for params in posterior]
)
println(size(posterior_Xs))


end  # module
