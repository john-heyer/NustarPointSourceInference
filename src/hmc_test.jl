using TransformVariables, LogDensityProblems, DynamicHMC,
    DynamicHMC.Diagnostics, Parameters, Statistics, Random,
    Distributions, Plots

const N_SOURCES_TRUTH = 10
const NUSTAR_IMAGE_LENGTH = 64
const PSF_IMAGE_LENGTH = 1300

# In radians/pixel
const NUSTAR_PIXEL_SIZE =  5.5450564776903175e-05
const PSF_PIXEL_SIZE = 2.9793119397393605e-06

const XY_MIN, XY_MAX, = -1.1 * PSF_IMAGE_LENGTH/2.0 * PSF_PIXEL_SIZE, 1.1 * PSF_IMAGE_LENGTH/2.0 * PSF_PIXEL_SIZE

const P_SOURCE_XY = Uniform(XY_MIN, XY_MAX)
const P_SOURCE_B = Uniform(exp(4), exp(7))

struct NustarProblem
    observed_image::Array{Int64,2}
end


function apply_psf_transformation(x, y, b)
    x_loc_pixels, y_loc_pixels = -x/NUSTAR_PIXEL_SIZE, -y/NUSTAR_PIXEL_SIZE
    psf_half_length = NUSTAR_IMAGE_LENGTH/2
    function power_law(row_col)
        i, j = row_col
        distance = sqrt(
            ((psf_half_length - i) - y_loc_pixels)^2 +
            ((psf_half_length - j) - x_loc_pixels)^2
        )
        return 1.0/(1 + .1distance)
    end
    psf = map(power_law, ((i, j) for i in 1:NUSTAR_IMAGE_LENGTH, j in 1:NUSTAR_IMAGE_LENGTH))
    return psf * 1.0/sum(psf) * exp(b) # normalize and scale by b
end


function compose_mean_image(sources)
    return sum(
        apply_psf_transformation(source[1], source[2], source[3])
        for source in sources
    )
end


function log_prior(θ)
    # all 3 are uniform distributions over finite range
    return sum(
        log(pdf(P_SOURCE_XY, source[1])) +
        log(pdf(P_SOURCE_XY, source[2])) +
        log(pdf(P_SOURCE_B, exp(source[3])))
        for source in θ
    )
end


function log_likelihood(θ, observed_image)
    model_rate_image = compose_mean_image(θ)
    lg_likelihood = sum(
        logpdf(Poisson(model_rate_image[i]), observed_image[i])
        for i in 1:length(observed_image)
    )
    return lg_likelihood
end


function (problem::NustarProblem)(θ)
    @unpack h = θ
    head = [(h[i], h[i+1], h[i+2]) for i in 1:3:length(h)]
    return log_likelihood(head, problem.observed_image) + log_prior(head)
end


function random_sources(n_sources)
    sources_x = rand(P_SOURCE_XY, n_sources)
    sources_y = rand(P_SOURCE_XY, n_sources)
    sources_b = rand(P_SOURCE_B, n_sources)
    return [(sources_x[i], sources_y[i], log(sources_b[i])) for i in 1:n_sources]
end


sources_truth = random_sources(N_SOURCES_TRUTH)

mean_image =  compose_mean_image(sources_truth)
observed_image = [rand(Poisson(λ)) for λ in mean_image]

display(heatmap(mean_image))

θ_init = random_sources(N_SOURCES_TRUTH)
q_init = vcat([[t[1], t[2], t[3]] for t in sources_truth]...)
q_transformed = (h = q_init,)

p = NustarProblem(observed_image)
println("logp init: ", p(q_transformed))

t = as((h = as(Array, 3 * N_SOURCES_TRUTH), ))
Pr = TransformedLogDensity(t, p)

grad_P = ADgradient(:ForwardDiff, Pr)

results = mcmc_with_warmup(Random.GLOBAL_RNG, grad_P, 1000; initialization = (q = q_init,))
# results = mcmc_with_warmup(
#     Random.GLOBAL_RNG, grad_P, 1000;
#     initialization = (ϵ = 0.1, q = q_init),
#     warmup_stages = default_warmup_stages(
#         ; local_optimization = nothing,
#         stepsize_search = nothing
#     )
# )
summarize_tree_statistics(results.tree_statistics)
