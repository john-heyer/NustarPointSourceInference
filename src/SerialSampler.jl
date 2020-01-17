include("SampleSources.jl")

sources_truth = random_sources(N_SOURCES_TRUTH)
println("SAMPLES: ", SAMPLES)

mean_image =  compose_mean_image(sources_truth)
observed_image = sample_image(mean_image, 1)

println("Sampling")
μ_init = exp(rand(Uniform(log(N_MIN), log(N_MAX))))
n_init = rand(Poisson(μ_init))
θ_init = random_sources(n_init)
posterior, stats = nustar_rjmcmc(
    observed_image, θ_init, SAMPLES, BURN_IN_STEPS, COVARIANCE,
    JUMP_RATE, μ_init, HYPER_RATE, MersenneTwister()
)
println("Done Sampling. Writing ground truth and posterior to disk")
write_sample_results(sources_truth, posterior, stats, θ_init)
println("Done")
