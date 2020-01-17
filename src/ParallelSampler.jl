using Distributed
addprocs()

@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("src/SampleSources.jl")
end

println("running")
sources_truth = random_sources(N_SOURCES_TRUTH)
println("SAMPLES: ", SAMPLES)

mean_image =  compose_mean_image(sources_truth)
observed_image = sample_image(mean_image, 1)

N_CHAINS = nworkers()

function do_mcmc(rng)
    μ_init = exp(rand(Uniform(log(N_MIN), log(N_MAX))))
    n_init = rand(Poisson(μ_init))
    θ_init = random_sources(n_init)
    posterior, stats = nustar_rjmcmc(
        observed_image, θ_init, Int(floor(SAMPLES/N_CHAINS)), BURN_IN_STEPS, covariance, JUMP_RATE, μ_init, HYPER_RATE, rng)
    return posterior, stats
end


rngs = [MersenneTwister() for _ in 1:N_CHAINS]
posterior, stats = collect(pmap(do_mcmc, rngs, on_error=identity))

write_sample_results(sources_truth, posterior, stats)
