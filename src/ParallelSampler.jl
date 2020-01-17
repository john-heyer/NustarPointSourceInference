using Distributed
addprocs()

@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("/home/heyer/workspace/NustarPointSourceInference/src/SampleSources.jl")
    N_CHAINS = nworkers()
end

println("running")
sources_truth = random_sources(N_SOURCES_TRUTH)
println("SAMPLES: ", SAMPLES)

mean_image =  compose_mean_image(sources_truth)
observed_image = sample_image(mean_image, 1)

println("WORKERS: ", N_CHAINS)

rngs = [(observed_image, MersenneTwister()) for _ in 1:N_CHAINS]
posterior, stats = collect(pmap(do_mcmc, rngs, on_error=identity))

write_sample_results(sources_truth, posterior, stats)
