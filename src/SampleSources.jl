module SampleSources

using Distributions, Random, SpecialFunctions
using Distributed
using DataStructures
using NPZ
using Random
using JSON


include("NustarConstants.jl")
using .NustarConstants
include("SamplerConfigs.jl")
using .SamplerConfigs

include("TransformPSF.jl")
include("RJMCMCSampler.jl")


function random_sources(n_sources)
    sources_x = rand(P_SOURCE_XY, n_sources)
    sources_y = rand(P_SOURCE_XY, n_sources)
    sources_b = rand(P_SOURCE_B, n_sources)
    return [(sources_x[i], sources_y[i], log(sources_b[i])) for i in 1:n_sources]
end

function collect(results)
    combined_chain = vcat([result[1] for result in results]...)
    stats_dicts = [result[2] for result in results]
    move_stats = RJMCMCSampler.new_move_stats()
    proposals = 0
    accepted = 0
    n_sources_counts = Dict{Int, Int}()
    mus = []
    for stats in stats_dicts
        proposals += stats["proposals"]
        accepted += stats["accepted"]
        mv_stats = stats["stats by move type"]
        for (move, st) in mv_stats
            move_stats[move]["proposed"] += mv_stats[move]["proposed"]
            move_stats[move]["accepted"] += mv_stats[move]["accepted"]
            move_stats[move]["zero A moves"] += mv_stats[move]["zero A moves"]
        end
        mus = vcat(mus, stats["mus"])
        for (sources, counts) in stats["n_sources_counts"]
            n_sources_counts[sources] = get(n_sources_counts, sources, 0) + counts
        end
    end

    stats_out = OrderedDict(
        "proposals" => proposals,
        "accepted" => accepted,
        "acceptance rate" => accepted/proposals,
        "stats by move type" => move_stats,
        "n_sources_counts" => n_sources_counts,
        "mu_min" => minimum(mus),
        "mu_max" => maximum(mus),
        "mu_avg" => mean(mus)
    )
    return combined_chain, stats_out
end


function sample_sources_main(parallel=false)
    sources_truth = random_sources(N_SOURCES_TRUTH)
    println("SAMPLES: ", SAMPLES)

    mean_image =  TransformPSF.compose_mean_image(sources_truth)
    observed_image = TransformPSF.sample_image(mean_image, 1)

    covariance = [VAR_X 0.0 0.0; 0.0 VAR_Y 0.0; 0.0 0.0 VAR_B]
    N_CHAINS = nworkers()
    function do_mcmc(rng)
        μ_init = exp(rand(Uniform(log(N_MIN), log(N_MAX))))
        n_init = rand(Poisson(μ_init))
        θ_init = random_sources(n_init)
        posterior, stats = RJMCMCSampler.nustar_rjmcmc(
            observed_image, θ_init, Int(floor(SAMPLES/N_CHAINS)), BURN_IN_STEPS, covariance, JUMP_RATE, μ_init, HYPER_RATE, rng)
        return posterior, stats
    end

    println("Sampling")
    if parallel
        rngs = [MersenneTwister() for _ in 1:N_CHAINS]
        posterior, stats = collect(pmap(do_mcmc, rngs, on_error=identity))
    else
        μ_init = exp(rand(Uniform(log(N_MIN), log(N_MAX))))
        n_init = rand(Poisson(μ_init))
        θ_init = random_sources(n_init)
        posterior, stats = RJMCMCSampler.nustar_rjmcmc(
            observed_image, θ_init, SAMPLES, BURN_IN_STEPS, covariance, JUMP_RATE, μ_init, HYPER_RATE, MersenneTwister())
    end
    println("Done Sampling. Writing ground truth and posterior to disk")
    ground_truth = [sources_truth[j][i] for i in 1:length(sources_truth[1]), j in 1:length(sources_truth)]
    posterior_sources = vcat(posterior...)
    posterior_array = [posterior_sources[i][j] for j in 1:length(posterior_sources[1]), i in 1:length(posterior_sources)]
    posterior_data =  Dict("gt" => ground_truth, "posterior" => posterior_array)
    if !parallel
        sources_init = [θ_init[j][i] for i in 1:length(θ_init[1]), j in 1:length(θ_init)]
        posterior_data["init"] = sources_init
    end
    npzwrite("posterior_data.npz", posterior_data) #, "init" => sources_init))
    open("acceptance_stats.json", "w") do f
        JSON.print(f, stats)
    end
    println("Done")
end

end  # module
