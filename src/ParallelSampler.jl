using Distributed
addprocs(8)

@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("/home/heyer/workspace/NustarPointSourceInference/src/SampleSources.jl")
    const N_CHAINS = nworkers()
end


function write_post_maps_sample(results, N)
    """
    Write numpy array of size (N, M, 64, 64)
    where M is the number of chains.
    """
    M = length(results)
    out = [compose_mean_image(results[m][1][end-n]) for n in 1:N, m in 1:M]
    out = [out[n,m][i,j] for n in 1:N, m in 1:M, i in 1:64, j in 1:64]
    println("shape out: ", size(out))
    npzwrite("post_maps" * string(BURN_IN_STEPS) * ".npz", out)
end


function collect(results)
    combined_chain = vcat([result[1] for result in results]...)
    stats_dicts = [result[2] for result in results]
    move_stats = new_move_stats()
    proposals = 0
    accepted = 0
    n_sources_counts = Dict{Int, Int}()
    mus = []
    acceptance_rates = []
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
        push!(acceptance_rates, stats["acceptance_rates"])
    end

    stats_out = OrderedDict(
        "proposals" => proposals,
        "accepted" => accepted,
        "acceptance rate" => accepted/proposals,
        "stats by move type" => move_stats,
        "n_sources_counts" => n_sources_counts,
        "mus" => mus,
        "acceptance_rates" => mean(acceptance_rates)
    )
    return combined_chain, stats_out
end

println("running")
sources_truth = random_sources(N_SOURCES_TRUTH, TRUTH_DIST_B)
println("SAMPLES: ", SAMPLES)

mean_image =  compose_mean_image(sources_truth)
observed_image = sample_image(mean_image, 1)

println("WORKERS: ", N_CHAINS)

rngs = [(observed_image, MersenneTwister()) for _ in 1:N_CHAINS]
@time chains = pmap(do_mcmc, rngs)#, on_error=identity)
println("finished sampling")

if CHECK_CONVERGENCE
    println("writing recent maps")
    write_post_maps_sample(chains, N_KEEP)
end

posterior, stats = collect(chains)
write_sample_results(sources_truth, posterior, stats)
println("DONE")
