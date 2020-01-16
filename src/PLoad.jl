@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("/home/heyer/workspace/NustarPointSourceInference/src/SampleSources.jl")
    include("/home/heyer/workspace/NustarPointSourceInference/src/RJMCMCSampler.jl")
    using .SampleSources
    using .RJMCMCSampler
end
