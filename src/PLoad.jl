using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("SampleSources.jl")
    using .SampleSources
end
