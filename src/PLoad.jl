using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("src/SampleSources.jl")
    using .SampleSources
end
