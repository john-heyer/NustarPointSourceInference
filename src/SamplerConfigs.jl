module SamplerConfigs

include("NustarConstants.jl")
using .NustarConstants

export N_SOURCES_TRUTH, VAR_X, VAR_Y, VAR_B
export SAMPLES, BURN_IN_STEPS, JUMP_RATE, HYPER_RATE

# Set up configurations
N_SOURCES_TRUTH = 10
VAR_X, VAR_Y, VAR_B = (PSF_PIXEL_SIZE * 5)^2, (PSF_PIXEL_SIZE * 5)^2, .05^2
SAMPLES = 2
BURN_IN_STEPS = 0
JUMP_RATE = 0.1
HYPER_RATE = .02

end #  module
