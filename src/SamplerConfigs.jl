include("NustarConstants.jl")

N_SOURCES_TRUTH = 20
const VAR_X, VAR_Y, VAR_B = (PSF_PIXEL_SIZE * 5)^2, (PSF_PIXEL_SIZE * 5)^2, .05^2
const COVARIANCE = [VAR_X 0.0 0.0; 0.0 VAR_Y 0.0; 0.0 0.0 VAR_B]
const SAMPLES = 200000
const BURN_IN_STEPS = 0
const JUMP_RATE = 0.1
const HYPER_RATE = .02
const SPLIT_RATE = .1
