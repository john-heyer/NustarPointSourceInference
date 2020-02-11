include("NustarConstants.jl")

const N_MIN = 100
const N_MAX = 400
const N_SOURCES_TRUTH = 200
const VAR_X, VAR_Y, VAR_B = (.5 * PSF_PIXEL_SIZE * 1.0/sqrt(N_SOURCES_TRUTH))^2, (.5 * PSF_PIXEL_SIZE * 1.0/sqrt(N_SOURCES_TRUTH))^2, (.025 * 1.0/sqrt(N_SOURCES_TRUTH))^2
const COVARIANCE = [VAR_X 0.0 0.0; 0.0 VAR_Y 0.0; 0.0 0.0 VAR_B]
const SAMPLES = 100000
const BURN_IN_STEPS = 10000
const JUMP_RATE = 0.1
const HYPER_RATE = .02
const SPLIT_RATE = 1/N_SOURCES_TRUTH
const TRUTH_DIST_B = P_SOURCE_B
# const TRUTH_DIST_B = Truncated(Normal(100, 200), 5, Inf)
