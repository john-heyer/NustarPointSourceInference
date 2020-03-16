include("NustarConstants.jl")

const N_MIN = 100
const N_MAX = 400
const N_SOURCES_TRUTH = 1
# PROPOSAL_WIDTH.xy * PSF_PIXEL_SIZE * sqrt(1/n_sources) = σ_xy
# PROPOSAL_WIDTH.b * sqrt(1/n_sources) = σ_b  **NOTE: jumps made in exponential space
const PROPOSAL_WIDTH = (xy = .75, b = .025)
# SPLIT_PROPOSAL_WIDTH * sqrt(1/n_sources) = σ_split
const SPLIT_PROPOSAL_WIDTH = 200
const SAMPLES = 500000
const BURN_IN_STEPS = 400000
const JUMP_RATE = 0.1
const HYPER_RATE = .02
const SPLIT_RATE = 1/N_SOURCES_TRUTH
const TRUTH_DIST_B = P_SOURCE_B
# const TRUTH_DIST_B = Truncated(Normal(50, 100), 5, Inf)

const CHECK_CONVERGENCE = false
const N_KEEP = 10  # number of maps to produce if checking for convergence
