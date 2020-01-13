module NustarConstants
using Distributions

export PSF_IMAGE_LENGTH
export NUSTAR_IMAGE_LENGTH
export NUSTAR_PIXEL_SIZE
export PSF_PIXEL_SIZE
export IM_SCALE
export P_SOURCE_XY
export P_SOURCE_B


const NUSTAR_IMAGE_LENGTH = 64
const PSF_IMAGE_LENGTH = 1300
const IM_SCALE = 1300^2/64^2

# In radians/pixel
const NUSTAR_PIXEL_SIZE =  5.5450564776903175e-05
const PSF_PIXEL_SIZE = 2.9793119397393605e-06

const P_SOURCE_XY = Uniform(-1.1 * PSF_IMAGE_LENGTH/2.0 * PSF_PIXEL_SIZE, 1.1 * PSF_IMAGE_LENGTH/2.0 * PSF_PIXEL_SIZE)
# TODO: Update this later
const P_SOURCE_B = Uniform(exp(6), exp(8))


end #  module
