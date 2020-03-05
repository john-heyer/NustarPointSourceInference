using Distributions

const NUSTAR_IMAGE_LENGTH = 64
const PSF_IMAGE_LENGTH = 1300
const IM_SCALE = 1300^2/64^2

# In radians/pixel
const NUSTAR_PIXEL_SIZE =  5.5450564776903175e-05
const PSF_PIXEL_SIZE = 2.9793119397393605e-06
const XY_MIN, XY_MAX, = -1.1 * PSF_IMAGE_LENGTH/2.0 * PSF_PIXEL_SIZE, 1.1 * PSF_IMAGE_LENGTH/2.0 * PSF_PIXEL_SIZE

const P_SOURCE_XY = Uniform(XY_MIN, XY_MAX)
# TODO: Update this later
const P_SOURCE_B = Uniform(exp(0), exp(7))
