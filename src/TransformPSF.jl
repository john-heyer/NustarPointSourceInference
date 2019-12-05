module TransformPSF

using NPZ
using Images
using Plots
using ImageTransformations
using CoordinateTransformations
using Distributions, Random

include("NustarConstants.jl")
using .NustarConstants

psf = npzread("psf_9.npy")

# Negative values (-0.0)?
const PSF = [max(0, i) for i in psf]

function anneal!(psf, x_loc_pixels, y_loc_pixels)
    """Add to pixels by 1/d where d is distance from (x, y)"""
    psf_half_length = PSF_IMAGE_LENGTH/2
    # compute each pixels distance from x, y in pixels. x_loc_pixels, y_loc_pixels
    # measured in pixels from center while j, i is pixels from top left
    for j in 1:size(psf, 2)
        for i in 1:size(psf, 1)
            d = sqrt(
                ((psf_half_length - i) - y_loc_pixels)^2 +
                ((psf_half_length - j) - x_loc_pixels)^2
            )
            psf[i, j] += 1/sqrt((1+.1d^2))
        end
    end
end

function psf_by_r(r)
    return PSF
end

function apply_psf_transformation(x, y, brightness, new_shape=(64,64))
    """
    Params:
        - x: horizontal location of source relative to image center
        - y: vertical location of source relative to image center
        - brightness: source brightness
    Returns:
        64x64 image resulting from applying the psf to this source
    """
    # r, θ = cartesian_to_polar(x, y)
    # psf = psf_by_r(r)
    #
    # # first, rotate psf
    # psf = imrotate(psf, θ + pi/2, indices_spatial(psf), 0.0)
    #
    # # Note: TRANSLATION
    # # y shift first: negative = shift rows down (row 1 becomes row 2),
    # # which is displayed as shift up in plot because rows flipped
    # # x shift: negative = shift cols right (col 0 becomes col 1),
    # # which is displayed as right shift in plot; i.e., 0,0 in top left
    #
    # y_shift = -y/PSF_PIXEL_SIZE
    # x_shift = -x/PSF_PIXEL_SIZE
    #
    # trans = Translation(y_shift, x_shift)
    # # println(trans)
    #
    # # translation translates indices by inv(t), ie what's at index (0, 0)
    # # will now be at (1, 1) for a translation of (-1, -1).  The full image is
    # # returned, but by passing the indices of the original image as an argument,
    # # you get only the values at those locations (the shifted image)
    # psf = warp(psf, trans, indices_spatial(psf), 0.0)

    # Add to image according to power law so that image is differentiable everywhere
    psf = zeros(1300, 1300)
    anneal!(psf, -x/PSF_PIXEL_SIZE, -y/PSF_PIXEL_SIZE)

    # Resize by averaging and interpolating
    psf = imresize(psf, new_shape)
    # println(1/sum(psf))
    # Normalize resized image: because we have averaged the pixels in downscaling,
    # we no longer have a true probability distribution, should rescale by (~1300^2/64^2)
    return psf * IM_SCALE #* exp(brightness)
end

function cartesian_to_polar(x, y)
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    return r, θ
end

function compose_mean_image(sources)
    return sum(
        [
            apply_psf_transformation(source[1], source[2], source[3])
                for source in sources
        ]
    )
end


function sample_image(mean_image, t)
    return [rand(Poisson(convert(Float64, t*λ))) for λ in mean_image]
end

# tpsf = apply_psf_transformation(-302.623 * PSF_PIXEL_SIZE, 312.234* PSF_PIXEL_SIZE, 10)

# # TODO: Fix resizing
# tpsfs = [
#     apply_psf_transformation((r*PSF_PIXEL_SIZE, θ, 1))
#     for θ in range(0, 2*pi, length=16)
#     for r in range(0, length=20)
# ]
# s = [sum(t) for t in tpsfs]
# println(s[1:20])
# println("shape after")
# println(size(tpsf))
# println("sum after")
# println(sum(tpsf))
#
# println("displaying plot")
# plt_0 = heatmap(psf)
# plt_n = heatmap(tpsf)
#
# sample = sample_image(tpsf, 1)
# plt_s = heatmap(sample)

# display(plt_0)
# display(plt_n)
# display(plt_s)
# println()
end # module
