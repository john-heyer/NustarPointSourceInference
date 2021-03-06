using NPZ
# using Images
# using Plots
# using ImageTransformations
# using CoordinateTransformations
using Distributions, Random

include("NustarConstants.jl")

psf = npzread("psf_9.npy")

# Negative values (-0.0)?
const PSF = [max(0, i) for i in psf]

function anneal!(psf, x_loc_pixels, y_loc_pixels)
    psf_half_length = NUSTAR_IMAGE_LENGTH/2/2
    # compute each pixels distance from x, y in pixels. x_loc_pixels, y_loc_pixels
    # measured in pixels from center while j, i is pixels from top left
    for j in 1:size(psf, 2)
        for i in 1:size(psf, 1)
            d = sqrt(
                ((psf_half_length - i) - y_loc_pixels)^2 +
                ((psf_half_length - j) - x_loc_pixels)^2
            )
            psf[i, j] += 1/(1 + d^2)
        end
    end
end


function psf_by_r(r)
    return PSF
end


function fn_apply_psf_transformation(x, y, b)
    x_loc_pixels, y_loc_pixels = -x/(NUSTAR_PIXEL_SIZE*2), -y/(NUSTAR_PIXEL_SIZE*2)
    psf_half_length = NUSTAR_IMAGE_LENGTH/2/2
    function power_law(row_col)
        i, j = row_col
        distance = sqrt(
            ((psf_half_length - i) - y_loc_pixels)^2 +
            ((psf_half_length - j) - x_loc_pixels)^2
        )
        return 1.0/(1 + distance^2)
    end
    psf = map(power_law, ((i, j) for i in 1:NUSTAR_IMAGE_LENGTH/2, j in 1:NUSTAR_IMAGE_LENGTH/2))
    return psf * 1.0/sum(psf) * exp(b) # normalize and scale by b
end


function apply_psf_transformation(x, y, brightness, new_shape=(64,64))
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
    psf = zeros(32, 32)
    anneal!(psf, -x/(NUSTAR_PIXEL_SIZE*2), -y/(NUSTAR_PIXEL_SIZE*2))
    # Resize by averaging and interpolating
    # psf = imresize(psf, new_shape)
    # println(1/sum(psf))
    # Normalize resized image: because we have averaged the pixels in downscaling,
    # we no longer have a true probability distribution, should rescale by (~1300^2/64^2)
    out = psf * 1.0/sum(psf) * exp(brightness)
    if minimum(out) < 0.0
        println("non positive value: ", minimum(out), brightness)
    end
    # println(minimum(out))
    # println(maximum(out))
    return out
end

function cartesian_to_polar(x, y)
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    return r, θ
end

function compose_mean_image(sources)
    return sum(
        apply_psf_transformation(source[1], source[2], source[3])
                for source in sources
    )
end


function sample_image(mean_image, t)
    return [rand(Poisson(convert(Float64, t*λ))) for λ in mean_image]
end

# @time tpsf = apply_psf_transformation(8 * NUSTAR_PIXEL_SIZE*2, 8* NUSTAR_PIXEL_SIZE*2, 0)
# plt_n = heatmap(tpsf)
# display(plt_n)
# # TODO: Fix resizing
# function what(tpsf)
#     max_v = -1
#     println(max_v)
#     max_i = (0, 0)
#     for j in 1:size(tpsf, 2)
#         for i in 1:size(tpsf, 1)
#             if tpsf[i, j] > max_v
#                 max_v = tpsf[i, j]
#                 max_i = (i, j)
#             end
#         end
#     end
#     println(max_v)
#     println(max_i)
# end
# what(tpsf)
# println(sum(tpsf))
# radius = tpsf[27:37, 27:37]
# println(sum(radius))
