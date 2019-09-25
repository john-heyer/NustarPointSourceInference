module NustarPointSourceInference

using NPZ
using Images
using Plots
using ImageTransformations
using CoordinateTransformations
println()
println("start")
psf = npzread("psf_9.npy")
println(size(psf))

nustar_pixel_size =  5.5450564776903175e-05
psf_pixel_size = 2.9793119397393605e-06

area_nustar_img = nustar_pixel_size^2 * 64^2
area_psf_img = psf_pixel_size^2 * size(psf)[1]^2

function psf_by_r(r)
    return psf
end

function apply_psf_transformation(r, θ, brightness, new_shape=(64,64))
    psf = psf_by_r(r)
    trans = Translation(r * sin(θ)/psf_pixel_size, -r * cos(θ)/psf_pixel_size)
    psf = warp(psf, trans, indices_spatial(psf), 0.0)
    psf = imrotate(psf, θ, indices_spatial(psf), 0.0)
    return psf * brightness
end

tpsf = apply_psf_transformation(300 * psf_pixel_size, pi/4, 5000)
println("after")
println(size(tpsf))
# println(area_nustar_img)
# println(area_psf_img)
#
# println(area_nustar_img/area_psf_img)
println("displaying plot")
plt = heatmap(psf)
pltt = heatmap(tpsf)
# plt = Plots.heatmap(rand(5,5))
display(pltt)
println()
end # module
