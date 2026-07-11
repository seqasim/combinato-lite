"""Spike waveform upsample / align / downsample (ported from Combinato)."""

from numpy import arange, zeros
from scipy.interpolate import make_interp_spline


def upsample(data, factor):
    num_m, num_vpe = data.shape
    up_num_vpe = (num_vpe - 1) * factor + 1
    axis = arange(0, up_num_vpe, factor)
    up_axis = arange(up_num_vpe)
    splines = make_interp_spline(axis, data.T)
    up_data = splines(up_axis)
    return up_data.T


def align(data, center, low, high):
    width = 5
    index_max = (
        data[:, center - width * low : center + width * high].argmax(1)
        + center
        - width * low
    )
    num_e, num_vpe = data.shape
    aligned_data = zeros((num_e, num_vpe - width * low - width * high))
    for i in range(num_e):
        aligned_data[i] = data[
            i,
            index_max[i] - center + width * low : index_max[i]
            - center
            + num_vpe
            - width * high,
        ]
    return aligned_data, center - width * low


def clean(data, center):
    index_max = data.argmax(1)
    return data[index_max == center], (index_max != center)


def downsample(data, old_center, skip, new_center=19, num_points=64):
    index = arange(num_points) * skip
    return data[:, index], num_points
