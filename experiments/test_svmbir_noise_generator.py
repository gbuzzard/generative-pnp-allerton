import numpy as np
import cv2
import matplotlib.pyplot as plt
import bm3d
import scipy.interpolate as si
import os
import svmbir
from gpnp_utils import precompute_svmbir_spectra, get_svmbir_noise_spectrum_, generate_svmbir_proximal_generator_noise


def main():

    # Define the schedule for sampling
    # Sample at each sigma for the given number of iterations, then save the
    # generated image and the denoised generated image
    id_tag = '_svmbir_v1'  # Output directory identifier - could automate this with the date
    num_total_iters = 40
    sigmas = 0.5

    # Phantom generations parameters
    num_rows_cols = 128     # assumes a square image
    num_views = 128
    tilt_angle = np.pi / 2  # Tilt range of +-90deg
    sigma_y = 1.0           # sinogram noise level
    gamma = 1.0             # prox map parameter

    # Generate the array of view angles
    angles = np.linspace(-tilt_angle, tilt_angle, num_views, endpoint=False)

    # Precompute required spectra
    psf_spectrum, awgn_spectrum = precompute_svmbir_spectra(num_rows_cols=num_rows_cols, angles=angles)

    # Get noise power spectrum for forward proximal generator
    noise_spectrum = get_svmbir_noise_spectrum_(psf_spectrum, awgn_spectrum, gamma=gamma, sigma_y=sigma_y)

    # Generate noise for forward proximal generator
    noise = generate_svmbir_proximal_generator_noise(psf_spectrum, awgn_spectrum, gamma=gamma, sigma_y=sigma_y)

    noise_rmse = np.sqrt(np.mean(noise**2))
    print('Noise RMSE={}'.format(noise_rmse))

    plt.ion()

    plt.figure(1)
    plt.imshow(psf_spectrum)
    plt.title('PSF Spectrum')
    plt.colorbar()
    plt.show()

    plt.figure(2)
    plt.imshow(awgn_spectrum)
    plt.title('AWGN Spectrum')
    plt.colorbar()
    plt.show()

    plt.figure(3)
    plt.imshow(np.sqrt(noise_spectrum))
    plt.title('SQRT Noise Spectrum')
    plt.colorbar()
    plt.show()

    plt.figure(4)
    plt.imshow(noise)
    plt.title('Noise')
    plt.colorbar()
    plt.show()

    input("Press return to quit:")


if __name__ == '__main__':

    main()