import numpy as np
import cv2
import matplotlib.pyplot as plt
import bm3d
import os
import svmbir


def get_phantom_data(num_rows_cols, num_views, tilt_angle, sigma_y, add_phantom_noise=True):
    """Generate phantom and sinogram data"""

    # Display parameters
    vmin = 1.0
    vmax = 1.1

    # Generate phantom with a single slice
    phantom = svmbir.phantom.gen_shepp_logan(num_rows_cols, num_rows_cols)
    phantom = np.expand_dims(phantom, axis=0)
    phantom = (np.clip(phantom, vmin, vmax) - vmin) / (vmax - vmin)

    # Generate the array of view angles
    angles = np.linspace(-tilt_angle, tilt_angle, num_views, endpoint=False)

    # Generate sinogram by projecting phantom
    sino = svmbir.project(phantom, angles, num_rows_cols)
    if add_phantom_noise:
        sino = sino + sigma_y * np.random.standard_normal(sino.shape)

    return sino, angles, sigma_y, phantom


def prior_generator(v, sigma, beta, denoising_factor, generate_option=True):
    """Prior model proximal generator"""

    # Denoise image
    v_denoise = bm3d.bm3d(v, denoising_factor * sigma)

    # Add generative noise if required
    if generate_option:  # GPnP option
        v_out = (1 - beta) * v + beta * v_denoise
        awgn = np.random.standard_normal(v.shape)
        v_out = v_out + np.sqrt(beta) * sigma * awgn
    else:  # PnP option
        v_out = v_denoise

    return v_out


def forward_generator_svmbir(v, gamma, data, generate_option=True, svmbir_noise=True):
    """Forward model proximal generator for svmbir reconstruction"""

    sino = data['sino']
    angles = data['angles']
    sigma_y = data['sigma_y']

    v_out = svmbir.recon(sino=sino, angles=angles, prox_image=v, sigma_y=sigma_y, sigma_p=gamma, max_resolutions=0, verbose=0)
    # Then the noise generation
    if generate_option:
        if svmbir_noise:
            psf_spectrum = data['psf_spectrum']
            awgn_spectrum = data['awgn_spectrum']
            noise = generate_svmbir_proximal_generator_noise(psf_spectrum, awgn_spectrum, gamma, sigma_y)
            v_out = v_out + noise
        else:
            awgn = np.random.standard_normal(v.shape)
            v_out = v_out + gamma * awgn

    return v_out


def forward_generator_sparse_interpolation(v, gamma, y, sigma_y, mask, generate_option=True):
    """Forward model proximal generator for sparse interpolation"""

    # First the proximal map
    gamma_sq = gamma * gamma
    sigma_y_sq = sigma_y * sigma_y
    prox_sigma = gamma_sq / (sigma_y_sq + gamma_sq + np.finfo(float).eps)

    v_out = v + mask * prox_sigma * (y - v)

    # Then the noise generation
    if generate_option:
        awgn = np.random.standard_normal(v.shape)
        measured_sigma = np.sqrt(sigma_y_sq * prox_sigma)
        unmeasured_sigma = gamma
        generator_sigma = measured_sigma * mask + unmeasured_sigma * (1 - mask)
        v_out = v_out + generator_sigma * awgn

    return v_out


def generate_svmbir_proximal_generator_noise(psf_spectrum, awgn_spectrum, gamma, sigma_y):
    """Generates noise for svmbir proximal generator"""

    # Get desired power spectrum
    power_spectrum = get_svmbir_noise_spectrum_(psf_spectrum, awgn_spectrum, gamma, sigma_y)

    agwn = np.random.standard_normal(power_spectrum.shape)
    spectral_noise = np.sqrt(power_spectrum) * np.fft.fft2(agwn)

    spatial_noise = np.real(np.fft.ifft2(spectral_noise))

    return spatial_noise


def get_svmbir_noise_spectrum_(psf_spectrum, awgn_spectrum, gamma, sigma_y):
    """Computes noise power spectrum for svmbir proximal generator"""

    # Generate impulse image
    power_spectrum = 1 / ((awgn_spectrum / (gamma ** 2)) + (psf_spectrum / (sigma_y ** 2)))

    return power_spectrum


def precompute_svmbir_spectra(num_rows_cols, angles):
    """Precompute tomographic spectra that are used for svmbir proximal generator"""

    # Compute average PSF power spectrum
    number_of_trials = 10
    psf_image = compute_psf_image_(num_rows_cols, angles)
    psf_spectrum = abs(np.fft.fft2(psf_image))
    for i in range(number_of_trials - 1):
        # Compute a PSF
        psf_image = compute_psf_image_(num_rows_cols, angles)
        psf_spectrum = psf_spectrum + abs(np.fft.fft2(psf_image))

    psf_spectrum = psf_spectrum / number_of_trials

    # Generate impulse image
    impulse = np.zeros((num_rows_cols, num_rows_cols))
    impulse[int(num_rows_cols / 2), int(num_rows_cols / 2)] = 1.0

    # Compute AWGN power spectrum
    awgn_spectrum = abs(np.fft.fft2(impulse))

    return psf_spectrum, awgn_spectrum


def compute_psf_image_(num_rows_cols, angles):
    """Compute psf for svmbir using forward/back projection at random location"""

    # Compute sampling range for location of impulse
    irange = int(num_rows_cols / 20) + 1
    icenter = int(num_rows_cols / 2)
    istart = icenter - irange
    istop = icenter + irange
    window_width = num_rows_cols - 3 * irange  # 3>2sqrt(2)
    window_width = 2 * int(window_width / 2)  # ensure that window width is even

    # Compute random location
    location = [np.random.randint(low=istart, high=istop), np.random.randint(low=istart, high=istop)]

    # Construct 2D Hamming window that is centered at location
    window1d = np.hamming(window_width + 1)  # Center of Hamming window at window_width/2
    pad_size = num_rows_cols - window_width - 1
    window1d = np.pad(window1d, (0, pad_size))  # Pad window to match image size
    window1d = np.roll(window1d, -int(window_width / 2))  # Center window at 0
    window = np.outer(np.roll(window1d, location[0]), np.roll(window1d, location[1]))

    # Generate impulse image
    impulse = np.zeros((num_rows_cols, num_rows_cols))
    impulse[location[0], location[1]] = 1.0

    # Forward and backproject impulse
    sino = svmbir.project(image=impulse, angles=angles, num_channels=num_rows_cols)
    psf_image = window * np.squeeze(svmbir.backproject(sino, angles))

    return psf_image


def gpnp_plot_and_save(fig_index, image, title, save_directory, save_name, vmin=None, vmax=None):
    """ Plot a figure with the given image and save both the figure and the image.
    Args:
        fig_index:
        image:
        title:
        save_directory:
        save_name:
    """

    # Calculate the figure size and a font size to be proportional to figure size
    image2d = np.squeeze(image)
    figsize = np.array(image2d.shape)[0:2][::-1] / 20.0
    figsize[figsize < 4] = 4
    fontsize = int(np.mean(figsize) * 64.0 / 30.0)
    plt.rcParams.update({'font.size': fontsize})

    # Display the image
    plt.figure(fig_index, figsize=figsize)
    plt.clf()  # clear figure before next iteration
    plt.imshow(image2d, interpolation=None, vmin=vmin, vmax=vmax)
    plt.title(title)
    if len(image2d.shape) == 2:  # Don't show a colorbar for color images
        plt.colorbar()

    # Save the figure and raw image
    file_name = os.path.join(save_directory, save_name)
    dot_loc = file_name.find('.')
    file_base_name = file_name[:dot_loc]
    file_ext = file_name[dot_loc:]
    fig_name = file_base_name + '_fig' + file_ext
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.pause(0.1)
    save_image_(file_name, image2d, vmax=vmax)


def save_image_(file_name, image, vmax=None):
    """Utility for saving images"""

    if vmax is None:
        vmax = np.amax(image)
    uint_image = np.clip(255 * image[..., ::-1] / vmax, 0, 255).astype(int)
    if len(image.shape) == 2:
        uint_image = np.fliplr(uint_image)
    cv2.imwrite(file_name, uint_image)


def read_image_(image_path):
    """ Utility to read images into float format
    Args:
        image_path:

    Returns: float image
    """
    image = cv2.imread(image_path)
    image = image[..., ::-1] / 255.0
    return image
