import numpy as np
import matplotlib.pyplot as plt
import os
import gpnp_utils as gu


def main():
    """
    This is a script to run multiple trials of generative plug-and-play using sparse-view
    CT of a 2-dimensional phantom.  The output images plus mean and standard deviation
    images will be stored in a subdirectory (determined by parameters) of the 'results'
    directory.
    """
    # Define parameters
    num_total_iters = 100  # Number of steps in generative process
    sigma_max = 0.5
    sigma_min = 0.005
    beta = 0.25  # Controls the step size
    alpha = 1.5  # Higher value gives more regularized samples.
    generate_option = True  # If false, then find the PnP fixed point.

    # Set parameters for sparse-view CT
    # Phantom generations parameters
    num_rows_cols = 128     # assumes a square image
    num_views = 16
    tilt_angle = np.pi / 2  # Tilt range of +-90deg
    sigma_y = 0.25          # sinogram noise level
    svmbir_noise = True
    add_phantom_noise = False
    initialization = 'proximal_map'  # 'proximal_map', 'random' (default) or 'ground_truth'

    # Define the number of trials and the random number seeds for the mask (which is the same
    # for all trials) and the trial_seed_start, which defines the seed for the first trial.
    num_trials = 10
    mask_seed = 0
    trial_seed_start = 0

    # Determine derived parameters
    id_tag = 'svmbir_ct_{}iter_{}views'.format(num_total_iters, num_views)  # Output directory
    mult_factor = (sigma_min / sigma_max) ** (1 / (num_total_iters - 1))
    sigmas = sigma_max * mult_factor ** np.arange(num_total_iters)
    save_directory = os.path.join('results', id_tag)
    os.makedirs(save_directory, exist_ok=True)

    # Get phantom for svmbir
    sino, angles, sigma_y, phantom = gu.get_phantom_data(num_rows_cols=num_rows_cols, num_views=num_views, tilt_angle=tilt_angle, sigma_y=sigma_y, add_phantom_noise=add_phantom_noise)

    # Precompute spectra required for more accurate svmbir proximal generator
    psf_spectrum, awgn_spectrum = gu.precompute_svmbir_spectra(num_rows_cols=num_rows_cols, angles=angles)

    # Construct data dictionary
    data = dict([('sino', sino), ('angles', angles), ('sigma_y', sigma_y), ('psf_spectrum',psf_spectrum), ('awgn_spectrum',awgn_spectrum)])

    all_images = []
    plt.ion()

    file_name = 'phantom_{}.png'.format(mask_seed)
    gu.gpnp_plot_and_save(1, phantom, 'Phantom', save_directory, file_name)
    file_name = 'sino_{}.png'.format(mask_seed)
    gu.gpnp_plot_and_save(2, sino.T, 'Sinogram', save_directory, file_name)
    plt.pause(0.1)

    # Run the sampling algorithm over multiple trials
    # One generative sample per seed for reproducibility
    for seed in range(num_trials):
        # Set new seed for each trial
        cur_seed = seed + trial_seed_start
        np.random.seed(cur_seed)

        if initialization == 'proximal_map':
            x = gu.forward_generator_svmbir(np.zeros(phantom.shape)+0.5, gamma=sigmas[0], data=data, generate_option=False)
        elif initialization == 'ground_truth':
            x = phantom
        else:  # (initialization == 'random')
            initial_noise = np.random.standard_normal(phantom.shape)
            x = 0.5 + sigmas[0] * initial_noise

        # Outer loop over varying sigma
        for i, sigma in enumerate(sigmas):
            gamma = np.sqrt(beta) * sigma

            # Update at a fixed sigma
            x = gu.prior_generator(np.squeeze(x), sigma, beta, alpha, generate_option=generate_option)
            x = np.expand_dims(x, axis=0)   # Add back singleton axis for one tomographic slice.

            x = gu.forward_generator_svmbir(x, gamma, data, generate_option=generate_option, svmbir_noise=svmbir_noise)
            print('sigma={}, iteration={} of {}'.format(sigma, i, len(sigmas)))

            # Show and save the result of the sample at this sigma
            if i == (len(sigmas) - 1) or (i % int(num_total_iters / 3)) == 0:
                file_name = 'generated_image_{}_{}_sigma_{:.4f}.png'.format(mask_seed, cur_seed, sigma)
                title = 'Generated x, sigma={:.4f}'.format(sigma)
                gu.gpnp_plot_and_save(3, x, title, save_directory, file_name)
                plt.pause(0.1)

                # On the smallest sigma, save the image for mean and std
                if i == len(sigmas) - 1:
                    all_images += [np.squeeze(x)]

    all_images = np.array(all_images)
    mean_image = np.mean(all_images, axis=0)
    sd_image = np.std(all_images, axis=0)
    file_name = 'generated_mean_{}.png'.format(mask_seed)
    gu.gpnp_plot_and_save(5, mean_image, 'Mean image', save_directory, file_name)
    file_name = 'generated_std_dev_{}.png'.format(mask_seed)
    gu.gpnp_plot_and_save(6, sd_image, 'Std dev image', save_directory, file_name)

    input("Press return to quit:")


if __name__ == '__main__':

    main()
