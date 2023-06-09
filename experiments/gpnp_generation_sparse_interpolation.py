import numpy as np
import matplotlib.pyplot as plt
import os
import errno
import scipy.interpolate as si
import gpnp_utils as gu


def main():
    """
    This is a script to run multiple trials of generative plug-and-play using pixel-wise
    subsampling of a specified image.  The output images plus mean and standard deviation
    images will be stored in a subdirectory (determined by parameters) of the 'results'
    directory.
    """

    # Set gpnp parameters
    num_total_iters = 100  # Number of steps in generative process
    sigma_max = 0.5
    sigma_min = 0.005
    beta = 0.25  # Controls the step size
    alpha = 1.3  # Higher value gives more regularized samples.
    generate_option = True  # If false, then find the PnP fixed point.

    # Set parameters for image subsampling
    mask_threshold = 0.10  # Probability of measuring a pixel
    image_path = 'data/butterfly.png'
    initialization = 'RBF'  # 'RBF' (default), 'random' or 'ground_truth'
    sigma_y = sigma_min  # Assumed uncertainty in measured pixel values

    # Define the number of trials and the random number seeds for the mask (which is the same
    # for all trials) and the trial_seed_start, which defines the seed for the first trial.
    num_trials = 10
    mask_seed = 0
    trial_seed_start = 0

    # Determine derived parameters
    id_tag = 'subsample_{}iter_{}pct'.format(num_total_iters, int(100*mask_threshold))  # Output directory
    mult_factor = (sigma_min / sigma_max) ** (1 / (num_total_iters - 1))
    sigmas = sigma_max * mult_factor ** np.arange(num_total_iters)
    image_path = os.path.abspath(image_path)
    save_directory = os.path.join('results', id_tag)
    os.makedirs(save_directory, exist_ok=True)

    # Read the image and convert to RGB float in [0,1]
    image = gu.read_image_(image_path)
    if image is None:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), image_path)

    # Set up the mask and masked image for the forward model - use the same mask for all trials
    # The measured pixels are when mask = 1
    mask = np.zeros(image.shape[0:2] + (1,))
    np.random.seed(mask_seed)
    mask[np.random.uniform(0, 1, image.shape[0:2]) < mask_threshold] = 1
    y = mask * image

    all_images = []
    plt.ion()

    file_name = 'original_image_{}.png'.format(mask_seed)
    gu.gpnp_plot_and_save(0, image, 'Original image', save_directory, file_name)
    file_name = 'sampled_points_{}.png'.format(mask_seed)
    gu.gpnp_plot_and_save(1, mask, 'Sampled points', save_directory, file_name)

    # Run the sampling algorithm over multiple trials
    # One generative sample per seed for reproducibility
    for seed in range(num_trials):
        # Set new seed for each trial
        cur_seed = seed + trial_seed_start
        np.random.seed(cur_seed)

        if initialization == 'ground_truth':
            x = image
        elif initialization == 'random':
            initial_noise = np.random.standard_normal(image.shape)
            x = 0.5 + sigmas[0] * initial_noise
        else:  # (initialization == 'RBF')
            # Do an RBF interpolation
            measured_points = np.array(np.where(np.squeeze(mask) > 0.5)).T
            measured_values = [image[c[0], c[1]] for c in measured_points]
            interp = si.RBFInterpolator(measured_points, measured_values, neighbors=10)
            xgrid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
            xflat = xgrid.reshape(2, -1).T
            yflat = interp(xflat)
            yinterp = yflat.reshape(image.shape[0], image.shape[1], image.shape[2])
            yinterp = np.clip(yinterp, 0, 1)
            x = y + (1 - mask) * yinterp

        file_name = 'initial_image_{}.png'.format(mask_seed)
        gu.gpnp_plot_and_save(2, x, 'Initial image', save_directory, file_name)
        print("Initialized using " + initialization)

        # Outer loop over varying sigma
        for i, sigma in enumerate(sigmas):
            gamma = np.sqrt(beta) * sigma

            # Update at a fixed sigma
            x = gu.prior_generator(x, sigma, beta, alpha)
            x = gu.forward_generator_sparse_interpolation(x, gamma, y, sigma_y, mask, generate_option=generate_option)
            print('sigma={:.4f}, iteration={} of {}'.format(sigma, i, len(sigmas)), flush=True)

            # Show and save the result of the sample at specified iterations
            if i == (len(sigmas) - 1) or (i % int(num_total_iters / 3)) == 0:
                file_name = 'generated_image_{}_{}_sigma_{:.4f}.png'.format(mask_seed, cur_seed, sigma)
                title = 'Generated x, sigma={:.4f}'.format(sigma)
                gu.gpnp_plot_and_save(3, x, title, save_directory, file_name)

                # On the smallest sigma, save the image for mean and std
                if i == len(sigmas) - 1:
                    all_images += [x]

    all_images = np.array(all_images)
    mean_image = np.mean(all_images, axis=0)
    all_images_amplitude = np.sqrt(np.sum(all_images ** 2, axis=3))
    sd_image = np.std(all_images_amplitude, axis=0)
    file_name = 'generated_mean_{}.png'.format(mask_seed)
    gu.gpnp_plot_and_save(5, mean_image, 'Mean image', save_directory, file_name)

    file_name = 'generated_std_dev_{}.png'.format(mask_seed)
    gu.gpnp_plot_and_save(6, sd_image, 'Std dev image', save_directory, file_name)

    input("Press return to quit:")


if __name__ == '__main__':
    main()
