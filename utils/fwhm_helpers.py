import numpy as np
import scipy.optimize as opt


def logistic_function(x, alpha, beta, gamma, c):
    return alpha / (1. + np.exp((x - beta) / gamma)) + c


def calculate_fwhm(image, accuracy_factor=100, max_iterations=1000):
    np.seterr(all='ignore')
    number_of_pixels = image.shape[1]
    fitted_rows = []
    row_fwhms = []

    # Setting Parameter Estimates
    alpha_estimate = [1.]
    beta_estimate = [number_of_pixels / 2.]
    gamma_estimate = [1.]
    c_estimate = [0.5]
    estimated_parameters = alpha_estimate + beta_estimate + gamma_estimate + c_estimate

    # Setting Parameter Bounds
    lower_alpha_bound = [0.001]
    upper_alpha_bound = [np.inf]
    lower_beta_bound = [0.4 * number_of_pixels]
    upper_beta_bound = [0.6 * number_of_pixels]
    lower_gamma_bound = [0.0001]
    upper_gamma_bound = [25]
    lower_c_bound = [-np.inf]
    upper_c_bound = [np.inf]
    lower_bounds = lower_alpha_bound + lower_beta_bound + lower_gamma_bound + lower_c_bound
    upper_bounds = upper_alpha_bound + upper_beta_bound + upper_gamma_bound + upper_c_bound
    bounds = (np.array(lower_bounds), np.array(upper_bounds))

    for row_index, row in enumerate(image[:, :, 0]):
        (a, b, g, c), _ = opt.curve_fit(f=logistic_function,
                                        xdata=np.arange(number_of_pixels),
                                        ydata=row,
                                        p0=estimated_parameters,
                                        bounds=bounds,
                                        maxfev=max_iterations)
        extended_x = np.linspace(-number_of_pixels, 2 * number_of_pixels, num=number_of_pixels * accuracy_factor)
        fitted_row = logistic_function(extended_x, a, b, g, c)
        differences = -np.diff(fitted_row)
        half_max = np.max(differences) / 2.
        indices = np.where(np.diff(np.sign(differences - half_max)))[0]
        row_fwhm = (indices[-1] - indices[0]) / accuracy_factor

        fitted_rows.append(fitted_row)
        row_fwhms.append(row_fwhm)

    return np.mean(row_fwhms), np.mean(fitted_rows, axis=0)


def calculate_vertical_profile(image):
    columns = np.mean(image, axis=0)
    return columns


