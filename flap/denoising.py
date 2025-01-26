"""
Denoising tools for FLAP

@author: Donat M. Takacs (takacs.donat@ek.hun-ren.hu)
Centre for Energy Research
"""

import numpy as np

def remove_sharp_peaks_func(
    signal,
    time,
    max_width_samples,
    diff_limit,
    dt=1.0,
    return_peak_center_times=False,
    return_peak_width_samples=False,
    return_peak_amplitudes=False,
    remove_peaks=True,
    interpolation='linear',
):
    """Remove sharp peaks with only a few samples of duration from the data,
    e.g. due to neutron/gamma noise affecting APDCAM.

    Peaks are detected by looking for sharp rising and falling edges in the
    signal, within a window of maximum `max_width_samples` samples. Detected
    peaks can be removed from the signal by interpolation.

    Parameters
    ----------
    signal : np.ndarray
        The signal to process.
    time : np.ndarray
        The time coordinate of the signal.
    max_width_samples : int
        The maximum width of the peak in samples.
    diff_limit : float
        The derivative limit for the detection of sharp rising/falling edges of
        peaks. Calculated in units of signal units / time units.
    dt : float, optional, default=1.0
        The time difference between consecutive samples.
    return_peak_center_times : bool, optional, default=False
        Return the time coordinates of the center of the detected peaks.
    return_peak_width_samples : bool, optional, default=False
        Return the widths of the detected peaks in samples.
    return_peak_amplitudes : bool, optional, default=False
        Return the amplitudes of the detected peaks.
    remove_peaks : bool, optional, default=True
        Remove the detected peaks from the signal by interpolation.
    interpolation : str, optional, default='linear'
        The interpolation method to use for removing peaks. Only 'linear' is
        implemented.
    
    Returns
    -------
    processed_signal : np.ndarray
        The processed signal.
    num_detected_samples : int
        The number of detected samples.
    loc_detected_samples : np.ndarray
        The location of detected samples as a boolean index array.
    peak_center_times : np.ndarray, optional
        The time coordinates of the center of the detected peaks.
    peak_width_samples : np.ndarray, optional
        The widths of the detected peaks in samples.
    peak_amplitudes : np.ndarray, optional
        The amplitudes of the detected peaks.

    Notes
    -----
    A sharp peak is defined as a sequence of consecutive samples that are
    between two samples of maximum distance `max_width_samples`, where:

    - at the first sample, the derivative exceeds `+diff_limit`
      (raising edge of `over_positive_difflimit`)
    - at the last sample, the derivative raises above `-diff_limit`
      (falling edge of `under_negative_difflimit`)
    
    The implementation uses numpy-implementated convolution for faster
    calculations. This has been found to be ca. 50% faster than a pure Python
    for-loop implementation.
    """

    if max_width_samples is None:
        raise ValueError("No value was given for 'max_width_values'")

    if max_width_samples <= 0:
        raise ValueError("The value of 'max_width_values' has to be positive.")

    if diff_limit is None:
        raise ValueError("No value was given for 'diff_limit'")

    if diff_limit <= 0:
        raise ValueError("The value of 'diff_limit' has to be positive.")

    if signal.ndim != 1:
        raise ValueError("The signal has to be a 1D array.")

    if time.ndim != 1:
        raise ValueError("The time has to be a 1D array.")

    if len(signal) != len(time):
        raise ValueError("The signal and time arrays have to be the same length.")

    # Use numpy implementation of convolution
    conv = np.convolve
    
    # Peak detection start
    diff = np.diff(signal, n=1, prepend=[signal[0]])

    # Convert the derivative limit to a difference limit
    diff_limit_ = diff_limit * dt

    over_positive_difflimit = diff > diff_limit_
    under_negative_difflimit = diff < -diff_limit_

    opd_raising_edge = (conv(over_positive_difflimit, [1, -1], mode='same') == 1)
    und_falling_edge = (conv(under_negative_difflimit, [1, -1], mode='same') == -1)

    # Generating 'slopes': beginning a descending slope at a raising edge,
    # and ending an ascending slope at a falling edge
    desc_slope_kernel = np.zeros(2*max_width_samples - 1)
    asc_slope_kernel = np.zeros(2*max_width_samples - 1)
    for i in range(1, max_width_samples + 1):
        desc_slope_kernel[-i]  = i
        asc_slope_kernel[i-1]  = i

    slope_conv1 = conv(opd_raising_edge, desc_slope_kernel, mode='same')
    slope_conv2 = conv(und_falling_edge, asc_slope_kernel, mode='same')

    # treat the case where two raising/falling edges are too close to each other
    slope_conv1 = np.clip(slope_conv1, 0, max_width_samples)
    slope_conv2 = np.clip(slope_conv2, 0, max_width_samples)

    # If two sloped fields overlap such that the slope peaks are closer than
    # `max_width_samples`, their sum will be larger than `max_width_samples`.
    # The overlap is equivalent to a peak existing in the original signal with a
    # width of at most  `max_width_samples`.
    slope_conv_sum = slope_conv1 + slope_conv2
    is_peak_location = slope_conv_sum > max_width_samples
    # Peak detection end
    
    # Removing and interpolating peaks found, if requested
    processed_signal = np.copy(signal)
    if remove_peaks:
        if interpolation=='linear':
            processed_signal[is_peak_location] = np.interp(
                time[is_peak_location],
                time[~is_peak_location],
                signal[~is_peak_location]
            )
        else:
            ValueError("Only linear interpolation is implemented.")
    else:
        # processed_signal is already assigned
        pass
    
    # Handle optional returns based on arguments
    ret = (processed_signal, )

    # These are already calculated, so we return them
    ret += (np.sum(is_peak_location), )
    ret += (is_peak_location, )
        
    if return_peak_center_times or return_peak_width_samples or return_peak_amplitudes:
        segment_split_indices = np.where(np.diff(is_peak_location) == True)[0]+1
        indices = np.arange(len(signal))

        split = np.split(indices, segment_split_indices)

        if return_peak_center_times:
            ret+= (np.array([
                (time[segment[0]] + time[segment[-1]]) / 2
                for segment
                in split
                if np.all(is_peak_location[segment])
            ]), )
            
        if return_peak_width_samples:
            ret+= (np.array([
                len(segment)
                for segment
                in split
                if np.all(is_peak_location[segment])
            ]), )
            
        if return_peak_amplitudes:
            min_signal_index = 0
            max_signal_index = len(signal)-1
            ret+= (np.array([
                np.max(signal[segment]) - (signal[max(min_signal_index,segment[0]-1)] + signal[min(segment[-1]+1, max_signal_index)]) / 2
                for segment
                in split
                if np.all(is_peak_location[segment])
            ]), )
        
    if len(ret) == 1:
        ret = ret[0]
    
    return ret