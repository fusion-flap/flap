# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:34:49 2019

@author: Zoletnik

Spectral analysis tools fro FLAP
"""
import math
import numpy as np
import flap.config
import flap.coordinate
#from .coordinate import *
from scipy import signal
import copy
#import matplotlib.pyplot as plt

def _spectral_calc_interval_selection(d, ref, coordinate,intervals,interval_n):
    """ Helper function for spectral and correlation calculation.
        Determines the processing intervals and returns in a
        flap.Intervals object. The intervals will have identical length.

        INPUT:
            d, ref: flap.DataObjects
            If ref is set it is assumed that the selection coordinate step size is identical in d and ref.
            coordinate: Coordinate name (string)
            intervals: Information of processing intervals.
                       If dictionary with a single key: {selection coordinate: description})
                           Key is a coordinate name which can be different from the calculation
                           coordinate.
                           Description can be flap.Intervals, flap.DataObject or
                           a list of two numbers. If it is a data object with data name identical to
                           the coordinate the error ranges of the data object will be used for
                           interval. If the data name is not the same as coordinate a coordinate with the
                           same name will be searched for in the data object and the value_ranges
                           will be used fromm it to set the intervals.
                       If not a dictionary and not None is is interpreted as the interval
                           description, the selection coordinate is taken the same as
                           coordinate.
                       If None, the whole data interval will be used as a single interval.
            interval_n: Minimum number of intervals to use for the processing. These are identical
                        length intervals inserted into the input interval list.
        Returns:
            intervals, index_intervals
                intervals: The intervals in the coordinate unit (Intervals object)
                index_intervals: The index intervals in the data array (Intervals object)
    """

    if (type(intervals) is dict):
        sel_coordinate = list(intervals.keys())[0]
    else:
        sel_coordinate = coordinate
    if (sel_coordinate != coordinate):
        raise ("At present for spectral calculation the interval selection coordinate should be the same as the calculation coordinate.")

    try:
        coord = d.get_coordinate_object(coordinate)
    except Exception as e:
        raise e
    
    try:    
        calc_int, calc_int_ind, sel_int, sel_int_ind = d.proc_interval_limits(coordinate, intervals=intervals)
    except Exception as e:
        raise e
    intervals_low = sel_int[0]
    intervals_high = sel_int[1]    
# This part is commented out as we assume identiacl coordinate for d and ref

#    d_intervals_low = sel_int[0]
#    d_intervals_high = sel_int[1]

#
#    if (ref is not None):
#        try:    
#            calc_int, calc_int_ind, sel_int, sel_int_ind = ref.proc_interval_limits(coordinate, intervals=intervals)
#        except Exception as e:
#            raise e
#        ref_intervals_low = sel_int[0]
#        ref_intervals_high = sel_int[1]
#        intervals_low = []
#        intervals_high = []
#        d_int_low_min = np.amin(d_intervals_low)
#        d_int_low_max = np.amax(d_intervals_low)
#        for i in range(len(ref_intervals_low)):
#            if ((ref_intervals_low[i] >= d_int_low_min) and
#                  (ref_intervals_low <= d_int_low_max)):
#                intervals_low.append(ref_intervals_low[i])
#                intervals_high.append(ref_intervals_high[i])
#        ref_coord = ref.get_coordinate_object(coordinate)
#        if ((math.fabs(ref_coord.start - coord.start) > math.fabs(ref_coord.step[0] ) / 10)
#            or (math.fabs(ref_coord.step[0] - coord.step[0]) / math.fabs(ref_coord.step[0]) > 1e-4) 
#            ):
#            raise ValueError("The start and step of the calculating coordinates in the two data objects should be identical.")
#    else:
#        intervals_low = d_intervals_low 
#        intervals_high = d_intervals_high

    if (len(intervals_low) > 1):
        # Ensuring that the intervals are in asceding order
        sort_ind = np.argsort(intervals_low)
        intervals_low = intervals_low[sort_ind]
        intervals_high = intervals_high[sort_ind]
        ind_overlap = np.nonzero(intervals_high[0:-2] > intervals_low[1:-1])[0]
        if (len(ind_overlap) != 0):
            raise ValueError("Intervals overlap, not suitable for calculation.")
        intervals_length = intervals_high - intervals_low
        # Determining how many different intervals are available
        int_lens = np.ndarray(0,dtype=intervals_low.dtype)
        int_num = np.ndarray(0,dtype=np.int32)
        margin = abs(coord.step[0])
        i = 0
        while (True):
            ind_new = np.nonzero(intervals_length > 0)[0]
            if (len(ind_new) == 0):
                break
            ind = np.nonzero(np.abs(intervals_length - intervals_length[ind_new[0]]) < margin)[0]
            int_lens = np.append(int_lens, intervals_length[ind_new[0]])
            int_num = np.append(int_num, len(ind))
            intervals_length[ind] = -1

        # Sorting in reverse order according to interval length
        sort_ind = np.argsort(int_lens)
        int_num = np.flip(int_num[sort_ind])
        int_lens = np.flip(int_lens[sort_ind])
        # Dropping too small intervals
        ind_small = np.nonzero(int_lens < int_lens[0] / 2)[0]
        if (len(ind_small) != 0):
            int_lens = int_lens[0:ind_small[0]]
            int_num = int_num[0:ind_small[0]]
        # Trying to use the shortest interval as processing length
        proc_len = int_lens[-1]
        ind = np.nonzero(int_lens >= proc_len)[0]
        proc_n = np.sum(int_num[ind])
        if (proc_n < interval_n):
            # If this is not successful splitting the intervals smaller and smaller
            proc_len_start = proc_len
            for n_split in range(2,interval_n):
                proc_len = proc_len_start / n_split
                proc_n = 0
                for j in range(len(int_lens)):
                    proc_n += (int_lens[j] // proc_len) * int_num[j]
                if (proc_n >= interval_n):
                    break
            else:
                raise ValueError("Could not find "+str(interval_n)+" processing intervals.")

        proc_interval_start = np.ndarray(0,dtype=intervals_low.dtype)
        proc_interval_end = np.ndarray(0,dtype=intervals_high.dtype)
        for i in range(len(intervals_low)):
            st = intervals_low[i]
            while (st + proc_len <= intervals_high[i] + margin):
                proc_interval_start = np.append(proc_interval_start, st)
                proc_interval_end = np.append(proc_interval_end, st + proc_len)
                st += proc_len
        if (proc_interval_start.size < interval_n):
            raise RuntimeError("Internal error in finding processing intervals.")
        proc_interval_len = proc_len
    else:
        proc_interval_len = (intervals_high[0] - intervals_low[0]) / interval_n
        proc_interval_start = np.arange(interval_n) * proc_interval_len + intervals_low[0]
        proc_interval_end = proc_interval_start + proc_interval_len

    if (coord.step[0] > 0):
        proc_interval_index_start = np.round((proc_interval_start - coord.start) / coord.step[0]).astype(np.int32) + 1
        proc_interval_index_len = int(np.round(proc_interval_len / coord.step[0])) - 2
        proc_interval_index_end = proc_interval_index_start + proc_interval_index_len
    else:
        step = -coord.step[0]
        #npoint = d.shape[coord.dimension_list[0]]                              #UNUSED VARIABLE
        proc_interval_index_len = int(round(proc_interval_len / step)) - 2
        proc_interval_index_start = np.round((proc_interval_end - coord.start) / coord.step[0]).astype(np.int32) + 1
        proc_interval_index_end = proc_interval_index_start + proc_interval_index_len
        
    return flap.coordinate.Intervals(proc_interval_start, proc_interval_end),  \
           flap.coordinate.Intervals(proc_interval_index_start, proc_interval_index_end),

def trend_removal_func(d, ax, trend, x=None, return_trend=False, return_poly=False):
    """ This function makes the _trend_removal internal function public
    """
    return _trend_removal(d, ax, trend, x=x, return_trend=return_trend, return_poly=return_poly)

def _trend_removal(d, ax, trend, x=None, return_trend=False, return_poly=False):
    """
    Removes the trend from the data. Operates on one axis between two indices of the data array.
    INPUT:
        d: Data array (Numpy array)
        ax: The axis along which to operate (0...)
        trend: Trend removal description. A list, string or None.
                None: Don't remove trend.
                Strings:
                    'Mean': subtract mean
                Lists:
                    ['Poly', n]: Fit an n order polynomial to the data and subtract.
        x: X axis. If not used equidistant will be assumed.
        return_trend: If True the trend data is returned
        return_poly: Return polynomial coefficients for poly trend removal. The coefficients will
                     be in axis ax.
    RETURN value:
        If return_trend == True the trend data is returned
        If (return_poly == True) and polyfit then return polynomial parameters
        Otherwise return None
        The input array is modified.
    """
    if (trend is None):
        return
    if ((type(trend) is list) and (trend[0] == 'Poly')):
        pass
    else:
        if (return_poly):
            raise ValueError("Polynomial trend fit parameters can be returned only for polynomial trend removal.")
    if (type(trend) is str):
        if (trend == 'Mean'):
            d[:] = signal.detrend(d, axis=ax, type='constant')
            return
        else:
            raise ValueError("Unknown trend removal method: "+trend)
    elif (type(trend) is list):
        if ((len(trend) == 2) and (trend[0] == 'Poly')):
            try:
                order = int(trend[1])
            except ValueError:
                raise ValueError("Bad order in polynomial trend removal.")
            # This is a simple solution but not very effective.
            # Flattens all dimensions except x and handles all functions one-by-one
            # Finally rearranges the data back to the original shape
            if (x is None):
                _x = np.arange(d.shape[ax],dtype=float)
            else:
                _x = copy.deepcopy(x)
            if ((_x.dtype.kind == 'i') or (_x.dtype.kind == 'u')):
                _x = np.asarray(_x,'float')
            if (d.ndim > 1):
                if (ax != 0):
                    d = np.swapaxes(d,ax,0)
                orig_shape = d.shape
                if (d.ndim > 2):
                    new_shape = tuple([d.shape[0], d.size // d.shape[0]])
                    d = d.reshape(new_shape,order='F')
                    if (return_trend):
                        trend_data = np.ndarray(new_shape,dtype=d.dtype)
                else:
                    new_shape = d.shape
                xx = np.zeros((new_shape[0],order), dtype=float)
                for i in range(order):
                    xx[:,i] = _x ** (i + 1)
                p = np.polynomial.polynomial.polyfit(_x,d,order)
                for i in range(new_shape[1]):
                    tr = p[0,i]
                    for j in range(order):
                        tr = tr + p[j + 1, i] * xx[:,j]
                    d[:,i] = d[:,i] - tr.astype(d.dtype)
                    if (return_trend):
                        trend_data[:,i] = tr.astype(d.dtype)
                if (len(orig_shape) > 2):
                   d = d.reshape(orig_shape,order='F')
                   if (return_trend):
                       trend_data = trend_data.reshape(orig_shape,order='F')
                   if (return_poly):
                       p_shape = list(orig_shape)
                       p_shape[0] = p.shape[0]
                       p = p.reshape(p_shape,order='F')
                if (ax != 0):
                    d = np.swapaxes(d,ax,0)
                    if (return_trend):
                        trend_data = np.swapaxes(trend_data,ax,0)
                    if (return_poly):
                        p = np.swapaxes(p,ax,0)
                if (return_trend):
                    return trend_data
                elif(return_poly):
                    return p
                else:
                    return
            else:
                p = np.polynomial.polynomial.polyfit(_x,d,order)
                d = d - p[0]
                for i in range(order):
                    d = d - p[i + 1] * _x ** (i + 1)
                return
    raise ValueError("Unknown trend removal method.")

def _apsd(d, coordinate=None, intervals=None, options=None):
    """
        Auto power Spectral Density caclculation for the data object d.
        Returns a data object with the coordinate replaced by frequency or wavenumber.
        The power spectrum is calculated in multiple intervals (described by slicing)
        and the mean and variance will be returned.

        INPUT:
            d: A flap.DataObject.
            coordinate: The name of the coordinate (string) along which to calculate APSD.
                        This coordinate should change only along one data dimension and should be equidistant.
                        This and all other cordinates changing along the data dimension of
                        this coordinate will be removed. A new coordinate with name
                        Frequency/Wavenumber will be added. The unit will be
                        derived from the unit of the coordinate (e.g., Hz cm-1, m-1)
            intervals: Information of processing intervals.
                       If dictionary with a single key: {selection coordinate: description})
                           Key is a coordinate name which can be different from the calculation
                           coordinate.
                           Description can be flap.Intervals, flap.DataObject or
                           a list of two numbers. If it is a data object with data name identical to
                           the coordinate the error ranges of the data object will be used for
                           interval. If the data name is not the same as coordinate a coordinate with the
                           same name will be searched for in the data object and the value_ranges
                           will be used fromm it to set the intervals.
                       If not a dictionary and not None it is interpreted as the interval
                           description, the selection coordinate is taken the same as
                           coordinate.
                       If None, the whole data interval will be used as a single interval.
            options: Dictionary. (Keys can be abbreviated)
                'Wavenumber' : True/False. Will use 2*Pi*f for the output coordinate scale, this is useful for
                               wavenumber calculation.
                'Resolution': Output resolution in the unit of the output coordinate.
                'Range': Output range in the unit of the output coordinate.
                'Logarithmic': True/False. If True will create logarithmic frequency binning.
                'Interval_n': Minimum number of intervals to use for the processing. These are identical
                              length intervals inserted into the input interval list. Default is 8.
                'Error calculation' : True/False. Calculate or not error. Omitting error calculation
                                      increases speed. If Interval_n is 1 no error calculation is done.
                'Trend removal': Trend removal description (see also _trend_removal()). A list, string or None.
                             None: Don't remove trend.
                             Strings:
                               'Mean': subtract mean
                             Lists:
                               ['Poly', n]: Fit an n order polynomial to the data and subtract.
                            Trend removal will be applied to each interval separately.
                 'Hanning': True/False Use a Hanning window.


    """
    if (d.data is None):
        raise ValueError("Cannot do spectral analysis without data.")
    default_options = {'Wavenumber': False,
                       'Resolution': None,
                       'Range': None,
                       'Logarithmic': False,
                       'Interval_n': 8,
                       'Trend removal': ['Poly', 2],
                       'Error calculation': True,
                       'Hanning' : True
                       }
    _options = flap.config.merge_options(default_options, options, data_source=d.data_source, section='PS')
    if (coordinate is None):
        c_names = d.coordinate_names()
        try:
            c_names.index('Time')
            _coordinate = 'Time'
        except ValueError:
            raise ValueError("No coordinate is given for spectrum calculation and no Time coordinate found.")
    else:
        _coordinate = coordinate
    trend = _options['Trend removal']
    wavenumber = _options['Wavenumber']
    interval_n = _options['Interval_n']
    log_scale = _options['Logarithmic']
    hanning = _options['Hanning']
    try:
        coord_obj = d.get_coordinate_object(_coordinate)
    except Exception as e:
        raise e
    if (len(coord_obj.dimension_list) != 1):
        raise ValueError("Spectrum calculation is possible only along coordinates changing along one dimension.")
    if (not coord_obj.mode.equidistant):
        raise ValueError("Spectrum calculation is possible only along equidistant coordinates.")

    try:
        intervals, index_intervals = _spectral_calc_interval_selection(d,None,_coordinate,intervals,interval_n)
    except Exception as e:
        raise e
    interval_n, start_ind = intervals.interval_number()
    calc_error = _options['Error calculation']
    if (interval_n < 2):
        calc_error = False

    int_low, int_high = intervals.interval_limits()
    res_nat = 1./(int_high[0] - int_low[0])
    range_nat = [0., 1./float(coord_obj.step[0])/2]

    index_int_low, index_int_high = index_intervals.interval_limits()
    interval_sample_n = (index_int_high[0] - index_int_low[0]) + 1
    
    # Determining the output array shape.
    out_shape = list(d.shape)
    # Determining two pairs of index tuples for copying the data after PS calculation
    # of one time interval. For complex data we need to copy in two steps
    proc_dim = coord_obj.dimension_list[0]
    if (d.data.dtype.kind == 'c' ):
        # For complex data negative frequencies are also valuable
        # n_apsd is the number of valuable points in the spectrum after rearrangement but before
        # and range and resolution transformation
        n_apsd = index_int_high[0] - index_int_low[0] + 1
        # These will be tuples used in reorganizing the raw FFT spectra into continuous
        # frequency scale. We need this as for complex data the negative frequencies are
        # in the second half of the array
        ind_in1 = [slice(0,d) for d in d.shape]
        ind_in2 = copy.deepcopy(ind_in1)
        ind_out1 = copy.deepcopy(ind_in1)
        ind_out2 = copy.deepcopy(ind_in1)
        ind_in1[proc_dim] = slice(0,int(n_apsd/2))
        ind_out1[proc_dim] = slice(n_apsd-int(n_apsd/2), n_apsd)
        # zero_ind is the index where the 0 frequency will be after rearranging the spectrum
        zero_ind = n_apsd - int(n_apsd/2)
        ind_in2[proc_dim] = slice(int(n_apsd/2),n_apsd)
        ind_out2[proc_dim] = slice(0, n_apsd-int(n_apsd/2))
    else:
        n_apsd = int((index_int_high[0] - index_int_low[0] + 1) / 2)
        ind_in1 = [slice(0,d) for d in d.shape]
        ind_in1[proc_dim] = slice(0,n_apsd)
        zero_ind = 0
        ind_out1 = None
        ind_in2 = None
        ind_out2 = None


    # Calculating the binning boxes from the resolution and range and related indices   
    ind_bin, ind_slice, out_data_num, ind_nonzero, index_nonzero, ind_zero, nf_out, f_cent, \
    fcent_index_range, res = _spectrum_binning_indices(wavenumber,
                                                       n_apsd, 
                                                       _options, 
                                                       zero_ind, 
                                                       res_nat,
                                                       range_nat,
                                                       log_scale, 
                                                       out_shape, 
                                                       proc_dim)
                
    out_shape[proc_dim] = nf_out
    # These arrays will collect the data and the square of the data to enable error calculation
    out_data = np.zeros(tuple(out_shape), dtype=float)
    if (calc_error):
        out_data_square = np.zeros(tuple(out_shape), dtype=float)
    # This is a tuple to index into the original data array to get data for processing
    ind_proc = [slice(0,d) for d in d.shape]
    # Number of processing intervals
    n_proc_int = len(int_low)
    if (hanning):
        hanning_window = np.hanning(index_int_high[0] - index_int_low[0] + 1) 
        hanning_window /= math.sqrt(3./8)
        if (len(d.shape) > 1):
            han_sh = [1] * len(d.shape)
            han_sh[proc_dim] = hanning_window.size
            hanning_window = hanning_window.reshape(han_sh)
    # We need to determine a shape to which the out_data_num array will be broadcasted to 
    # allow dividing all spectra. bs is this shape
    if (ind_nonzero is not None):
        bs = [1]*out_data.ndim
        bs[proc_dim] = len(index_nonzero)
        bs= tuple(bs)
    else:
        bs = [1]*out_data.ndim
        bs[proc_dim] = out_data.shape[proc_dim]
        bs = tuple(bs)

    for i_int in range(n_proc_int):
        # Setting the index range of the actual interval
        ind_proc[proc_dim] = slice(index_int_low[i_int], index_int_high[i_int] + 1)
        # Getting the data for processing, this might be multi-dim
        data_proc = copy.deepcopy(d.data[tuple(ind_proc)])
        if (trend is not None):
            try:
                _trend_removal(data_proc,proc_dim,trend)
            except Exception as e:
                raise e
        if (hanning):
            data_proc = data_proc * hanning_window
        # Calculating APS on natural resolution, full frequency scale
        dfft = np.fft.fft(data_proc,axis=proc_dim)
        dps = (dfft.conjugate() * dfft).real
        # Rearranging the negative frequencies
        if (ind_in2 is not None):
            dps1 = np.empty(dps.shape,dtype=dps.dtype)
            dps1[tuple(ind_out1)] = dps[tuple(ind_in1)]
            dps1[tuple(ind_out2)] = dps[tuple(ind_in2)]
            dps = dps1
        else:
            dps = dps[tuple(ind_in1)]
        # Cutting the range
        if ((ind_slice) is not None):
            dps = dps[tuple(ind_slice)]
        # Binning the spectrum and summing up the time intervals
        if (ind_bin is not None):
            out_data_interval = np.zeros(tuple(out_shape), dtype=float)
            np.add.at(out_data_interval, tuple(ind_bin), dps)
        else:
            out_data_interval = dps
        # Dividing by the number of points in each bin
        if (ind_nonzero is not None):
            out_data_interval[tuple(ind_nonzero)] /= out_data_num[index_nonzero].reshape(bs)
        else:
            out_data_interval /= out_data_num.reshape(bs)
        out_data_interval /= interval_sample_n
        out_data += out_data_interval
        if (calc_error):
            out_data_square += out_data_interval ** 2
    out_data /= n_proc_int
    if (calc_error):
        out_err = np.sqrt(np.clip(out_data_square / n_proc_int - out_data ** 2,
                                  0,None)) / math.sqrt(n_proc_int)
    else:
        out_err = None
    # If there are frequency bins without data setting them to np.NaN
    if (ind_nonzero is not None):
        out_data[tuple(ind_zero)] = np.NaN
    # We create the new data object with this trick as data_object.py cannot be imported
    d_out = type(d)(data_array=out_data,
                    error=out_err,
                    coordinates=d.coordinates,
                    exp_id=d.exp_id,
                    data_unit=flap.coordinate.Unit("Spectral density"))

    if (wavenumber):
        out_name = 'Wavenumber'
        out_unit = '1/'+coord_obj.unit.unit
        res *= 2 * math.pi
        #fcent *= 2* math.pi                                                        #UNUSED VARIABLE
    else:
        out_name = 'Frequency'
        out_unit = 'Hz'

    # Finding all coordinates which have common dimension with the converted one.
    # These will be deleted.
    del_coord_list = []
    for c in d_out.coordinates:
        try:
            c.dimension_list.index(proc_dim)
            del_coord_list.append(c.unit.name)
        except ValueError:
            pass
    for c in del_coord_list:
        d_out.del_coordinate(c)

    if (log_scale):
        c = flap.coordinate.Coordinate(name = out_name,
                                       unit = out_unit,
                                       mode = flap.coordinate.CoordinateMode(equidistant=False),
                                       shape = [f_cent.size],
                                       values = f_cent,
                                       dimension_list=[proc_dim])
    else:
        c = flap.coordinate.Coordinate(name = out_name,
                                       unit = out_unit,
                                       mode = flap.coordinate.CoordinateMode(equidistant=True),
                                       shape = [],
                                       start = (fcent_index_range[0] - zero_ind) * res,
                                       step = res,
                                       dimension_list=[proc_dim])
    d_out.add_coordinate_object(c,index=0)

    return d_out

def _spectrum_binning_indices(wavenumber, n_apsd, _options, zero_ind, res_nat, range_nat, log_scale, out_shape, proc_dim):
    """ Helper routine for apsd and cpsd for calculating numbers and indices
        for processing the spectra.
        Returns: ind_bin, ind_slice, out_data_num, ind_nonzero, index_nonzero, ind_zero, nf_out, 
        f_cent, fcent_index_range, res
    """
    # Calculating the binning boxes from the resolution and range
    fscale_nat = (np.arange(n_apsd,dtype=float) - zero_ind) * res_nat
    if (log_scale):
        if (_options['Range'] is not None):
            rang = _options['Range']
            if ((type(rang) is not list) or (len(rang) != 2)):
                raise ValueError("Invalid spectrum range setting.")
            if (_options['Resolution'] is not None):
                res = _options['Resolution']
            else:
                res = rang[0] / 10
        else:
            if (_options['Resolution'] is not None):
                res = _options['Resolution']
                rang = [res, range_nat[1]]
            else:
                res = res_nat
                rang = [res_nat * 10, range_nat[1]]
        if (rang[0] >= rang[1]):
            raise ValueError("Illegal frequency range.")
        if (rang[0] <= 0):
            raise ValueError("Illegal frequency range for logarithmic frequency resolution.")
        # Setting the lower and upper limit of the first box so as f_high-f_low=res and
        # (log10(f_low)+log10(f_high))/2 = log10(range[0])
        f_low = (-res + math.sqrt(res ** 2 + 4 * rang[0] ** 2))/ 2
        f_high = rang[0] ** 2 / f_low
        # Setting up a box list which is linear on the log scale
        delta = math.log10(f_high/f_low)
        nf_out = (math.log10(rang[1]) - math.log10(f_low)) // delta + 2
        f_box = 10 ** (math.log10(f_low) + np.arange(nf_out) * delta)
#        if (f_box[-1] > range_nat[1]):
#            f_box[-1] = range_nat[1]
        # Box index for the original spectrum points
        apsd_index = np.digitize(fscale_nat, f_box) - 1
        ind_out_low = np.nonzero(apsd_index < 0)[0]
        ind_out_high = np.nonzero(apsd_index >= f_box.size - 1)[0]
        if ((ind_out_low.size != 0) or (ind_out_high.size != 0)):
            if (ind_out_low.size == 0):
                slice_start = 0
            else:
                slice_start = ind_out_low[-1]+1
            if (ind_out_high.size == 0):
                slice_end = fscale_nat.size
            else:
                slice_end = ind_out_high[0]
            apsd_slice = slice(slice_start, slice_end)
            ind_slice = [slice(0,d) for d in out_shape]
            ind_slice[proc_dim] = apsd_slice
            apsd_index = apsd_index[apsd_slice]
        else:
            apsd_slice = None
            ind_slice = None
        f_cent = np.sqrt(f_box[0:-1] * f_box[1:])
        nf_out = f_cent.size
        out_data_num = np.zeros(nf_out,dtype=np.int32)
        np.add.at(out_data_num, apsd_index, np.int32(1))
        index_nonzero = np.nonzero(out_data_num != 0)[0]
        if (index_nonzero.size == out_data_num.size):
            ind_nonzero = None
            ind_zero = None
        else:
            ind_nonzero = [slice(0,d) for d in out_shape]
            ind_nonzero[proc_dim] = index_nonzero
            index_zero = np.nonzero(out_data_num == 0)[0]
            ind_zero = [slice(0,d) for d in out_shape]
            ind_zero[proc_dim] = index_zero
        ind_bin = [slice(0,d) for d in out_shape]
        ind_bin[proc_dim] = apsd_index
        fcent_index_range = None
    else:
        # Linear scale
        if (_options['Resolution'] is not None):
            if (wavenumber):
                res = _options['Resolution']/2/math.pi
            else:
                res = _options['Resolution']
            if (res > range_nat[1] / 2):
                raise ValueError("Requested resolution is too coarse.")
        else:
            res = res_nat
        res_bin = int(round(res / res_nat))
        if (res_bin < 1):
            res_bin = 1
        res = res_nat * res_bin
        # Determining the number of bins

        nf_out = int(n_apsd / res_bin)

        # The index range in the apsd array where the central frequencies are available
        if (res_bin == 1):
            fcent_index_range = [0, n_apsd - 1]
            nf_out = n_apsd
        else:
            fcent_index_range = [zero_ind % res_bin,
                                 (n_apsd - 1 - (zero_ind % res_bin)) // res_bin * res_bin
                                   + zero_ind % res_bin]
            nf_out = (fcent_index_range[1] - fcent_index_range[0]) // res_bin + 1
        if (_options['Range'] is not None):
            rang = _options['Range']
            if ((type(rang) is not list) or (len(rang) != 2)):
                raise ValueError("Invalid spectrum range setting.")
            if (rang[0] >= rang[1]):
                raise ValueError("Illegal frequency range.")
            if (fcent_index_range[0] < rang[0] / res_nat + zero_ind):
                fcent_index_range[0] = int(round(rang[0] / res)) * res_bin + zero_ind
            if (fcent_index_range[1] > rang[1] / res_nat + zero_ind):
                fcent_index_range[1] = int(round(rang[1] / res)) * res_bin + zero_ind
        nf_out = (fcent_index_range[1] - fcent_index_range[0]) // res_bin + 1
        if (nf_out < 3):
            raise ValueError("Too coarse spectrum resolution.")
        if ((fcent_index_range[0] < 0) or (fcent_index_range[0] > n_apsd - 1) \
             or (fcent_index_range[1] < 0) or (fcent_index_range[1] > n_apsd - 1)):
            raise ValueError("Spectrum axis range is outside of natural ranges.")
        # This slice will cut the necessary part from the raw APSD sepctrum
        apsd_slice = slice(fcent_index_range[0] - res_bin // 2,
                          fcent_index_range[1] + (res_bin - res_bin // 2))
        # A full box start is this number of apsd spectrum pints before apsd_slice.start
        start_shift = 0
        if (apsd_slice.start < 0):
            start_shift = - apsd_slice.start
            apsd_slice = slice(0, apsd_slice.stop)
        if (apsd_slice.stop > n_apsd):
            apsd_slice = slice(apsd_slice.start, n_apsd)
        # This index array will contain the box index for each APSD spectral point remaining after
        # the above slice
        if (res_bin != 1):
            apsd_index = (np.arange(apsd_slice.stop - apsd_slice.start,dtype=np.int32)
                          + start_shift) // res_bin
        else:
            apsd_index = None
        if (apsd_slice is not None):
            ind_slice = [slice(0,d) for d in out_shape]
            ind_slice[proc_dim] = apsd_slice
        else:
            ind_slice = None
        if (apsd_index is not None):
            ind_bin = [slice(0,d) for d in out_shape]
            ind_bin[proc_dim] = apsd_index
            out_data_num = np.zeros(nf_out,dtype=np.int32)
            np.add.at(out_data_num, apsd_index, np.int32(1))
        else:
            ind_bin = None
            out_data_num = np.zeros(nf_out,dtype=np.int32) + 1
        ind_nonzero = None
        index_nonzero = None
        ind_zero = None
        f_cent = None
    return ind_bin, ind_slice, out_data_num, ind_nonzero, index_nonzero, ind_zero, nf_out, \
           f_cent,fcent_index_range, res
    
def _cpsd(d, ref=None, coordinate=None, intervals=None, options=None):
    """
        Complex Cross Power Spectrum calculation for the data object d taking d_ref as reference.
        If ref is not set d is used as reference, that is all spectra are calculated within d.
        Calculates all spectra between all signals in ref and d, but not inside d and ref.
        d and ref both should have the same equidistant coordinate with equal sampling points.
        Returns a data object with dimension number d.dim+ref.dim-1. The coordinate is replaced 
        by frequency or wavenumber.
        The spectrum is calculated in multiple intervals (described by slicing)
        and the mean and variance will be returned.

        INPUT:
            d: A flap.DataObject.
            ref: Another flap.DataObject
            coordinate: The name of the coordinate (string) along which to calculate CPSD.
                        This coordinate should change only along one data dimension and should be equidistant.
                        This and all other cordinates changing along the data dimension of
                        this coordinate will be removed. A new coordinate with name
                        Frequency/Wavenumber will be added. The unit will be
                        derived from the unit of the coordinate (e.g., Hz cm-1, m-1)
            intervals: Information of processing intervals.
                       If dictionary with a single key: {selection coordinate: description})
                           Key is a coordinate name which can be different from the calculation
                           coordinate.
                           Description can be flap.Intervals, flap.DataObject or
                           a list of two numbers. If it is a data object with data name identical to
                           the coordinate the error ranges of the data object will be used for
                           interval. If the data name is not the same as coordinate a coordinate with the
                           same name will be searched for in the data object and the value_ranges
                           will be used fromm it to set the intervals.
                       If not a dictionary and not None it is interpreted as the interval
                           description, the selection coordinate is taken the same as
                           coordinate.
                       If None, the whole data interval will be used as a single interval.
            options: Dictionary. (Keys can be abbreviated)
                'Wavenumber' : True/False. Will use 2*Pi*f for the output coordinate scale, this is useful for
                               wavenumber calculation.
                'Resolution': Output resolution in the unit of the output coordinate.
                'Range': Output range in the unit of the output coordinate.
                'Logarithmic': True/False. If True will create logarithmic frequency binning.
                'Interval_n': Minimum number of intervals to use for the processing. These are identical
                              length intervals inserted into the input interval list. Default is 8.
                'Error calculation' : True/False. Calculate or not error. Omitting error calculation
                                      increases speed. If Interval_n is 1 no error calculation is done.
                'Trend removal': Trend removal description (see also _trend_removal()). A list, string or None.
                             None: Don't remove trend.
                             Strings:
                               'mean': subtract mean
                             Lists:
                               ['Poly', n]: Fit an n order polynomial to the data and subtract.
                            Trend removal will be applied to each interval separately.
                'Hanning': True/False Use a Hanning window.
                'Error calculation' : True/False. Calculate or not error. Omitting error calculation
                                      increases speed. If Interval_n is 1 no error calculation is done.
                'Normalize': Normalize crosspower spectrum, that is return
        Return value:
            Three data objects:
            spectrum, phase, confidence
                spectrum: The complex power spectrum or coherency if options['Normalize'] is True
                          The error will contain the condifence level.
    """
    
    if (d.data is None):
        raise ValueError("Cannot do spectral analysis without data.")

    default_options = {'Wavenumber': False,
                       'Resolution': None,
                       'Range': None,
                       'Logarithmic': False,
                       'Interval_n': 8,
                       'Trend removal': ['Poly', 2],
                       'Hanning' : True,
                       'Error calculation': True,
                       'Normalize': False
                       }
    _options = flap.config.merge_options(default_options, options, data_source=d.data_source, section='PS')
    if (coordinate is None):
        c_names = d.coordinate_names()
        try:
            c_names.index('Time')
            _coordinate = 'Time'
        except ValueError:
            raise ValueError("No coordinate is given for spectrum calculation and no Time coordinate found.")
    else:
        _coordinate = coordinate
    trend = _options['Trend removal']
    wavenumber = _options['Wavenumber']
    interval_n = _options['Interval_n']
    log_scale = _options['Logarithmic']
    hanning = _options['Hanning']
    norm = _options['Normalize']
    error_calc = _options['Error calculation']
    
    try:
        coord_obj = d.get_coordinate_object(_coordinate)
    except Exception as e:
        raise e
    if (len(coord_obj.dimension_list) != 1):
        raise ValueError("Spectrum calculation is possible only along coordinates changing in one dimension.")
    if (not coord_obj.mode.equidistant):
        raise ValueError("Spectrum calculation is possible only along equidistant coordinates.")
    
    if (ref is None):
        _ref = d
        ref_coord_obj = coord_obj
        try:
            intervals, index_intervals = _spectral_calc_interval_selection(d,
                                                                           None, 
                                                                           _coordinate,
                                                                           intervals,
                                                                           interval_n)
        except Exception as e:
            raise e
    else:
        _ref = ref
        try:
            ref_coord_obj = _ref.get_coordinate_object(_coordinate)
        except Exception as e:
            raise e
        if (len(ref_coord_obj.dimension_list) != 1):
            raise ValueError("Spectrum calculation is possible only along coordinates changing in one dimension (ref).")
        if (not ref_coord_obj.mode.equidistant):
            raise ValueError("Spectrum calculation is possible only along equidistant coordinates (ref).")    
        if (math.fabs(ref_coord_obj.step[0] - coord_obj.step[0]) * d.shape[coord_obj.dimension_list[0]] \
            > math.fabs(ref_coord_obj.step[0])):
               raise ValueError("Incompatible coordinate step sizes." )
        if (math.fabs(ref_coord_obj.start - coord_obj.start) > math.fabs(coord_obj.step[0])):
                   raise ValueError("Incompatible coordinate start values." )
        try:
            intervals, index_intervals = _spectral_calc_interval_selection(d,
                                                                           _ref, 
                                                                           _coordinate,
                                                                           intervals,
                                                                           interval_n)
        except Exception as e:
            raise e
                        
    interval_n, start_ind = intervals.interval_number()

    int_low, int_high = intervals.interval_limits()
    res_nat = 1./(int_high[0] - int_low[0])
    range_nat = [0., 1./float(coord_obj.step[0])/2]

    index_int_low, index_int_high = index_intervals.interval_limits()
    interval_sample_n = (index_int_high[0] - index_int_low[0]) + 1

    # The processing dimensions in the two objects
    proc_dim = coord_obj.dimension_list[0]
    proc_dim_ref = ref_coord_obj.dimension_list[0]
    
    # Determining the output array shape. First the d object dimensions will come, the
    # spectrum scale will be proc_dim. Then the ref dimensions will come with proc_dim_ref removed.
    # The size of the output array in the processig dimension will be entered later
    out_shape = list(d.shape)
    out_shape_add = list(_ref.shape)
    del out_shape_add[proc_dim_ref]
    out_shape += out_shape_add
    proc_dim_out = proc_dim
    
    # Flag to show whether the APSDs should be calculated
    aps_calc = error_calc or norm
    
    # Determining two pairs of index tuples for copying the data after PS calculation
    # of one time interval. For complex data we need to copy in two steps
    if ((d.data.dtype.kind == 'c' ) or (_ref.data.dtype.kind == 'c')):
        # For complex data negative frequencies are also valuable
        # n_apsd is the number of valuable points in the spectrum after rearrangement but before
        # and range and resolution transformation
        n_apsd = index_int_high[0] - index_int_low[0] + 1                                   #THIS WAS n_cpsd BEFORE, WOULD HAVE CAUSED AN ERROR
        # These will be tuples used in reorganizing the raw FFT spectra into continuous
        # frequency scale. We need this as for complex data the negative frequencies are
        # in the second half of the array
        ind_in1 = [slice(0,d) for d in out_shape]
        ind_in2 = copy.deepcopy(ind_in1)
        ind_out1 = copy.deepcopy(ind_in1)
        ind_out2 = copy.deepcopy(ind_in1)
        ind_in1[proc_dim_out] = slice(0,int(n_apsd/2))
        ind_out1[proc_dim_out] = slice(n_apsd-int(n_apsd/2), n_apsd)
        # zero_ind is the index where the 0 frequency will be after rearranging the spectrum
        zero_ind = n_apsd - int(n_apsd/2)
        ind_in2[proc_dim_out] = slice(int(n_apsd/2),n_apsd)
        ind_out2[proc_dim_out] = slice(0, n_apsd-int(n_apsd/2))
        if (aps_calc):
            ind_in1_apsd = [slice(0,ds) for ds in d.shape]
            ind_in2_apsd = copy.deepcopy(ind_in1_apsd)
            ind_out1_apsd = copy.deepcopy(ind_in1_apsd)
            ind_out2_apsd = copy.deepcopy(ind_in1_apsd)
            ind_in1_apsd[proc_dim] = slice(0,int(n_apsd/2))
            ind_in2_apsd[proc_dim] = slice(int(n_apsd/2),n_apsd)
            ind_out1_apsd[proc_dim] = slice(n_apsd-int(n_apsd/2), n_apsd)
            ind_out2_apsd[proc_dim] = slice(0, n_apsd-int(n_apsd/2))
            ind_in1_apsd_ref = [slice(0,ds) for ds in _ref.shape]
            ind_in2_apsd_ref = copy.deepcopy(ind_in1_apsd_ref)
            ind_out1_apsd_ref = copy.deepcopy(ind_in1_apsd_ref)
            ind_out2_apsd_ref = copy.deepcopy(ind_in1_apsd_ref)
            ind_in1_apsd_ref[proc_dim_ref] = slice(0,int(n_apsd/2))
            ind_in2_apsd_ref[proc_dim_ref] = slice(int(n_apsd/2),n_apsd)
            ind_out1_apsd_ref[proc_dim_ref] = slice(n_apsd-int(n_apsd/2), n_apsd)
            ind_out2_apsd_ref[proc_dim_ref] = slice(0, n_apsd-int(n_apsd/2))
    else:
        n_apsd = int((index_int_high[0] - index_int_low[0] + 1) / 2)
        ind_in1 = [slice(0,ds) for ds in out_shape]
        ind_in1[proc_dim_out] = slice(0,n_apsd)
        zero_ind = 0
        ind_out1 = None
        ind_in2 = None
        ind_out2 = None
        if (aps_calc):
            ind_in1_apsd = [slice(0,ds) for ds in d.shape]
            ind_in1_apsd[proc_dim] = slice(0,n_apsd)
            ind_in1_apsd_ref = [slice(0,ds) for ds in _ref.shape]
            ind_in1_apsd_ref[proc_dim_ref] = slice(0,n_apsd)
 
    ind_bin, ind_slice, out_data_num, ind_nonzero, index_nonzero, ind_zero, nf_out, f_cent, \
    fcent_index_range, res = _spectrum_binning_indices(wavenumber, n_apsd, 
                                            _options, 
                                            zero_ind, 
                                            res_nat,
                                            range_nat,
                                            log_scale, 
                                            out_shape, 
                                            proc_dim_out)
    if (aps_calc):
        if (ind_slice is not None):
            ind_slice_apsd = [slice(0,ds) for ds in d.shape]
            ind_slice_apsd[proc_dim] = ind_slice[proc_dim_out]
            ind_slice_apsd_ref = [slice(0,ds) for ds in _ref.shape]
            ind_slice_apsd_ref[proc_dim_ref] = ind_slice[proc_dim_out]
        else:
            ind_slice_apsd = None
            ind_slice_apsd_ref = None 
        if (ind_bin is not None):
            ind_bin_apsd = [slice(0,ds) for ds in d.shape]
            ind_bin_apsd[proc_dim] = ind_bin[proc_dim_out]
            ind_bin_apsd_ref = [slice(0,ds) for ds in _ref.shape]
            ind_bin_apsd_ref[proc_dim_ref] = ind_bin[proc_dim_out]
        else:
            ind_bin_apsd = None
            ind_bin_apsd_ref = None
        if (ind_nonzero is not None):
            ind_nonzero_apsd = [slice(0,ds) for ds in d.shape]
            ind_nonzero_apsd[proc_dim] = index_nonzero
            ind_nonzero_apsd_ref = [slice(0,ds) for ds in _ref.shape]
            ind_nonzero_apsd_ref[proc_dim_ref] = index_nonzero
        else:
            ind_nonzero_apsd = None
            ind_nonzero_apsd_ref = None
        
    out_shape[proc_dim_out] = nf_out

    # This will collect the output data
    out_data = np.zeros(tuple(out_shape), dtype=complex)
    # These will collect the autospectra
    if (aps_calc):
        apsd_shape = list(d.shape)
        apsd_shape[proc_dim] = nf_out
        apsd_ref_shape = list(_ref.shape)
        apsd_ref_shape[proc_dim_ref] = nf_out
        apsd = np.zeros(apsd_shape, dtype=float)
        apsd_ref = np.zeros(apsd_ref_shape, dtype=float)
    # This is a tuple to index into the original data arrays to get data for processing
    ind_proc = [slice(0,ds) for ds in d.shape]
    ind_proc_ref = [slice(0,ds) for ds in _ref.shape]
    # Number of processing intervals
    n_proc_int = len(int_low)
    if (hanning):
        hanning_window = np.hanning(index_int_high[0] - index_int_low[0] + 1) 
        hanning_window /= math.sqrt(3./8)
        hanning_window_ref=hanning_window
        if (len(d.shape) > 1):
            han_sh = [1] * len(d.shape)
            han_sh[proc_dim] = hanning_window.size
            hanning_window = hanning_window.reshape(han_sh)
        if (len(_ref.shape) > 1):
            han_sh = [1] * len(_ref.shape)
            han_sh[proc_dim_ref] = hanning_window_ref.size
            hanning_window_ref = hanning_window_ref.reshape(han_sh)

    # We need to determine a shape to which the out_data_num array will be broadcasted to 
    # allow dividing all spectra. bs is this shape
    if (ind_nonzero is not None):
        bs = [1]*out_data.ndim
        bs[proc_dim_out] = len(index_nonzero)
        bs= tuple(bs)
        if (aps_calc):
            bs_apsd = [1]*d.data.ndim
            bs_apsd[proc_dim] = len(index_nonzero)
            bs_apsd_ref = [1]*_ref.data.ndim
            bs_apsd_ref[proc_dim_ref] = len(index_nonzero)
    else:
        bs = [1]*out_data.ndim
        bs[proc_dim_out] = out_data.shape[proc_dim_out]
        bs = tuple(bs)
        if (aps_calc):
            bs_apsd = [1]*d.data.ndim
            bs_apsd[proc_dim] = out_data.shape[proc_dim_out]
            bs_apsd_ref = [1]*_ref.data.ndim
            bs_apsd_ref[proc_dim_ref] = out_data.shape[proc_dim_out]

    for i_int in range(n_proc_int):
        # Setting the index range of the actual interval
        ind_proc[proc_dim] = slice(index_int_low[i_int], index_int_high[i_int] + 1)
        ind_proc_ref[proc_dim_ref] = slice(index_int_low[i_int], index_int_high[i_int] + 1)
        # Getting the data for processing, this might be multi-dim
        data_proc = copy.deepcopy(d.data[tuple(ind_proc)])
        data_proc_ref = copy.deepcopy(_ref.data[tuple(ind_proc_ref)])
        if (trend is not None):
            try:
                _trend_removal(data_proc,proc_dim,trend)
                _trend_removal(data_proc_ref,proc_dim_ref,trend)
            except Exception as e:
                raise e
        if (hanning):
            data_proc = (data_proc * hanning_window).astype(data_proc.dtype)
            data_proc_ref = (data_proc_ref * hanning_window_ref).astype(data_proc_ref.dtype)
        # Calculating FFT
        dfft = np.fft.fft(data_proc,axis=proc_dim)
        dfft_ref = np.fft.fft(data_proc_ref,axis=proc_dim_ref)
        dps, axis_source, axis_number = flap.tools.multiply_along_axes(dfft, 
                                                                       dfft_ref.conjugate(), 
                                                                       [proc_dim, proc_dim_ref])
        if (aps_calc):
            dfft_aps = (dfft * dfft.conjugate()).real
            dfft_aps_ref = (dfft_ref * dfft_ref.conjugate()).real
        # Rearranging the negative frequencies
        if (ind_in2 is not None):
            dps1 = np.empty(dps.shape,dtype=dps.dtype)
            dps1[tuple(ind_out1)] = dps[tuple(ind_in1)]
            dps1[tuple(ind_out2)] = dps[tuple(ind_in2)]
            dps = dps1
            if (aps_calc):
                dfft_aps1 = np.empty(dfft_aps.shape,dtype=dfft_aps.dtype)   #THIS USED TO BE dfft_aps1 WHICH IS UNDEFINED
                dfft_aps1[tuple(ind_out1_apsd)] = dfft_aps[tuple(ind_in1_apsd)]
                dfft_aps1[tuple(ind_out2_apsd)] = dfft_aps[tuple(ind_in2_apsd)]
                dfft_aps = dfft_aps1
                dfft_aps1 = np.empty(dfft_aps_ref.shape,dtype=dfft_aps_ref.dtype)
                dfft_aps1[tuple(ind_out1_apsd_ref)] = dfft_aps_ref[tuple(ind_in1_apsd_ref)]
                dfft_aps1[tuple(ind_out2_apsd_ref)] = dfft_aps_ref[tuple(ind_in2_apsd_ref)]
                dfft_aps_ref = dfft_aps1
        else:
            dps = dps[tuple(ind_in1)]
            if (aps_calc):
                dfft_aps = dfft_aps[tuple(ind_in1_apsd)]
                dfft_aps_ref = dfft_aps_ref[tuple(ind_in1_apsd_ref)]
        # Cutting the range
        if ((ind_slice) is not None):
            dps = dps[tuple(ind_slice)]
            if (aps_calc):
                dfft_aps = dfft_aps[tuple(ind_slice_apsd)]
                dfft_aps_ref = dfft_aps_ref[tuple(ind_slice_apsd_ref)]
        # Binning the spectrum and summing up the time intervals
        if (ind_bin is not None):
            out_data_interval = np.zeros(tuple(out_shape), dtype=complex)
            np.add.at(out_data_interval, tuple(ind_bin), dps)
            if (aps_calc):
                apsd_interval = np.zeros(tuple(apsd_shape), dtype=float)
                np.add.at(apsd_interval, tuple(ind_bin_apsd), dfft_aps)
                apsd_ref_interval = np.zeros(tuple(apsd_ref_shape), dtype=float)
                np.add.at(apsd_ref_interval, tuple(ind_bin_apsd_ref), dfft_aps_ref)
        else:
            out_data_interval = dps
            if (aps_calc):
                apsd_interval  = dfft_aps
                apsd_ref_interval = dfft_aps_ref
        # Dividing by the number of points in each bin
        if (ind_nonzero is not None):
            out_data_interval[tuple(ind_nonzero)] /= out_data_num[index_nonzero].reshape(bs)
            if (aps_calc):
                apsd_interval[tuple(ind_nonzero_apsd)] /= out_data_num[index_nonzero].reshape(bs_apsd)
                apsd_ref_interval[tuple(ind_nonzero_apsd_ref)] /= out_data_num[index_nonzero].reshape(bs_apsd_ref)
        else:
            out_data_interval /= out_data_num.reshape(bs)
            apsd_interval /= out_data_num.reshape(bs_apsd)
            apsd_ref_interval /= out_data_num.reshape(bs_apsd_ref)
        out_data_interval /= interval_sample_n
        out_data += out_data_interval
        if (aps_calc):
            apsd_interval /= interval_sample_n
            apsd_ref_interval /= interval_sample_n
            apsd += apsd_interval
            apsd_ref += apsd_ref_interval
                        
    out_data /= n_proc_int
    if (aps_calc):
       apsd_norm, axis_source, axis_number = flap.tools.multiply_along_axes(apsd, 
                                                                            apsd_ref, 
                                                                            [proc_dim, proc_dim_ref])
       apsd_norm /= n_proc_int ** 2
    if (norm): 
       if (ind_nonzero is not None):
           out_data[tuple(ind_nonzero)] /= np.sqrt(apsd_norm[tuple(ind_nonzero)])
       else:
           out_data /= np.sqrt(apsd_norm)
    # If there are frequency bins without data setting them to np.NaN
    if (ind_nonzero is not None):
        out_data[tuple(ind_zero)] = np.NaN
    # Putting significance into error
    error = np.full(tuple(out_shape), np.NaN, dtype = float)
    error_arr = np.full(out_data_num.shape, np.NaN, dtype=float)
    if (ind_nonzero is not None):
        error_arr[index_nonzero] = 1./np.sqrt(out_data_num[index_nonzero] * n_proc_int)
    else:
        error_arr = 1./np.sqrt(out_data_num * n_proc_int)
    es = [1] * len(out_shape)
    es[proc_dim] = out_shape[proc_dim]
    error[:] = error_arr.reshape(tuple(es))
    if (not norm and error_calc):
        error *= apsd_norm
      
    #Assembling the coordinates
    coord_list = []
    for c in d.coordinates:
        if (c.unit.name == coordinate):
            continue
        try:
            c.dimension_list.index(coord_obj.dimension_list[0])
            continue
        except ValueError:
            # If this coordinate has no common dimension with the processing
            # then it will stay
            coord_list.append(copy.deepcopy(c))
            # Has to update the dimension list
            for idim in range(len(c.dimension_list)):
                for i in range(len(axis_number)):
                    if ((axis_source[i] == 0) 
                          and (axis_number[i] == c.dimension_list[idim])):
                        coord_list[-1].dimension_list[idim] = i
                        break
                else:
                    raise RuntimeError("Internal error, cannot find dimension mapping (1).")
    for c in _ref.coordinates:
        if (c.unit.name == coordinate):
            continue
        try:
            c.dimension_list.index(ref_coord_obj.dimension_list[0])
            continue
        except ValueError:
            # If this coordinate has no common dimension with the processing
            # then it will stay
            coord_list.append(copy.deepcopy(c))
            # Has to update the dimension list
            for idim in range(len(c.dimension_list)):
                for i in range(len(axis_number)):
                    if ((axis_source[i] == 1) 
                          and (axis_number[i] == c.dimension_list[idim])):
                        coord_list[-1].dimension_list[idim] = i
                        break
                else:
                    raise RuntimeError("Internal error, cannot find dimension mapping (2).")
            # Checking whether there is already an axis with this name
            for c1 in coord_list:
                if (c1.unit.name == c.unit.name):
                    coord_list[-1].unit.name = coord_list[-1].unit.name + ' (Ref)'
    
    # Adding the frequency/wavenumber coordinate            
    if (wavenumber):
        out_name = 'Wavenumber'
        out_unit = '1/'+coord_obj.unit.unit
        res *= 2 * math.pi
        #fcent *= 2* math.pi                                                        #UNUSED VARIABLE
    else:
        out_name = 'Frequency'
        out_unit = 'Hz'
    if (log_scale):
        c = flap.coordinate.Coordinate(name = out_name,
                                       unit = out_unit,
                                       mode = flap.coordinate.CoordinateMode(equidistant=False),
                                       shape = [f_cent.size],
                                       values = f_cent,
                                       dimension_list=[proc_dim])
    else:
        c = flap.coordinate.Coordinate(name = out_name,
                                       unit = out_unit,
                                       mode = flap.coordinate.CoordinateMode(equidistant=True),
                                       shape = [],
                                       start = (fcent_index_range[0] - zero_ind) * res,
                                       step = res,
                                       dimension_list=[proc_dim])
    coord_list.append(c)

    if (norm):
        unit_name = 'Coherency'
    else:
        unit_name = 'Spectral density'
 
    if (d.exp_id == _ref.exp_id):
        exp_id_out = d.exp_id
    else:
        exp_id_out = None

    # We create the new data object with this trick as data_object.py cannot be imported
    d_out = type(d)(data_array = out_data,
                    error = error,
                    coordinates = coord_list,
                    exp_id = exp_id_out,
                    data_unit = flap.coordinate.Unit(unit_name)
                    )
    
    return d_out

def _ccf(d, ref=None, coordinate=None, intervals=None, options=None):
    """
        N dimensional Cross Correlation Function or covariance calculation for the data object d taking d_ref 
        as reference. If ref is not set d is used as reference, that is all CCFs are calculated 
        within d. Calculates all CCF between all signals in ref and d, but not inside d and ref.
        Correlation is calculated along the coordinate(s) listed in coordinate which should be
        identical for the to input data objects.
        Returns a data object with dimension number d.dim+ref.dim-len(coordinate). 
        The coordinates are replaced by coordinate+' lag'.
        The CCF is calculated in multiple intervals (described by intervals)
        and the mean and variance will be returned.
    
        INPUT:
            d: A flap.DataObject.
            ref: Another flap.DataObject
            coordinate: The name of the coordinate (string) along which to calculate CCF or a list of names.
                        Each coordinate should change only along one data dimension and should be equidistant.
                        This and all other cordinates changing along the data dimension of
                        these coordinates will be removed. New coordinates with name+' lag' will be added. 
            intervals: Information of processing intervals.
                       If dictionary with a single key: {selection coordinate: description})
                           Key is a coordinate name which can be different from the calculation
                           coordinate.
                           Description can be flap.Intervals, flap.DataObject or
                           a list of two numbers. If it is a data object with data name identical to
                           the coordinate the error ranges of the data object will be used for
                           interval. If the data name is not the same as coordinate a coordinate with the
                           same name will be searched for in the data object and the value_ranges
                           will be used fromm it to set the intervals.
                       If not a dictionary and not None it is interpreted as the interval
                           description, the selection coordinate is taken the same as
                           (the first) coordinate.
                       If None, the whole data interval will be used as a single interval.
            options: Dictionary. (Keys can be abbreviated)
                'Resolution': Output resolution for each coordinate. (list of values or single value)
                'Range': Output ranges for each coordinate. (List or list of lists)
                'Interval_n': Minimum number of intervals to use for the processing. These are identical
                              length intervals inserted into the input interval list. Default is 8.
                'Error calculation' : True/False. Calculate or not error. Omitting error calculation
                                      increases speed. If Interval_n is 1 no error calculation is done.
                'Trend removal': Trend removal description (see also _trend_removal()). A list, string or None.
                             None: Don't remove trend.
                             Strings:
                               'mean': subtract mean
                             Lists:
                               ['Poly', n]: Fit an n order polynomial to the data and subtract.
                            Trend removal will be applied to each interval separately.
                            At present trend removal can be applied to 1D CCF only.
                'Normalize': Normalize with autocorrelations, that is to calculate correlation instead of 
                             covariance.
                'Correct ACF peak': Switch to correct the photon peak of the ACF during normalization. 
                                    Default: False
                'Correct ACF range': Indices for the ACF peak correction around the peak.
                                     Default: [-2,-1,1,2]
                'Verbose': Print progress messages
    """
    if (d.data is None):
        raise ValueError("Cannot do correlation calculation without data.")

    default_options = {'Resolution': None,
                       'Range': None,
                       'Interval_n': 8,
                       'Trend removal': ['Poly', 2],
                       'Error calculation': True,
                       'Normalize': False,
                       'Correct ACF peak': False,
                       'Correct ACF range':[-2,-1,1,2],
                       'Verbose':True
                       }
    _options = flap.config.merge_options(default_options, options, data_source=d.data_source, section='Correlation')
    if (coordinate is None):
        c_names = d.coordinate_names()
        try:
            c_names.index('Time')
            _coordinate = ['Time']
        except ValueError:
            raise ValueError("No coordinate is given for correlation calculation and no Time coordinate found.")
    else:
        if (type(coordinate) is list):
            _coordinate = coordinate
        else:
            _coordinate = [coordinate]

    trend = _options['Trend removal']
    if (len(_coordinate) != 1 and (trend is not None)):
        raise NotImplementedError("Trend removal for multi-dimensional correlation is not implemented.")
        
    interval_n = _options['Interval_n']
    
    norm = _options['Normalize']
    
    correct_acf_peak=_options['Correct ACF peak']
    correct_acf_range=np.asarray(_options['Correct ACF range'])
    if correct_acf_range.shape[0] < 4:
        raise ValueError("The range of the correlation peak subtraction should have at least 4 elements, e.g., [-2,-1,1,2]")
        
    error_calc = _options['Error calculation']
    if (len(_coordinate) != 1 and (intervals is not None)):
        raise NotImplementedError("Interval selection for multi-dimensional correlation is not implemented.")
    if (len(_coordinate) != 1):
        error_calc = False
        interval_n = 1
        
    # Getting coordinate objects and checking properties    
    coord_obj = []
    # This will contain the dimension list in d.data and the output of the correlations
    correlation_dimensions = []
    for c_name in _coordinate:
        try:
            c = d.get_coordinate_object(c_name)
        except Exception as e:
            raise e      
        coord_obj.append(c)
        if (len(c.dimension_list) != 1):
            raise ValueError("Correlation calculation is possible only along coordinates changing in one dimension.")
        if (not c.mode.equidistant):
            raise ValueError("Correlation calculation is possible only along equidistant coordinates.")
        try:
            correlation_dimensions.index(c.dimension_list[0])
            raise ValueError("Cannot calculate multi dimensional correlation with common dimensions.")
        except ValueError:
            pass
        correlation_dimensions.append(c.dimension_list[0])
        
    if (ref is None):
        _ref = d
        coord_obj_ref = coord_obj
        correlation_dimensions_ref = correlation_dimensions
        try:
            intervals, index_intervals = _spectral_calc_interval_selection(d,
                                                                           None, 
                                                                           _coordinate[0],
                                                                           intervals,
                                                                           interval_n)
        except Exception as e:
            raise e
    else:
        _ref = ref
        coord_obj_ref = []
        correlation_dimensions_ref = []
        for i,c_name in enumerate(_coordinate):
            try:
                c = _ref.get_coordinate_object(c_name)
            except Exception as e:
                raise e      
            coord_obj_ref.append(c)
            if (len(c.dimension_list) != 1):
                raise ValueError("Correlation calculation is possible only along coordinates changing in one dimension.")
            if (not c.mode.equidistant):
                raise ValueError("Correlation calculation is possible only along equidistant coordinates.")
            try:
                correlation_dimensions_ref.index(c.dimension_list[0])
                raise ValueError("Cannot calculate multi dimensional correlation with common dimensions.")
            except ValueError:
                pass
            correlation_dimensions_ref.append(c.dimension_list[0])
            if (math.fabs(c.step[0] - coord_obj[i].step[0]) / math.fabs(c.step[0]) > 1e-4):
                   raise ValueError("Incompatible coordinate step sizes." )
            if (math.fabs(c.start - coord_obj[i].start) > math.fabs(c.step[0])):
                   raise ValueError("Incompatible coordinate start values." )
            if (list(_ref.data.shape)[correlation_dimensions_ref[i]] 
                           != list(d.data.shape)[correlation_dimensions[i]]):
                   raise ValueError("Incompatible data dimensions." )
        try:
            intervals, index_intervals = _spectral_calc_interval_selection(d,
                                                                           _ref, 
                                                                           _coordinate[0],
                                                                           intervals,
                                                                           interval_n)
        except Exception as e:
            raise e
                        
    interval_n, start_ind = intervals.interval_number()
    int_low, int_high = intervals.interval_limits()
    index_int_low, index_int_high = index_intervals.interval_limits()
    # Number of processing intervals
    n_proc_int = len(int_low)
    # Numper of points in the selection coordinate
    n_sample_sel = index_int_high[0] - index_int_low[0]
    
    # Creating indices to take out data for each processing interval and place it into the processing arrays
    interval_slice_1 = [slice(0,dim) for dim in list(d.data.shape)]
    interval_out_slice = copy.deepcopy(interval_slice_1)
    interval_out_slice[correlation_dimensions[0]] = slice(0,n_sample_sel)
    interval_slice = [None] * n_proc_int
    for i in range(n_proc_int):
        interval_slice[i] = copy.deepcopy(interval_slice_1)
    for i in range(n_proc_int):
        interval_slice[i][correlation_dimensions[0]] = slice(index_int_low[i],index_int_high[i])
    if (ref is not None):
        interval_slice_ref_1 = [slice(0,dim) for dim in list(_ref.data.shape)]
        interval_out_slice_ref = copy.deepcopy(interval_slice_ref_1)
        interval_out_slice_ref[correlation_dimensions_ref[0]] = slice(0,n_sample_sel)
        interval_slice_ref = [None] * n_proc_int
        for i in range(n_proc_int):
            interval_slice_ref[i] = copy.deepcopy(interval_slice_ref_1)
        for i in range(n_proc_int):
            interval_slice_ref[i][correlation_dimensions_ref[0]] = slice(index_int_low[i],index_int_high[i])

    # Number of correlation points in the correlation dimensions after FFT before binning 
    corr_point_n_nat = [list(d.data.shape)[dim] for dim in correlation_dimensions]   
    corr_point_n_nat[0] = n_sample_sel
    
    # Setting resolution
    corr_res = _options['Resolution']
    if (corr_res is None):
        corr_res = [c.step[0] for c in coord_obj]
    if (type(corr_res) is not list):
        corr_res = [corr_res] * len(_coordinate)
    corr_range = _options['Range']
    #Setting default correlation range to 10-th of coordinate range 
    if (corr_range is None):
        corr_range = []
        for i,coord in enumerate(coord_obj):
            corr_range.append([-(coord.step[0] * corr_point_n_nat[i]) /10 , 
                               (coord.step[0] * corr_point_n_nat[i]) /10
                               ])
    if (type(corr_range) is not list):
        raise ValueError("Correlation range must be a list.")
    if (type(corr_range[0]) is not list):
        corr_range = [corr_range] * len(_coordinate)
    # Sanity check for lag range
    for i,coord in enumerate(coord_obj):
        dr, dr1 = coord.data_range(data_shape=d.shape)
        if ((abs(corr_range[i][0]) > (dr[1] - dr[0])) or (abs(corr_range[i][1]) > (dr[1] - dr[0]))):
            raise ValueError("Correlation lag calculation range is too large for coordinate '{:s}'.".format(coord.unit.name))
        if ((corr_range[i][1] - corr_range[i][0]) < coord.step[0]*3):
            raise ValueError("Correlation lag calculation range is too small for coordinate '{:s}'.".format(coord.unit.name))
        if (corr_range[i][0] >= corr_range[i][1]):
            raise ValueError("Invalid correlation lag range for coordinate '{:s}'.".format(coord.unit.name))

    # Correlation final resolutions in sample numbers
    corr_res_sample = []
    # Correlation range in final resolution
    range_sampout = []
    # Number of correlation points at final resolution
    n_corr = []
    # The shift range in original sample numbers
    shift_range = []
    # Number of 0 samples to add in each correlation dimensions to prevent correlation from wrapping around
    pad_length = []
    for i,coord in enumerate(coord_obj):
        # Resolutions in sample number. Ensuring that it is odd sample
        res = int(round(corr_res[i] / coord.step[0]))
        if (res % 2 == 0):
            res = res + 1
        corr_res_sample.append(res)
        # Correlation point ranges in output sample number
        if ((type(corr_range[i]) is not list) or (len(corr_range[i]) != 2)):
            raise ValueError("Invalid correlation range description. Must be 2-element list.")
        r_samp = [(corr_range[i][k] / (c.step[0] * corr_res_sample[-1])) for k in range(2)]
        r_samp = [int(round(r_samp[0])), int(round(r_samp[1]))]
        range_sampout.append(r_samp)
        n_corr.append(r_samp[1] - r_samp[0] + 1)
        shift_range.append([r_samp[0] * corr_res_sample[-1] - int(corr_res_sample[-1] / 2),
                            r_samp[1] * corr_res_sample[-1] + int(corr_res_sample[-1] / 2)])
        pad_length.append(max([abs(shift_range[-1][0]), abs(shift_range[-1][1])]))
        corr_point_n_nat[i] += pad_length[i]
        
    # Check if we have lag 0 data in  all dimensions
    lag0_present = True
    for i in range(len(correlation_dimensions)):
        if ((range_sampout[i][0] > 0) or (range_sampout[i][1] < 0)):
            lag0_present = False
    
    # This flag indicates whether we need to calculate ACFs separately
    calc_acf = not lag0_present or (ref is not None) and _options['Normalize']
                
    # Determining the output shape. First the dimensions of d without the common dimensions, 
    # then the common dimensions, then the reference with the calculation dimensions removed
    out_shape = []
    correlation_dimensions_out = []
    out_shape_acf = []
    out_shape_acf_ref = []
    
    for i in range(len(d.data.shape)):
        try:
            correlation_dimensions.index(i)
        except ValueError:
            pass
            out_shape.append(d.data.shape[i])
            out_shape_acf.append(d.data.shape[i])
    for i in range(len(correlation_dimensions)):
        out_shape.append(n_corr[i])
        out_shape_acf.append(n_corr[i])
        out_shape_acf_ref.append(n_corr[i])
        correlation_dimensions_out.append(len(out_shape)-1)
    for i in range(len(_ref.data.shape)):
        try:
            correlation_dimensions_ref.index(i)
        except ValueError:
            pass
            out_shape.append(_ref.data.shape[i])
            out_shape_acf_ref.append(_ref.data.shape[i])

    # Array shapes for doing FFT
    proc_shape = list(d.data.shape)
    if (ref is not None):
        proc_shape_ref = list(_ref.data.shape)
        
    pad_slice = [slice(0,ds) for ds in d.data.shape]
    for i,dim in enumerate(correlation_dimensions):
        pad_slice[dim] = slice(corr_point_n_nat[i]-pad_length[i],corr_point_n_nat[i])
        proc_shape[dim] = corr_point_n_nat[i] + pad_length[i]

    if (ref is not None):
        pad_slice_ref = [slice(0,ds) for ds in _ref.data.shape]
        for i,dim in enumerate(correlation_dimensions_ref):
            pad_slice_ref[dim] = slice(corr_point_n_nat[i]-pad_length[i],corr_point_n_nat[i]) 
            proc_shape_ref[dim] = corr_point_n_nat[i] + pad_length[i]
    # Index arrays to rearrange after FFT and multiplying
    ind_in1 = [slice(0,d) for d in out_shape]
    ind_in2 = copy.deepcopy(ind_in1)
    ind_out1 = copy.deepcopy(ind_in1)
    ind_out2 = copy.deepcopy(ind_in1)
    if (calc_acf):
        # We need to calculate APSDs separately
        ind_in1_acf = [slice(0,d) for d in out_shape_acf]
        ind_in2_acf = copy.deepcopy(ind_in1_acf)
        ind_out1_acf = copy.deepcopy(ind_in1_acf)
        ind_out2_acf = copy.deepcopy(ind_in1_acf)
        if (ref is not None):
            ind_in1_acf_ref = [slice(0,d) for d in out_shape_acf_ref]
            ind_in2_acf_ref = copy.deepcopy(ind_in1_acf_ref)
            ind_out1_acf_ref = copy.deepcopy(ind_in1_acf_ref)
            ind_out2_acf_ref = copy.deepcopy(ind_in1_acf_ref)
            
    # zero_ind is the index where the 0 time lag will be after rearranging the CCF to final form
    zero_ind = [0] * len(correlation_dimensions)
    # A slicing list which will cut out the needed part of the CCF
    # The CCF will have the remaining dimensions of d first, then the correlation dimensions, 
    # then the remaining dimensions of ref
    ind_slice = [slice(0,d) for d in out_shape]
    ind_bin = copy.deepcopy(ind_slice)
    if (calc_acf):
        ind_slice_acf = [slice(0,dim) for dim in out_shape_acf]
        ind_bin_acf = copy.deepcopy(ind_slice_acf)
        if (ref is not None):
            ind_slice_acf_ref = [slice(0,dim) for dim in out_shape_acf_ref]
            ind_bin_acf_ref = copy.deepcopy(ind_slice_acf_ref)
    for i,cd in enumerate(correlation_dimensions_out):
        nfft = corr_point_n_nat[i] + pad_length[i]
        ind_in1[cd] = slice(0,int(nfft / 2))
        ind_out1[cd] = slice(nfft-int(nfft / 2), nfft)
        
        zero_ind[i] = nfft - int(nfft / 2)
        ind_in2[cd] = slice(int(nfft / 2),nfft)
        ind_out2[cd] = slice(0, nfft - int(nfft / 2))
        
        ind_slice[cd] = slice(shift_range[i][0] + zero_ind[i], shift_range[i][1] + zero_ind[i]+1)
        ind_bin[cd] = np.arange(ind_slice[cd].stop - ind_slice[cd].start,dtype=np.int32) // corr_res_sample[i]
        
        if (calc_acf):
            ind_in1_acf[cd] = ind_in1[cd]
            ind_out1_acf[cd] = ind_out1[cd]
            ind_in2_acf[cd] = ind_in2[cd]
            ind_out2_acf[cd] = ind_out2[cd]
            ind_slice_acf[cd] = ind_slice[cd]
            ind_bin_acf[cd]  = ind_bin[cd]
            if (ref is not None):
                ind_in1_acf_ref[i] = ind_in1[cd]
                ind_out1_acf_ref[i] = ind_out1[cd]
                ind_in2_acf_ref[i] = ind_in2[cd]
                ind_out2_acf_ref[i] = ind_out2[cd]
                ind_slice_acf_ref[i] = ind_slice[cd]
                ind_bin_acf_ref[i]  = ind_bin[cd]
                
    # The processing array for each interval, data will be read here                
    proc_array = np.zeros(tuple(proc_shape),dtype=d.data.dtype)
    if (ref is not None):
        proc_array_ref = np.zeros(tuple(proc_shape_ref),dtype=_ref.data.dtype)
    # The output array    
    if ((d.data.dtype.kind != 'c') and (_ref.data.dtype.kind != 'c')):
        out_dtype = float
    else:
        out_dtype= complex
    out_corr = np.zeros(tuple(out_shape), dtype=out_dtype)
    if (error_calc):
        out_corr_square = np.zeros(tuple(out_shape), dtype=out_dtype)  
    all_points = 1
    for i in range(len(correlation_dimensions)):
        all_points *= n_corr[i]
    for i_int in range(n_proc_int):
        if (_options['Verbose']):
            print("Interval {:d}/{:d}".format(i_int+1, n_proc_int))
        # Taking data for this processing interval
        if (trend is not None):
            proc_array_trend = copy.deepcopy(d.data[tuple(interval_slice[i_int])])
            try:
                _trend_removal(proc_array_trend,correlation_dimensions[0],trend)
            except Exception as e:
                raise e
            proc_array[tuple(interval_out_slice)] = proc_array_trend 
        else:
            proc_array[tuple(interval_out_slice)] = d.data[tuple(interval_slice[i_int])]
        proc_array[tuple(pad_slice)] = 0
        fft = np.fft.fftn(proc_array,axes=correlation_dimensions)
        if (ref is not None):
            if (trend is not None):
                proc_array_trend_ref = copy.deepcopy(_ref.data[tuple(interval_slice_ref[i_int])])
                try:
                    _trend_removal(proc_array_trend_ref,correlation_dimensions_ref[0],trend)
                except Exception as e:
                    raise e
                proc_array_ref[tuple(interval_out_slice_ref)] = proc_array_trend_ref 
            else:
                proc_array_ref[tuple(interval_out_slice_ref)] = _ref.data[tuple(interval_slice_ref[i_int])]
            proc_array_ref[tuple(pad_slice_ref)] = 0
            fft_ref = np.fft.fftn(proc_array_ref,axes=correlation_dimensions_ref)
        else:
            fft_ref = fft
        res, axis_source, axis_number = flap.multiply_along_axes(fft, 
                                                                 np.conj(fft_ref),
                                                                 [correlation_dimensions,correlation_dimensions_ref],
                                                                 keep_a1_dims = False
                                                                 )
        corr_dim_start = len(d.data.shape) - len(correlation_dimensions)
        cps_corr_dims = np.arange(len(correlation_dimensions),dtype=int) + corr_dim_start 
        res = np.fft.ifftn(res,axes=cps_corr_dims)
        if (out_dtype is float):
            res = np.real(res)        

        if len(correlation_dimensions) == 2:
            # This part of the code rearranges the corners of the 2D CCF function
            # so the maximum is going to be in the middle. It does this in a generalized way
            # where the non-correlating dimensions are left alone.
            
            corr=flap.tools.reorder_2d_ccf_indices(res,
                                                   correlation_dimensions, 
                                                   ind_in1, 
                                                   ind_in2,
                                                   ind_out1,
                                                   ind_out2)

        else:
            corr = np.empty(res.shape,dtype=res.dtype)
            corr[tuple(ind_out1)] = res[tuple(ind_in1)]
            corr[tuple(ind_out2)] = res[tuple(ind_in2)]
            
        corr_sliced = corr[tuple(ind_slice)]
        corr_binned = np.zeros(tuple(out_shape),dtype=res.dtype)
        if corr_binned.shape != corr_binned.shape:
            np.add.at(corr_binned,tuple(ind_bin),corr_sliced)
        else:
            corr_binned=corr_sliced
        if (norm):
            zero_ind_out = [0] * len(correlation_dimensions)
            for i in range(len(correlation_dimensions)):
                zero_ind_out[i] = 0 - range_sampout[i][0]
            if (not calc_acf):
                # We already have the autocorrelations
                if (len(d.data.shape) == len(correlation_dimensions)):
                    # There is a single correlation function
                    corr_binned /= corr_binned[zero_ind_out[0]]
                else:
                    corr_dimension_start = len(d.data.shape)-len(correlation_dimensions)
                    autocorr_index_shape = copy.deepcopy(out_shape[:corr_dimension_start])
                    ind_autocorr = [0]*len(out_shape)
                    for i in range(len(autocorr_index_shape)):
                        ind = np.arange(out_shape[i],dtype=int)
                        temp_shape = [1] * len(autocorr_index_shape)
                        temp_shape[i] = autocorr_index_shape[i]
                        ind = ind.reshape(tuple(temp_shape))
                        tile_shape = copy.deepcopy(autocorr_index_shape)
                        tile_shape[i] = 1
                        ind_autocorr[i] = np.tile(ind,tile_shape)
                    for i in range(len(correlation_dimensions)):
                        ind_autocorr[corr_dimension_start + i] = np.full(tuple(autocorr_index_shape),zero_ind_out[i])
                    ind_autocorr[len(autocorr_index_shape)+len(correlation_dimensions):] \
                           = copy.deepcopy(ind_autocorr[0:len(autocorr_index_shape)])     
                    autocorr_mx = corr_binned[tuple(ind_autocorr)]
                    extend_shape = [1] * (len(out_corr.shape) - len(autocorr_mx.shape))
                    corr_binned /= np.sqrt(np.reshape(autocorr_mx, tuple(list(autocorr_mx.shape) + extend_shape)))
                    corr_binned /= np.sqrt(np.reshape(autocorr_mx, tuple(extend_shape + list(autocorr_mx.shape))))
            else:
                # We do not have the autocorrelations, calculating
                res_acf = np.fft.ifftn(fft * np.conj(fft),axes=correlation_dimensions)
                
                if (out_dtype is float):
                    res_acf = np.real(res_acf)
                # we need to move the correlation axes to the end to be consistent with the CCF
                res_acf, ax_map = flap.tools.move_axes_to_end(res_acf,correlation_dimensions)

                if len(correlation_dimensions) == 2:
                    # This part of the code rearranges the corners of the 2D CCF function
                    # so the maximum is going to be in the middle. It does this in a generalized way
                    # where the non-correlating dimensions are left alone.
                    
                    corr_acf=flap.tools.reorder_2d_ccf_indices(res_acf,
                                                               correlation_dimensions, 
                                                               ind_in1_acf, 
                                                               ind_in2_acf,
                                                               ind_out1_acf,
                                                               ind_out2_acf)
                else:
                    corr_acf = np.empty(res_acf.shape,dtype=res.dtype)
                    corr_acf[tuple(ind_out1_acf)] = res_acf[tuple(ind_in1_acf)]
                    corr_acf[tuple(ind_out2_acf)] = res_acf[tuple(ind_in2_acf)]

                corr_sliced_acf = corr_acf[tuple(ind_slice_acf)]
                corr_binned_acf = np.zeros(tuple(out_shape_acf),dtype=res_acf.dtype)
                if not corr_binned_acf.shape == corr_sliced_acf.shape:
                    np.add.at(corr_binned_acf,tuple(ind_bin_acf),corr_sliced_acf)
                else:
                    corr_binned_acf=corr_sliced_acf
                # We need to take the zero lag elements from each correlation dimension
                corr_dimension_start = len(out_shape_acf)-len(correlation_dimensions)
                if correct_acf_peak:
                #if False:
                    
                    if corr_dimension_start == 0: #Every dimension is correlation with a lag dimension
                        array_2b_corrected=copy.deepcopy(corr_binned_acf)
                        peak_value=0.
                        for i in range(len(correlation_dimensions)):
                            if zero_ind_out[i]-2 < 0:
                                raise ValueError("Not enough datapoints for ACF correction.")
                            # ind_correction = np.arange(zero_ind_out[i]-2,zero_ind_out[i]+3)
                            ind_correction=correct_acf_range+zero_ind_out[i]
                            array_2b_corrected = np.take(array_2b_corrected,ind_correction,axis=corr_dimension_start+i)
                            
                        for i in range(len(correlation_dimensions)):
                            # coeff = np.polyfit([-2,-1,1,2],
                            #                    np.take(array_2b_corrected, 2, axis=corr_dimension_start+i)[np.asarray([0,1,3,4])],
                            #                    2)
                            coeff = np.polyfit(correct_acf_range,
                                                np.take(array_2b_corrected, 2, axis=corr_dimension_start+i),
                                                2)                            
                            
                            peak_value += coeff[2]-coeff[1]**2/(4*coeff[0])
                        peak_value /= (i+1)
                        corr_binned_acf[zero_ind_out]=peak_value
                        
                    elif corr_dimension_start == 1: #The first dimension is not correlation, but the rest is.
                        for i_non_corr in range(corr_binned_acf.shape[0]):
                            array_2b_corrected=np.take(corr_binned_acf,i_non_corr,axis=0)
                            peak_value=0.
                            for i in range(len(correlation_dimensions)):
                                if zero_ind_out[i]-2 < 0:
                                    raise ValueError("Not enough datapoints for ACF correction.")
                                # ind_correction = np.arange(zero_ind_out[i]-2,zero_ind_out[i]+3)
                                ind_correction=correct_acf_range+zero_ind_out[i]
                                array_2b_corrected = np.take(array_2b_corrected,ind_correction,axis=i)
                            if len(correlation_dimensions) == 1:
                                for i in range(len(correlation_dimensions)):
                                    # coeff = np.polyfit([-2,-1,1,2],
                                    #                    array_2b_corrected[np.asarray([0,1,3,4])],
                                    #                    2)
                                    coeff = np.polyfit(correct_acf_range,
                                                        array_2b_corrected,
                                                        2)   
                                    
                                    peak_value += coeff[2]-coeff[1]**2/(4*coeff[0])
                                peak_value /= (i+1)
                                corr_binned_acf[i_non_corr,zero_ind_out]=peak_value
                            elif len(correlation_dimensions) == 2:
                                for i in range(len(correlation_dimensions)):
                                    # coeff = np.polyfit([-2,-1,1,2],
                                    #                    np.take(array_2b_corrected, 2, axis=i)[np.asarray([0,1,3,4])],
                                    #                    2)
                                    coeff = np.polyfit(correct_acf_range,
                                                       np.take(array_2b_corrected, 2, axis=i),
                                                       2)
                                    
                                    peak_value += coeff[2]-coeff[1]**2/(4*coeff[0])
                                peak_value /= (i+1)
                                corr_binned_acf[i_non_corr,zero_ind_out]=peak_value
                            elif len(correlation_dimensions) == 3:
                                
                                raise NotImplementedError('3D correlation dimensions has not been implemented yet.')
                                
                    elif corr_dimension_start == 2:
                        for i_non_corr_1 in range(corr_binned_acf.shape[0]):
                            for i_non_corr_2 in range(corr_binned_acf.shape[1]):
                                array_2b_corrected=np.take(corr_binned_acf,i_non_corr_1,axis=0)
                                array_2b_corrected=np.take(array_2b_corrected,i_non_corr_2,axis=0)
                                peak_value=0.
                                for i in range(len(correlation_dimensions)):
                                    if zero_ind_out[i]-2 < 0:
                                        raise ValueError("Not enough datapoints for ACF correction.")
                                    #ind_correction = np.arange(zero_ind_out[i]-2,zero_ind_out[i]+3)
                                    ind_correction=correct_acf_range+zero_ind_out[i]
                                    array_2b_corrected = np.take(array_2b_corrected,ind_correction,axis=i)
                                    
                                for i in range(len(correlation_dimensions)):
                                    # coeff = np.polyfit([-2,-1,1,2],
                                    #                    np.take(array_2b_corrected, 2, axis=i)[np.asarray([0,1,3,4])],
                                    #                    2)
                                    coeff = np.polyfit(correct_acf_range,
                                                       np.take(array_2b_corrected, 2, axis=i),
                                                       2)
                                    
                                    peak_value += coeff[2]-coeff[1]**2/(4*coeff[0])
                                peak_value /= (i+1)
                                corr_binned_acf[i_non_corr_1,i_non_corr_2,zero_ind_out]=peak_value
                        
                    elif corr_dimension_start == 3:
                        for i_non_corr_1 in range(corr_binned_acf.shape[0]):
                            for i_non_corr_2 in range(corr_binned_acf.shape[1]):
                                for i_non_corr_3 in range(corr_binned_acf.shape[2]):
                            
                                    array_2b_corrected=np.take(corr_binned_acf,i_non_corr_1,axis=0)
                                    array_2b_corrected=np.take(array_2b_corrected,i_non_corr_2,axis=0)
                                    array_2b_corrected=np.take(array_2b_corrected,i_non_corr_3,axis=0)
                                    peak_value=0.
                                    for i in range(len(correlation_dimensions)):
                                        if zero_ind_out[i]-2 < 0:
                                            raise ValueError("Not enough datapoints for ACF correction.")
                                        #ind_correction = np.arange(zero_ind_out[i]-2,zero_ind_out[i]+3)
                                        ind_correction=correct_acf_range+zero_ind_out[i]
                                        array_2b_corrected = np.take(array_2b_corrected,ind_correction,axis=i)
                                        
                                    for i in range(len(correlation_dimensions)):
                                        # coeff = np.polyfit([-2,-1,1,2],
                                        #                    np.take(array_2b_corrected, 2, axis=i)[np.asarray([0,1,3,4])],
                                        #                    2)
                                        coeff = np.polyfit(correct_acf_range,
                                                           np.take(array_2b_corrected, 2, axis=i),
                                                           2)
                                        
                                        peak_value += coeff[2]-coeff[1]**2/(4*coeff[0])
                                    peak_value /= (i+1)
                                    corr_binned_acf[i_non_corr_1,i_non_corr_2,i_non_corr_3,zero_ind_out]=peak_value

                    
                # We need to take the zero lag elements from each correlation dimension
                for i in range(len(correlation_dimensions)):
                     # Always the cor_dimension_start axis is taken as it is removed by take
                    corr_binned_acf = np.take(corr_binned_acf,zero_ind_out[i],axis=corr_dimension_start)
                    
                if (ref is not None):
                    res_acf_ref = np.fft.ifftn(fft_ref * np.conj(fft_ref),axes=correlation_dimensions_ref)
                    if (out_dtype is float):
                        res_acf_ref = np.real(res_acf_ref)
                    # we need to move the correlation axes to the start to be consistent with the CCF
                    if len(_coordinate) != 2:
                        res_acf_ref,ax_map_ref = flap.tools.move_axes_to_end(res_acf_ref,correlation_dimensions_ref)
                        
                    
                    if len(correlation_dimensions) == 2:
                        corr_acf_ref=flap.tools.reorder_2d_ccf_indices(res_acf_ref,
                                                                       correlation_dimensions, 
                                                                       ind_in1_acf_ref, 
                                                                       ind_in2_acf_ref,
                                                                       ind_out1_acf_ref,
                                                                       ind_out2_acf_ref)
                    else:
                        corr_acf_ref = np.empty(res_acf_ref.shape,dtype=res_acf.dtype)
                        corr_acf_ref[tuple(ind_out1_acf_ref)] = res_acf_ref[tuple(ind_in1_acf_ref)]
                        corr_acf_ref[tuple(ind_out2_acf_ref)] = res_acf_ref[tuple(ind_in2_acf_ref)]
                    
                    corr_sliced_acf_ref = corr_acf_ref[tuple(ind_slice_acf_ref)]
                    corr_binned_acf_ref = np.zeros(tuple(out_shape_acf_ref),dtype=res_acf_ref.dtype)
                    
                    if not corr_binned_acf_ref.shape == corr_sliced_acf_ref.shape:
                        np.add.at(corr_binned_acf_ref,tuple(ind_bin_acf_ref),corr_sliced_acf_ref)
                    else:
                        corr_binned_acf_ref=corr_sliced_acf_ref

                    if correct_acf_peak:

                        array_2b_corrected=copy.deepcopy(corr_binned_acf_ref)

                        peak_value=0.
                        for i in range(len(correlation_dimensions_ref)):
                            if zero_ind_out[i]-2 < 0:
                                raise ValueError("Not enough datapoints for ACF correction.")
                            #ind_correction = np.arange(zero_ind_out[i]-2,zero_ind_out[i]+3)
                            ind_correction=correct_acf_range+zero_ind_out[i]
                            array_2b_corrected = np.take(array_2b_corrected,ind_correction,axis=i)
                        if len(correlation_dimensions) == 1:
                            for i in range(len(correlation_dimensions)):
                                # coeff = np.polyfit([-2,-1,1,2],
                                #                    array_2b_corrected[np.asarray([0,1,3,4])],
                                #                    2)
                                coeff = np.polyfit(correct_acf_range,
                                                   array_2b_corrected,
                                                   2)
                                
                                peak_value += coeff[2]-coeff[1]**2/(4*coeff[0])
                            peak_value /= (i+1)
                            corr_binned_acf_ref[zero_ind_out]=peak_value
                        elif len(correlation_dimensions) == 2:
                            for i in range(len(correlation_dimensions)):
                                # coeff = np.polyfit([-2,-1,1,2],
                                #                    np.take(array_2b_corrected, 2, axis=i)[np.asarray([0,1,3,4])],
                                #                    2)
                                coeff = np.polyfit(correct_acf_range,
                                                   np.take(array_2b_corrected, 2, axis=i),
                                                   2)
                                
                                peak_value += coeff[2]-coeff[1]**2/(4*coeff[0])
                            peak_value /= (i+1)
                            corr_binned_acf_ref[zero_ind_out]=peak_value
                        elif len(correlation_dimensions) == 3:
                            
                            raise NotImplementedError('3D correlation dimensions has not been implemented yet.')
                            
                    # Always the cor_dimension_start axis is taken as it is removed by take
                    # We need to take the zero lag elements from each correlation dimension
                    for i in range(len(correlation_dimensions)):
                        corr_binned_acf_ref = np.take(corr_binned_acf_ref,zero_ind_out[i],axis=0)
                            
                    # for i in range(len(correlation_dimensions)):
                    #     corr_binned_acf_ref = np.take(corr_binned_acf_ref,zero_ind_out[i],axis=0)
                else:
                    corr_binned_acf_ref = corr_binned_acf

                extend_shape = [1] * (len(out_corr.shape) - len(corr_binned_acf.shape))
                corr_binned /= np.sqrt(np.reshape(corr_binned_acf,tuple(list(corr_binned_acf.shape) + extend_shape)))
                extend_shape = [1] * (len(out_corr.shape) - len(corr_binned_acf_ref.shape))
                corr_binned /= np.sqrt(np.reshape(corr_binned_acf_ref,tuple(extend_shape + list(corr_binned_acf_ref.shape))))
        else:
            corr_binned /= all_points
        out_corr += corr_binned
        if (error_calc):   
            out_corr_square += corr_binned ** 2
    out_corr /= n_proc_int
    if (error_calc):
        out_error = np.sqrt(np.clip((out_corr_square / n_proc_int - out_corr ** 2),
                                  0,None)) / math.sqrt(n_proc_int) 
    else:
        out_error = None
    
    #Assembling the coordinates
    # Removing all coordinates which have common dimensions with the correlation coordinates
    coord_list = []
    for c in d.coordinates:
        for dim in c.dimension_list:
            try:
                correlation_dimensions.index(dim)
                break
            except ValueError:
                pass
        else:
            coord_list.append(copy.deepcopy(c))
            for i_dim, dim in enumerate(coord_list[-1].dimension_list):
                for i in range(len(axis_source)):
                    if ((axis_source[i] == 0) and (axis_number[i] == dim)):
                       coord_list[-1].dimension_list[i_dim] = i

    for c in _ref.coordinates:
        for dim in c.dimension_list:
            try:
                correlation_dimensions_ref.index(dim)
                break
            except ValueError:
                pass
        else:
            coord_list.append(copy.deepcopy(c))
            for i_dim, dim in enumerate(coord_list[-1].dimension_list):
                for i in range(len(axis_source)):
                    if ((axis_source[i] == 1) and (axis_number[i] == dim)):
                       coord_list[-1].dimension_list[i_dim] = i
            # Checking whether there is already an axis with this name
            for c1 in coord_list:
                if (c1.unit.name == c.unit.name):
                    coord_list[-1].unit.name = coord_list[-1].unit.name + ' (Ref)'
    
    # Adding the correlation coordinates
    for i,c in enumerate(coord_obj):           
        c_new = flap.coordinate.Coordinate(name = c.unit.name + ' lag',
                                           unit = c.unit.unit,
                                           mode = flap.coordinate.CoordinateMode(equidistant=True),
                                           shape = [],
                                           start = range_sampout[i][0] * corr_res_sample[i] * c.step[0],
                                           step = [corr_res_sample[i] * c.step[0]],
                                           dimension_list=[len(d.data.shape) - len(correlation_dimensions) + i])
        coord_list.append(c_new)
    if (norm):
        unit_name = 'Correlation'
    else:
        unit_name = 'Covariance'
 
    if (d.exp_id == _ref.exp_id):
        exp_id_out = d.exp_id
    else:
        exp_id_out = None

    # We create the new data object with this trick as data_object.py cannot be imported
    d_out = type(d)(data_array = out_corr,
                    error = out_error,
                    coordinates = coord_list,
                    exp_id = exp_id_out,
                    data_unit = flap.coordinate.Unit(unit_name)
                    )
    
    return d_out