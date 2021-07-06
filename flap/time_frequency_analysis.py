from scipy import signal
import numpy as np
import flap.config
import flap.coordinate


def stft(d, coordinate=None, window='hann', nperseg=256, noverlap=None, nfft=None,
         detrend=False, return_onesided=True, boundary='zeros', padded=True):
    """
    calculates the STFT of the data in a dataobject using scipy's stft method

    INPUT:
            d: A flap.DataObject.
            coordinate: The name of the coordinate (string) along which to calculate STFT.
                        This coordinate should change only along one data dimension and should be equidistant.
                        This and all other cordinates changing along the data dimension of
                        this coordinate will be removed. A new coordinate with name
                        Frequency will be added. The unit will be
                        derived from the unit of the coordinate (e.g., Hz cm-1, m-1)
            all other options:    Options of STFT, each will be given to scipy.signal.stft

    """
    if (noverlap is None):
        noverlap = nperseg // 4 * 3  # the default setting for stft - it should contain 4x the information

    if (d.data is None):
        raise ValueError("Cannot do spectral analysis without data.")
    if (coordinate is None):
        c_names = d.coordinate_names()
        try:
            c_names.index('Time')  # this will fail if Time coordinate doesn't exist
            _coordinate = 'Time'
        except ValueError:
            raise ValueError("No coordinate is given for spectrum calculation and no Time coordinate found.")
    else:
        _coordinate = coordinate
    try:
        coord_obj = d.get_coordinate_object(_coordinate)
    except Exception as e:
        raise e

    if (len(coord_obj.dimension_list) != 1):
        raise ValueError("STFT calculation is possible only along coordinates changing along one dimension.")
    if (not coord_obj.mode.equidistant):
        raise ValueError("STFT calculation is possible only along equidistant coordinates.")

    proc_dim = coord_obj.dimension_list[0]
    # the dimension to transform along is the only element in this ccordinate's dimension list

    fs = 1/coord_obj.step[0]  # sampling frequency is the step of time axis

    f_ax, t_ax, stft = signal.stft(d.data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                                   nfft=nfft, detrend=detrend, return_onesided=return_onesided,
                                   boundary=boundary, padded=padded, axis=proc_dim)
    # all parameters are taken from inputs, except the axis to transform along

    d_out = type(d)(data_array=stft,
                    # error=out_err,  # error calculation not yet implemented
                    coordinates=d.coordinates,
                    exp_id=d.exp_id,
                    data_unit=flap.coordinate.Unit("Spectral density"))

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

    # create new coodrdinate axes from stft return axes (everything is equidistant)
    c_f = flap.coordinate.Coordinate(name='Frequency',
                                     unit='Hz',
                                     mode=flap.coordinate.CoordinateMode(equidistant=True),
                                     shape=[],
                                     start=f_ax[0],
                                     step=f_ax[1] - f_ax[0],
                                     dimension_list=[proc_dim])

    c_t = flap.coordinate.Coordinate(name='Time',
                                     unit='s',
                                     mode=flap.coordinate.CoordinateMode(equidistant=True),
                                     shape=[],
                                     start=t_ax[0],
                                     step=t_ax[1] - t_ax[0],
                                     dimension_list=[proc_dim])

    d_out.add_coordinate_object(c_t, index=0)
    d_out.add_coordinate_object(c_f, index=0)
    return d_out
