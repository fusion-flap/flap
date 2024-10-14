"""

Time-frequency analysis in FLAP.

@author: Marton Vavrik (vavrik.marton@reak.bme.hu)

"""

from scipy import signal
import numpy as np
import flap.config
import flap.coordinate


def _stft(d, coordinate=None, options=None):  # numpydoc ignore=SS05,PR01,RT01
    """Calculate the short-time Fourier transform of data `d`. See
    `flap.DataObject.stft()` for more details.
    """
    default_options = {'window': 'hann',
                       'nperseg': 256,
                       'noverlap': None,
                       'nfft': None,
                       'detrend': False,
                       'return_onesided': True,
                       'boundary': 'zeros',
                       'padded': True
                       }

    _options = flap.config.merge_options(default_options, options, data_source=d.data_source, section='STFT')

    window=_options['window']
    nperseg=_options['nperseg']
    noverlap=_options['noverlap']
    nfft=_options['nfft']
    detrend=_options['detrend']
    return_onesided=_options['return_onesided']
    boundary=_options['boundary']
    padded=_options['padded']

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

    fs = 1 / coord_obj.step[0]  # sampling frequency is the step of time axis

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

    # create new coordinate axes from stft return axes (everything is equidistant)
    # Frequency coordinate is parallel with the newly created dimension
    c_f = flap.coordinate.Coordinate(name='Frequency',
                                     unit='Hz',
                                     mode=flap.coordinate.CoordinateMode(equidistant=True),
                                     shape=[],
                                     start=f_ax[0],
                                     step=(f_ax[-1] - f_ax[0]) / len(f_ax),
                                     dimension_list=[proc_dim])

    c_t = flap.coordinate.Coordinate(name='Time',
                                     unit='s',
                                     mode=flap.coordinate.CoordinateMode(equidistant=True),
                                     shape=[],
                                     start=t_ax[0] + coord_obj.start,
                                     step=(t_ax[-1] - t_ax[0]) / len(t_ax),  # *np.prod(stft.shape[:-2]),
                                     # has to increase step if non-1d data (due to scipy's interpretation of fs)
                                     dimension_list=[len(stft.shape) - 1])

    d_out.add_coordinate_object(c_t, index=0)
    d_out.add_coordinate_object(c_f, index=0)
    return d_out
