"""
Created on Wed Jan 23 09:44:50 2019

This is the coordinate description and related methods for FLAP

@author: Sandor Zoletnik  (zoletnik.sandor@ek-cer.hu)
Centre for Energy Research
"""

import math
import numpy as np
from .tools import *
import flap.config
import copy
from decimal import Decimal

class Intervals:
    """Class to describe a series of intervals.

    Different types of intervals exist:

    - Regular intervals are identical length ones repeating with a fixed step.
    - Irregulars are just a list of start-stop values.
    - For integer type values both the start and stop value is included.
    - The optional number gives the number of ranges for the regular intervals.

    Parameters
    ----------
    start, stop : int | float | ndarray
        For regular intervals: start and stop value of first interval.
        For irregular intervals: start and stop value of all intervals.
        (Identical-length numpy arrays.)
    step : int | float, optional, default=None
        If not given, the intervals are assumed to be irregular, otherwise
        regular.
    number : int, optional, default=None
        The number of intervals in regular case. If not set, the intervals
        continue indefinitely.
    """
    def __init__(self,start,stop,step=None,number=None):
        if (step is None):
            regular = False
        else:
            regular = True

        if (regular):
            if ((type(start) is str) or (type(stop) is str) or (type(step) is str)):
                raise ValueError("Non-numeric type in selection interval.")
            if (start >= stop):
                raise ValueError("Bad interval description: stop before start.")
            self.start = start
            self.stop = stop
            self.step = step
            if (number is None):
                self.number = None
            else:
                self.number = round(number)
        else:
            if (np.isscalar(start)):
                self.start = np.array([start])
            else:
                if (type(start) is list):
                    _start = np.array(start)
                else:
                    _start = start
                if ((type(_start) is not np.ndarray) or (_start.ndim != 1)):
                    raise TypeError("Start values should be 1D numpy array or scalar.")
                self.start = _start
            if (np.isscalar(stop)):
                self.stop = np.array([stop])
            else:
                if (type(stop) is list):
                    _stop = np.array(stop)
                else:
                    _stop = stop
                if ((type(_stop) is not np.ndarray) or (_stop.ndim != 1)):
                    raise TypeError("Stop values should be 1D numpy array or scalar.")
                self.stop = _stop
            if (len(self.start) != len(self.stop)):
                raise ValueError("Length of start and stop arrays should be identical.")
            if (np.amin(self.stop - self.start) <= 0):
                raise ValueError("Interval length should be positive.")
            self.number = len(self.start)
            self.step = None
            if (not np.issubdtype(self.start.dtype, np.number) or \
                not np.issubdtype(self.stop.dtype, np.number)):
                raise ValueError("Non-numeric type in selection interval.")

    def regular(self):
        """Returns True if the interval is regular, otherwise returns False.

        Returns
        -------
        bool
        """
        if (self.step is None):
            return False
        else:
            return True

    def interval_limits(self,limits=None,partial_intervals=False):
        """Return the range lower and upper limits as two numpy arrays. Limit
        the ranges within argument limits.
        
        Parameters
        ----------
        limits : array_like, optional, default=None
            Two-element list with lower and upper limit, in this order.
        partial_intervals : bool, optional, default=False
            Return also partial ranges which extend over limits.  Their size
            will be truncated.

        Returns
        -------
        range_starts, range_stops : ndarray
            The range lower and upper limits.
        """
        if ((limits is None) and self.regular() and (self.number is None)):
            raise ValueError("For the range limits either the number of intervals or limits must be set.")

        if (limits is None):
            if (self.regular()):
                range_starts = np.arange(self.number) * self.step + self.start
                range_stops = np.arange(self.number) * self.step + self.stop
                return range_starts, range_stops
            else:
                return self.start, self.stop
        number, start_interval = self.interval_number(limits=limits,partial_intervals=partial_intervals)
        if (number == 0):
            raise ValueError("No interval within limits.")

        if (self.regular()):
            range_starts = (np.arange(number) + start_interval) * self.step + self.start
            range_stops = (np.arange(number) + start_interval) * self.step + self.stop
        else:
            range_starts = self.start[start_interval: start_interval + number]
            range_stops = self.stop[start_interval: start_interval + number]

        if (partial_intervals):
            ind = np.nonzero(np.logical_and(range_stops > limits[0],
                                            range_starts < limits[0]))[0]
            if (ind.size != 0):
                range_starts[ind] = limits[0]
            ind = np.nonzero(np.logical_and(range_stops > limits[1],
                                            range_starts < limits[1]))[0]
            if (ind.size != 0):
                range_stops[ind] = limits[1]

        if ((np.amax(range_stops) < limits[0]) or (np.amin(range_starts) > limits[1])):
            raise RuntimeError("Internal error: selected intervals outside of limits.")
        return range_starts, range_stops

    def interval_number(self,limits=None,partial_intervals=False):
        """Return the number of intervals and the index of the start interval.
        Limit the ranges within argument limits.

        Parameters
        ----------
        limits : array_like, optional, default=None
            2 elements list with lower and upper limit.
        partial_intervals : bool, optional, default=False
            Take into account also partial ranges which extend over limits.

        Returns
        -------
        int
            The number of intervals.
        int
            The index of the start interval.
        """
        if ((limits is None) and self.regular() and (self.number is None)):
            raise ValueError("For the range limits either the number of intervals or limits must be set.")

        if (limits is None):
            if (self.regular()):
                return self.number, 0
            else:
                return self.start.size, 0

        if (self.regular()):
            if (((self.step > 0) and (self.start > limits[1]))\
                or ((self.step < 0) and (self.start < limits[0]))):
                return 0, 0
            if (self.step > 0):
                if (partial_intervals):
                    if (self.stop < limits[0]):
                        start_interval = int((limits[0] - self.stop) / self.step) + 1
                    else:
                        start_interval = 0
                    interval_n = int((limits[1] - self.start) / self.step) - start_interval + 1
                else:
                    if (self.start >= limits[0]):
                        start_interval = 0
                    else:
                        start_interval = int((limits[0] - self.start) / self.step) + 1
                    interval_n = int((limits[1] - self.stop) / self.step) - start_interval + 1
            else:
                if (partial_intervals):
                    if (self.start >= limits[1]):
                        start_interval = int((self.start - limits[1]) / self.step) + 1
                    else:
                        start_interval = 0
                    interval_n = int((self.stop - limits[0]) / self.step) - start_interval + 1
                else:
                    if (self.stop > limits[1]):
                        start_interval = int((self.stop - limits[1]) / self.step) + 1
                    else:
                        start_interval = 0
                    interval_n = int((self.stop - limits[0]) / self.step) - start_interval + 1

            if (self.number is not None):
                if (start_interval + interval_n >= self.number):
                    interval_n = self.number - start_interval
                    if (interval_n < 0):
                        interval_n = 0
            return interval_n, start_interval


        else:
            if (partial_intervals):
                ind = np.nonzero(np.logical_and(self.stop >= limits[0],
                                                self.start <= limits[1]))[0]
            else:
                ind = np.nonzero(np.logical_and(self.start >= limits[0],
                                                self.stop <= limits[1]))[0]
            if (ind.size == 0):
                return 0, 0
            else:
                return ind.size, ind[0]

class Unit:
    """Class representing the unit of a quantity.

    Parameters
    ----------
    name : str, optional, default=''
        Name of the quantity. (E.g. 'Time'.)
    unit : str, optional, default=''
        Name of the unit. (E.g. 'Second'.)
    """
    def __init__(self,name="",unit=""):
        self.name = name
        self.unit = unit

    def title(self,language='EN',complex_txt=None,new_unit=None):
        """Returns a title string to be used in plotting.

        Parameters
        ----------
        language : {'EN', 'HU'}, optional, default='EN'
            Language to be used for creating the title string.
        complex_txt : {[0, 0], [0, 1], [1, 0], [1, 1]}, optional, default=None
            List of 2 numbers representing which component of a complex variable
            is used:

            - ``[0,0]``: Amplitude
            - ``[0,1]``: Phase
            - ``[1,0]``: Real
            - ``[1,1]``: Imaginary

        new_unit : str, optional, default=None
            Unit string override to be used instead of `self.unit`.

        Returns
        -------
        str
            The title string.
        """        
        if (language == 'EN'):
            if ((self.unit is None) or (self.unit == '')):
                unit_txt = ' [a.u.]'
            else:
                if new_unit is None:
                    unit_txt = ' [' + self.unit + ']'
                else:
                    unit_txt = ' [' + new_unit + ']'
            if (complex_txt is not None):
                if (complex_txt == [0,0]):
                    if (self.name is None or (self.name == '')):
                        txt = 'Amplitude ' + unit_txt
                    else:    
                        txt = 'Amplitude of ' + self.name + unit_txt
                elif (complex_txt == [0,1]):
                    if (self.name is None or (self.name == '')):
                        txt = 'Phase [rad]'
                    else:    
                        txt = 'Phase of ' + self.name + ' [rad]'
                elif (complex_txt == [1,0]):
                    if (self.name is None or (self.name == '')):
                        txt = 'Real part ' + unit_txt
                    else:
                        txt = 'Real part of ' + self.name + unit_txt
                elif (complex_txt == [1,1]):
                    if (self.name is None or (self.name == '')):
                        txt = 'Imaginary part ' + unit_txt
                    else:
                        txt = 'Imaginary part of ' + self.name + unit_txt
            else:
                txt = self.name + unit_txt
        elif (language == 'HU'):
            if ((self.unit is None) or (self.unit == '')):
                unit_txt = ' [tetsz. e.]'
            else:
                unit_txt = ' [' + self.unit + ']'
            if (complex_txt is not None):
                if (complex_txt == [0,0]):
                    txt = self.name +' amplitudo' + unit_txt
                elif (complex_txt == [0,1]):
                    txt = self.name + ' fazis [rad]'
                elif (complex_txt == [1,0]):
                    txt = self.name + ' valos resz' + unit_txt
                elif (complex_txt == [1,1]):
                    txt = self.name + ' kepzetes resz'+ unit_txt
            else:
                txt = self.name + unit_txt
        else:
            raise ValueError("Unknown language.")
        return txt

class CoordinateMode:
    """Class for storing mode flags of the coordinate description.

        Parameters
        ----------
        equidistant : bool, optional, default=True
            Whether the coordinate values are equidistant.
        range_symmetric : bool, optional, default=True
            Whether the ranges corresponding to the coordinates are symmetric.
    """
    def __init__(self,equidistant=True, range_symmetric=True):
        self.equidistant = equidistant
        self.range_symmetric = range_symmetric


class Coordinate:
    """Class for the description of a mapping of coordinate values from an
    n-dimensional coordinate sample space to coordinates of an m-dimensional
    data matrix.

    The coordinate sample space is a rectangular equidistant point matrix, with
    equal steps in each dimension.  For dimension ``i``, the sample index spans
    ``0 ... n-1`` if ``shape[i] == n``.
    
    The sample space in this coordinate description does not necessarily match
    the shape of any sub-matrix of the data object. If the shape is different
    then interpolation is done assuming the coordinate of the first(last) point
    of the coordinate matrix is the coordinate of the first(last) data point.
    
    A coordinate can be anything described by name and unit.  Standard
    coordinates: 'Time', 'Channel', 'Channel number', 'Device_x', 'Device_y',
    'Device_z', 'Device_R', 'Device_Z', 'Device_phi', 'Flux_r', 'Flux_theta',
    'Flux_phi', 'Frequency', 'Time lag'.
    
    The ranges and start-step pair of parameters are alternatives.

    Parameters
    ----------
    name : str, optional, default=None
        Name of the quantity measured in the coordinate. (E.g. 'Time'.)
    unit : str, optional, default=None
        Name of the unit used along the coordinate. (E.g. 'Second'.)
    mode : flap.CoordinateMode, optional, default=``CoordinateMode()``
        Coordinate mode to use.
    shape : array_like, optional, default=``[]``
        The shape of the sample space matrix in this coordinate description. If
        ``len(shape) == 0``, then the coordinate is the same for all samples and
        it is described by `values` and `value_ranges`. The parameter `shape`
        should not have a ``1`` element. In the equidistant case only the
        dimension of `shape` is used.
    start : float, optional, default=None
        Start of the coordinate mapping. Only valid if the coordinate mode is
        equidistant.
    step : list of float, optional, default=None
        Steps of the coordinate mapping. One-dimensional list with the same
        number of elements as `dimension_list`. Only valid if the coordinate
        mode is equidistant.  
    c_range : array_like, optional, default=None
        Used for specifying a range when reading data.
    values, value_index : array_like, optional, default=None
        Only valid if the coordinate mode is non-equidistant.  The `values`
        describe the coordinate values at `value_index`. Sample values are a 2D
        array of size ``(len(shape), N_sample)`` containing indices to the
        coordinate matrix described by `shape`. That is, ``values_index[:,i]``
        gives the coordinates of ``values[i]`` in the coordinate sample space.
    value_ranges : float | array_like | dict, optional, default=None
        Specifies the range around the coordinate values.

        If the coordinate mode has symmetric ranges, it is symmetric around the
        values: ``[values-value_ranges, value+value_ranges]``. If not
        symmetric, it is ``[values-value_ranges_low, value+value_ranges_high]``.

        For equidistant coordinates:

        - if the range is not symmetric, `value_ranges` should be a 2 element
          array;
        - if the range is symmetric, `value_ranges` should be a scalar.

        For non-equidistant coordinates:

        - if the range is not symmetric, `value_ranges` should be a dictionary
          with keys 'low', 'high', with each value having the same dimensions as
          `values`;
        - if the range is symmetric, `value_ranges` should have the same shape
          as `values`.

    dimension_list : list, optional, default=[]
        List of data dimensions with which this coordinate is associated (0...).
        Length of `dimension_list` should be equal to length of `shape`.
    """
    def __init__(self,
                 name=None,
                 unit=None,
                 mode=CoordinateMode(),
                 shape=[],
                 start=None,
                 step=None,
                 c_range=None,
                 values=None,
                 value_index=None,
                 value_ranges=None,
                 dimension_list=[]):
        self.unit = Unit(name=name,unit=unit)
        self.mode = copy.deepcopy(mode)
        self.shape = shape
        # Would have been good to use decimal.Decimal types but operations
        # between float and Decimal are not allowed, therefore using float
        self.start = start
        self.step = step
        if (type(self.start) is Decimal):
            self.start = float(start)
        if (type(self.step) is Decimal):
            self.step = float(step)

        self.c_range = c_range

        self.value_index = value_index
        self.values = values
        if (type(self.values) is list):
            self.values = np.array(self.values)
        self.value_ranges = value_ranges
        self.dimension_list = dimension_list

    @property
    def start(self):
        """Start of the coordinate mapping. Only valid if the coordinate mode is
        equidistant.

        Returns
        -------
        float or None
        """
        return self.__start

    @start.setter
    def start(self, start):
        if (type(start) is type(None)):
            self.__start = None
            return
        if (type(start) is Decimal):
            _start = float(start)
        else:
            _start = start
        # Checking this strange way to also enable numpy types
        kind = np.array([_start]).dtype.kind
        if ((kind != 'u') and (kind != 'i') and (kind != 'f') and (kind != 'c')):
            raise TypeError("Invalid coordinate start value.")
        self.__start = _start       

    @property
    def step(self):
        """Steps of the coordinate mapping. Only valid if the coordinate mode is
        equidistant.  

        Returns
        -------
        list or None
        """
        return self.__step

    @step.setter
    def step(self,step):
        if (step is None):
            self.__step = None
            return
        if (type(step) is np.ndarray):
            _step = list(step)
        else:
            _step = step
        if (type(_step) is not list):
            _step = [_step]
        for i in range(len(_step)):
            if (type(_step[i]) is Decimal):
                _step[i] = float(_step[i])
            kind = np.array([_step[i]]).dtype.kind
            if ((kind != 'u') and (kind != 'i') and (kind != 'f') and (kind != 'c')):
                raise TypeError("Invalid coordinate step value.")
        self.__step = _step

    @property
    def dimension_list(self):
        """List of data dimensions with which this coordinate is associated with.

        Returns
        -------
        list of int
        """
        return self.__dimension_list

    @dimension_list.setter
    def dimension_list(self,dimension_list):
        if (type(dimension_list) is not list):
            self.__dimension_list = [dimension_list]
        else:
            self.__dimension_list = dimension_list

    @property
    def shape(self):
        """The shape of the sample space matrix in this coordinate
        description.

        Returns
        -------
        list
        """
        return self.__shape

    @shape.setter
    def shape(self,shape):
        if (type(shape) is tuple):
            self.__shape = list(shape)
        elif (type(shape) is not list):
            self.__shape = [shape]
        else:
            self.__shape = shape

    def __ne__(self, c1):
        """Compares two coordinates, assuming the same data shape.

        Parameters
        ----------
        c1: flap.Coordinate
            The coordinate to compare to.

        Returns
        -------
        bool
        """
        return not (self == c1)

    def __eq__(self, c1):
        """Compares two coordinates, assuming the same data shape.

        Parameters
        ----------
        c1: flap.Coordinate
            The coordinate to compare to.

        Returns
        -------
        bool
        """
        if (self.unit.name != c1.unit.name):
            return False
        if (self.unit.unit != c1.unit.unit):
            return False
        if (self.dimension_list != c1.dimension_list):
            return False
        if (self.mode.equidistant != c1.mode.equidistant):
            return False
        if (self.mode.equidistant and c1.mode.equidistant):
            if (type(self.step) is not list):
                self_step = [self.step]
            else:
                self_step = self.step
            if (type(c1.step) is not list):
                c1_step = [c1.step]
            else:
                c1_step = c1.step
            minstep = math.fabs(self_step[0])    
            for i in range(len(self_step)):
                if (math.fabs(self_step[i] - c1_step[i]) / math.fabs(self_step[i]) > 0.001):
                    return False
                if (math.fabs(self_step[i]) < minstep):
                    minstep = math.fabs(self_step[i])
            if (math.fabs(self.start - c1.start) / minstep > 0.001):
                return False
            if (self.value_ranges != c1.value_ranges):
                return False
        else:
            if (self.shape != c1.shape):
                return False
            if (self.shape == []):
                if (self.values != c1.values):
                    return False
            else:
                if (self.values.shape != c1.values.shape):
                    return False
                if (np.nonzero(self.values != c1.values)[0].size != 0):
                    return False
                if ((self.value_ranges is not None) and (c1.value_ranges is None) or
                    (self.value_ranges is None) and (c1.value_ranges is not None)):
                    return False
                if (self.value_ranges is not None):
                    if (self.mode.range_symmetric != c1.mode.range_symmetric):
                        return False
                    if (self.mode.range_symmetric):
                        if (np.nonzero(self.value_ranges.flatten() != c1.value_ranges.flatten())[0].size != 0):
                            return False
                    else:
                        if (np.nonzero(self.value_ranges[0].flatten() != c1.value_ranges[0].flatten())[0].size != 0):
                            return False
                        if (np.nonzero(self.value_ranges[1].flatten() != c1.value_ranges[1].flatten())[0].size != 0):
                            return False
        return True
                    
    def non_interpol(self, data_shape):
        """Return True if the shape of the coordinate description is the same as
        the sub-data-array for which it applies and `self.value_index` is None.
        In this case there is no need for interpolation, coordinate values are copied.
        Valid only for non-equidistant coordinate mode.

        Parameters
        ----------
        data_shape : array_like
            The shape of the sub-data-array.
        
        Returns
        -------
        bool
        """
        if (self.value_index is not None):
            return False
        if (len(self.dimension_list) == 0):
            return True
        if (data_shape is None):
            return True
        # The shape for which the coordinate description applies
        ds_coord = [data_shape[x] for x in self.dimension_list]
        # Removing dimensions with one element
        ds_coord = [x for x in ds_coord if x != 1]
        if ((type(self.shape) is not tuple) and (type(self.shape) is not list)):
            shape = [self.shape]
        else:
            shape = list(self.shape)
        # Removing dimensions with one element
        shape_clean = [x for x in shape if x != 1]
        return ds_coord == shape_clean

    def dtype(self):
        """Return the data type of the coordinate.
        
        Returns standard Python types: str, int, float, complex, boolean or object.

        Returns
        -------
        type
        """
        if (self.mode.equidistant):
            if ((self.start is None) or (self.step is None)):
                raise ValueError("Bad equidistant coordinate description: step or start is missing.")
            # Reporting int type only if both start and step is integer
            kind = np.array([self.start]).dtype.kind
            for s in self.step:
                kind_step = np.array([s]).dtype.kind
                if (kind_step == 'f'):
                    kind = 'f'
                if (kind_step == 'c'):
                    kind = 'c'
            if ((kind == 'i') or (kind == 'u')):
                return type(1)
            elif (kind == 'f'):
                return type(1.0)
            if (kind == 'c'):
                return type(complex(1))
        else:
            try:
                kind = self.values.dtype.kind
                if ((kind == 'u') or (kind == 'i')):
                    return type(1)
                if (kind == 'f'):
                    return type(1.1)
                if (kind == 'c'):
                    return type(complex(1))
                if (kind == 'S'):
                    return type('a')
                if (kind == 'b'):
                    return type(True)
                if (kind == 'O'):
                    return object
                raise TypeError("Unknown coordinate value type.")
            except (IndexError, TypeError, AttributeError):
                return type(self.values)

    def isnumeric(self):
        """Returns whether the data type is numeric.

        Returns
        -------
        bool
        """
        dt = self.dtype()
        if ((dt == int) or (dt == float) or (dt == complex) or (dt == Decimal)):
            return True
        else:
            return False

    def string_subtype(self):
        """Returns the type of the values if the coordinate type is str.
        Otherwise returns None.

        Returns
        -------
        type or None
        """
        if (self.dtype == str):
            if (np.isscalar(self.values)):
                return type(self.values)
            else:
                return self.values.dtype
        else:
            return None

    def change_dimensions(self):
        """Return the list of dimensions of the data array along which this
        coordinate changes.

        Returns
        -------
        list
        """
        return self.dimension_list

    def nochange_dimensions(self, data_shape):
        """Return the list of dimensions of the data array along which this
        coordinate does not change.

        Parameters
        ----------
        data_shape : array_like
            The data shape to use.

        Returns
        -------
        list
        """
        if (self.change_dimensions == []):
            return list(range(len(data_shape)))

        nochange_list = []
        for i in range(len(data_shape)):
            try:
                self.dimension_list.index(i)
            except ValueError:
                nochange_list.append(i)
        return nochange_list

    def data(self, data_shape=None, index=None, options=None):
        """Returns the coordinates, low and high limits for a sub-array of the
        data array in a DataObject.

        Parameters
        ----------
        index : list | tuple of int, optional, default=None
            Describes the elements in `DataObject.data` for which the
            coordinates are required. The length of the array should be
            identical to the number of dimensions of the data array. Elements
            can be a mixture of numbers, slice objects, lists, numpy arrays,
            integer iterators and ellipses.

            Examples for 3D array:

            - ``(...,0,0)``: coordinates of the elements in the first row of the
              data array

            - ``(slice(2,5),2,...)``

        data_shape : array_like, optional, default=None
            The shape of the data array (without slicing) for which coordinates
            are requested.
        options : dict, optional, default=None
            Dictionary with options for processing:

            - 'Interpolation' (default='Linear'):

              - 'Linear': For non-equidistant axis, when
                values shape is different from data shape)

            - 'Change only' (default=False): 

              - True (return only the data for those dimensions where this
                coordinate changes. E.g. if it changes only along one dimension
                the output array will have 1 element in all other dimensions.)

        Returns
        -------
        values, value_range_low, value range_high : array_like
            Values and the respective low and high range values. The low and
            high values are the absolute values not, the difference from values.
        """
        default_options = {'Interpolation': 'Linear',
                           'Change only': False}

        _options = flap.config.merge_options(default_options, options)

        if (data_shape is None):
            raise ValueError("Missing data_shape argument in flap.Coordinate.data()")
        if (type(data_shape) is not np.ndarray):
            _data_shape = np.array(data_shape)
        else:
            _data_shape = data_shape

        if ((index is not None) and _options['Change only']):
            raise ValueError("index and 'Change only' option cannot be used at the same time in Coordinate.data().")
        if (_options['Change only']):
            if (self.change_dimensions() is []):
                index = ...
            else:    
                index = [0] * len(data_shape)
                for i in self.change_dimensions():
                    index[i] = ...            
        if (index is None) or (index is Ellipsis):
            _index = [...] * len(data_shape)
        else:
            _index = index
        if ((type(_index) is not list) and (type(_index) is not tuple)):
            _index = [_index]

        if (len(_index) != len(_data_shape)):
            raise \
                ValueError("Incompatible data_shape and index arguments in flap.Coordinate.data()")

        # Determining the shape of the output array and the indices for selecting the output elements
        # from the data array (This will be needed for selecting the coordinates.)
        # Out_index will be a list of np.arrays
        out_shape = [None] * len(_index)
        out_index = [None] * len(_index)
        for i in range(len(_index)):
            if (_index[i] is Ellipsis):
                out_shape[i] = _data_shape[i]
                out_index[i] = np.arange(_data_shape[i],dtype=int)
            elif (type(_index[i]) is slice):
                if (_index[i].step is None):
                    out_shape[i] = round(_index[i].stop - _index[i].start)
                else:
                    out_shape[i] = int(round((_index[i].stop - _index[i].start) / _index[i].step))
                out_index[i] = np.array(list(range(*_index[i].indices(_data_shape[i]))))
            elif (type(_index[i]) is range):
                out_shape[i] = int(round((_index[i].stop - _index[i].start) / _index[i].step))
                out_index[i] = np.array(list(_index[i]))
            elif (type(_index[i]) is list):
                out_shape[i] = len(_index[i])
                out_index[i] = np.array(_index[i])
            elif (type(_index[i]) is np.ndarray):
                out_shape[i] = len(_index[i])
                out_index[i] = _index[i]
            elif (np.isscalar(_index[i])):
                out_shape[i] = 1
                out_index[i] = np.array([_index[i]])

        if (len(self.dimension_list) == 0):
            # If coordinate is constant, returning it
            if (self.mode.equidistant):
                out_coord = np.full(tuple(out_shape),self.start)
                if (self.value_ranges is not None):
                    if (self.mode.range_symmetric):
                        vl = self.start - self.value_ranges
                        vh = self.start + self.value_ranges
                    else:
                        vl = self.start - self.value_ranges[0]
                        vh = self.start + self.value_ranges[1]
                else:
                    vl = self.start
                    vh = self.start

                out_coord_low = np.full(tuple(out_shape),vl)
                out_coord_high = np.full(tuple(out_shape),vh)
                return out_coord, out_coord_low, out_coord_high
            try:
                v = self.values[0]
            except (IndexError, TypeError):
                v = self.values
            out_coord = np.full(tuple(out_shape),v)
            if (self.value_ranges is not None):
                try:
                    vl = self.value_ranges[0,0]
                    vh = self.value_ranges[0,1]
                except (TypeError,IndexError):
                    vl = self.value_ranges[0]
                    vh = self.value_ranges[1]
                out_coord_low = np.zeros(tuple(out_shape),dtype=self.dtype()) + vl
                out_coord_high = np.zeros(tuple(out_shape),dtype=self.dtype()) + vh
            else:
                out_coord_low = None
                out_coord_high = None
            return out_coord, out_coord_low, out_coord_high

        # Selecting those dimensions in _index where the coordinate changes
        # This is an index into the value array of the coordinate
        value_index = [out_index[di] for di in self.dimension_list]
#        value_index = []
#        for di in self.dimension_list:
#            value_index.append(out_index[di])
        # This will be the shape of the coordinate value matrix in the dimension_list
        # subspace of the output array with out_shape
        value_mx_shape = tuple([out_shape[i] for i in self.dimension_list])

        if (self.mode.equidistant):
           #Equidistant
           # This will store the equidistant coordinate values for the selected part
            value_mx = np.full(value_mx_shape,self.start)
            for i in range(len(self.dimension_list)):
                # Creating an array (value_mx_temp) which has the same shape as value_mx but contains
                # only the dependency of the coordinate on dimension i.
                # First determining the shape of an array which has the same number of dimensions
                # as value_mx and has the same number of elements in dimension i as value_mx, but only
                # in all other dimensions.
                shape = [1] * len(self.dimension_list)
                shape[i] = value_mx.shape[i]
                # Creating an index array which indexes into all elements in dimension i but to element
                # 0 in all other dimensions
                ind = [0] * len(self.dimension_list)
                ind[i] = np.arange(value_mx.shape[i])
                # Creating value_ms_tmp as described above
                value_mx_tmp = np.zeros(tuple(shape),dtype=value_mx.dtype)
                # Creating the coordinate dependency and putting it into value_mx_tmp
                x = value_index[i]
#                if (x.dtype is not np.dtype(np.float64)):
#                    x = x.astype(float)
                try:
                    st = self.step[i]
                except (TypeError, IndexError):
                    st = self.step
                value_mx_tmp[tuple(ind)] = x * st
                # extending to the shape of value_mx
                for j in range(len(shape)):
                    if (j != i):
                        value_mx_tmp = np.repeat(value_mx_tmp,value_mx_shape[j],j)
                # Adding to value_mx
                value_mx += value_mx_tmp

            if (self.value_ranges is not None):
                if (self.mode.range_symmetric):
                    value_mx_low = value_mx - self.value_ranges
                    value_mx_high = value_mx + self.value_ranges
                else:
                    value_mx_low = value_mx - self.value_ranges[0]
                    value_mx_high = value_mx + self.value_ranges[1]
            else:
                value_mx_low = None
                value_mx_high = None
        else:
            # Non-equidistant
            if (self.non_interpol(_data_shape)):
                # Number of non-equidistant elements equals the elements in data array
                #   --> no interpolation is necessary
                if (len(value_index) == 0):
                    pass
                value_index_arrays = flap.tools.submatrix_index(np.array(self.values).shape, value_index)
                value_mx = np.array(self.values)[value_index_arrays]
                if (self.value_ranges is not None):
                    _value_ranges = np.array(self.value_ranges)
                    if (self.mode.range_symmetric):
                        value_mx_low = value_mx - _value_ranges[value_index_arrays]
                        value_mx_high = value_mx + _value_ranges[value_index_arrays]
                    else:
                        value_mx_low = value_mx - _value_ranges[0][value_index_arrays]
                        value_mx_high = value_mx + _value_ranges[1][value_index_arrays]
                else:
                    value_mx_low = None
                    value_mx_high = None

            else:
                # Interpolation is necessary
                raise NotImplementedError(
                       "Interpolating for coordinate determination is not implemented yet. Coordinate:{:s}".format(self.unit.name))

        # Extending value_mx and like to the output shape
        # First creating a dimension list where this coordinate does not change
        nochange_list = self.nochange_dimensions(_data_shape)
        # If this list is empty, nothing to be done
        if (len(nochange_list) == 0):
            out_coord = value_mx
            out_coord_low = value_mx_low
            out_coord_high = value_mx_high
        else:
            # if we need to extend along the dimensions not in dimension_list
            out_coord = flap.tools.expand_matrix(value_mx,out_shape,self.dimension_list)
            if (value_mx_low is not None):
                out_coord_low = flap.tools.expand_matrix(value_mx_low,out_shape,self.dimension_list)
            else:
                out_coord_low = None
            if (value_mx_high is not None):
                out_coord_high = flap.tools.expand_matrix(value_mx_high,out_shape,self.dimension_list)
            else:
                out_coord_high = None

        return out_coord, out_coord_low, out_coord_high

    def data_range(self, data_shape=None):
        """Returns the data range and the data range with errors for the
        coordinate.

        Parameters
        ----------
        data_shape : array_like, optional, default=None
            Data shape.

        Returns
        -------
        value_range, value_range_error : list
        """
        if (data_shape is None):
            raise ValueError("Missing data_shape argument in flap.Coordinate.data()")

        if (self.mode.equidistant):
            if (len(self.dimension_list) == 0):
                if (self.value_ranges is None):
                    return [self.start]*2, [self.start]*2
                else:
                    if (self.mode.range_symmetric):
                        return [self.start]*2, [self.start-self.value_ranges, self.start+self.value_ranges]
                    else:
                        return [self.start]*2, [self.start-self.value_ranges[0], self.start+self.value_ranges[1]]

            # Calculating the coordinates of the corner points in the multi-dimensional space
            corner_coords = [self.start,
                             self.start + self.step[0]*(data_shape[self.dimension_list[0]]-1)]
            for i in range(1,len(self.dimension_list)):
                n = len(corner_coords)
                for i_n in range(n):
                    corner_coords.append(corner_coords[i_n] + self.step[i]*(data_shape[self.dimension_list[i]]-1))
            value_range = [min(corner_coords), max(corner_coords)]
            if (self.value_ranges is not None):
                if (type(self.value_ranges) is list):
                    value_range_error = [value_range[0] - self.value_ranges[0],
                                         value_range[1] + self.value_ranges[1]]
                else:
                    value_range_error = [value_range[0] - self.value_ranges,
                                         value_range[1] + self.value_ranges]
            else:
                value_range_error = value_range
            return value_range, value_range_error

        else:
            if (not self.non_interpol(data_shape)):
                raise NotImplementedError("Interpolating coordinate description is not implemented.")
            value_range = [np.amin(self.values), np.amax(self.values)]
            if (self.value_ranges is not None):
                if (type(self.value_ranges) is list):
                    value_range_error = [np.amin(self.values-self.value_ranges[0]), np.amax(self.values+self.value_ranges[1])]
                else:
                    value_range_error = [np.amin(self.values-self.value_ranges), np.amax(self.values+self.value_ranges)]
            else:
                value_range_error = value_range
            return value_range, value_range_error
