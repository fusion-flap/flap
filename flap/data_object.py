# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:37:32 2019

@author: Zoletnik
"""

import numpy as np
from scipy import signal
import fnmatch
from decimal import *
import copy
import matplotlib.pyplot as plt
import math
import io
import pickle

from .tools import *
from .coordinate import *
import flap.config
from .spectral_analysis import _apsd, _cpsd, _trend_removal, _ccf
from .plot import _plot

class DataObject:
    """ This is the data object
    """

    def __init__(self,
                 data_array=None,
                 error=None,
                 data_unit=None,
                 coordinates=None,
                 exp_id=None,
                 data_title=None,
                 info=None,
                 data_shape=None,
                 data_source=None):
        """
        data_array: A numpy array with the data
        error: One or two numpy arrays with the error.
               An array: Symmetric error
               List of two arrays: asymmetric error ([low deviation,high deviation])
        data_unit: The unit of the data. Should be a UnitClass variable or None
        exp_id: Experiment ID. Can be None if not applicable.
        coordinates: flap.Coordinate class or list of flap.Coordinate class variables
                     compatible with the data array.
        data_title: Description of the data.
        data_shape: Used only if data_array is not present
        """

        self.coordinates = coordinates
        self.data_source = data_source
        self.exp_id = exp_id
        self.data_title = data_title
        self.info = info
        self.history = None
        self.shape = data_shape

        if (data_unit is not None):
            if (type(data_unit) is not Unit):
                raise ValueError("data_unit for DataObject must be a flap.Unit.")
            else:
                self.data_unit = copy.deepcopy(data_unit)
        else:
            self.data_unit = Unit()

        if (data_array is None):
            self.data = None
            self.error = None
            return

        self.exp_id = exp_id
        if (type(data_array) is not np.ndarray):
            raise ValueError("Data for DataObject must be a numpy array.")
        if (error is not None):
            if (type(error) is not list):
                if (type(error) is not np.ndarray):
                    raise ValueError("Error must be a numpy array or a list of two numpy arrays.")
                if (data_array.shape != error.shape):
                    raise ValueError("Shape of error array is different from that of data_array. ")
            else:
                if (len(error) != 2) or \
                   (type(error[0]) is not np.ndarray) or (type(error[1]) is not np.ndarray):
                    raise ValueError("Error must be a numpy array or a list of two numpy arrays.")
        self.data = copy.deepcopy(data_array)
        self.error = copy.deepcopy(error)
        self.shape = data_array.shape

    @property
    def coordinates(self):
        return self.__coordinates

    @coordinates.setter
    def coordinates(self,coordinates):
            if (coordinates is None):
                self.__coordinates = None
                return
            if (type(coordinates) is list):
                for c in coordinates:
                    if (type(c) is not Coordinate):
                       raise TypeError("Bad type in coordinates list.")
                self.__coordinates = copy.deepcopy(coordinates)
            elif (type(coordinates) is Coordinate):
                self.__coordinates = [copy.deepcopy(coordinates)]
            else:
                raise TypeError("Bad type for coordinates.")
                
    def check(self):
        """ Does a consistency check for the data object and raises errors if problems found.
        """
        if (self.data is not None):
            if (type(self.data) is not np.ndarray):
                raise TypeError("Wrong data array. DataObject.data should be numpy.ndarray.")
            if (self.data.shape != self.shape):
                raise ValueError("DataObject.data shape is not the same as DataObject.shape.")
        else:
            if (self.shape is None):
                raise ValueError("DataObject.shape is none. Even if no data is present shape should be set otherwise coordinates cannot be determined.")                
        if (self.error is not None):
            if (type(self.error) is list):
                if (len(self.error) is not 2):
                    raise ValueError("Wrong number of elements in error. If DataObject.error is a list it should have two elements." )
                error = self.error
            else:
                error = [self.error]
            for err in error:
                if (type(err) is not np.ndarray):
                    raise TypeError("Wrong error array in DataObject. It should be numpy.ndarray.")
                if (err.shape != self.shape):
                    raise ValueError("Shape of error  array un DataObject is different from data.")
        if (self.coordinates is not None):
            for i,c in enumerate(self.coordinates):
                for j in range(i):
                    if (id(self.coordinates[i]) == id(self.coordinates[j])):
                        raise ValueError("ID of coordinates '{:s}' and '{:s}' are identical.".\
                                         format(self.coordinates[i].unit.name, self.coordinates[j].unit.name))
                    if (id(self.coordinates[i].mode) == id(self.coordinates[j].mode)):
                        raise ValueError("ID of coordinate.mode for '{:s}' and '{:s}' are identical.".\
                                         format(self.coordinates[i].unit.name, self.coordinates[j].unit.name))
                if (type(c.dimension_list) is not list):
                    raise TypeError("Wrong type for dimension list for coordinate '{:s}'.".format(c.unit.name))
                if (len(c.dimension_list) > len(self.shape)):
                    raise TypeError("Too long dimension list for coordinate '{:s}'.".format(c.unit.name)) 
                for d in c.dimension_list:
                    if (d is None):
                        raise ValueError("Null in dimension list in coordinate '{:s}'.".format(c.unit.name))    
                    if (d >= len(self.shape)):
                        raise TypeError("Wrong dimension number in coordinate '{:s}'.".format(c.unit.name))                         
                if (type(c.unit) is not flap.Unit):
                    raise TypeError("Wrong coordinate unit in coordinate #{:d}. Should be flap.Unit().".format(i))
                if (type(c.mode) is not CoordinateMode):
                    raise TypeError("Wrong coordinate mode type in '{:s}'. Should be flap.CoordinateMode().".format(c.unit.name))
                if (c.mode.equidistant):
                    if (c.start is None):
                        raise TypeError("No start for equdistant coordinate '{:s}'.".format(c.unit.name))
                    if (type(c.start ) is str):
                        raise TypeError("Invalid start value type for equdistant coordinate '{:s}'.".format(c.unit.name))
                    if (len(c.step) != len(c.dimension_list)):
                        raise ValueError("Number of step values is different from length of dimension_list in '{:s}'.".format(c.unit.name))
                    for cstep in c.step:
                        if (cstep is None):
                            raise TypeError("One step is None for equdistant coordinate '{:s}'.".format(c.unit.name))
                        if (type(cstep ) is str):
                            raise TypeError("Invalid step type for equdistant coordinate '{:s}'.".format(c.unit.name))
                        try:
                            cstep + c.start
                        except:
                            raise TypeError("Equidistant coordinate '{:s}' start and step should have same type.".format(c.unit.name))
                    if (c.value_ranges is not None):
                        if (c.mode.range_symmetric):
                            try:
                                if (c.value_ranges * 0 != 0):
                                    raise TypeError("Invalid type for value_ranges in coordinate '{:s}'.".format(c.unit.name))
                            except:
                                    raise TypeError("Invalid type for value_ranges in coordinate '{:s}'.".format(c.unit.name))  
                            if (c.mode.equidistant):
                                try:
                                    c.value_ranges + c.start
                                except:
                                    raise TypeError("Incompatible value_range and start in coordinate {:s}.".format(c.unit.name))
                            else:
                                try:
                                    c.value_ranges[0] + c.values[0]
                                except:
                                    raise TypeError("Incompatible value_range and start in coordinate {:s}.".format(c.unit.name))
                                
                        else:
                            if (type(c.value_ranges) is not list):
                                raise TypeError("Invalid type for value_ranges in asymmetric coordinate '{:s}'.".format(c.unit.name))
                            if (len(c.value_ranges) is not 2):
                                raise TypeError("Invalid list length for value_ranges in asymmetric coordinate '{:s}'.".format(c.unit.name))
                            for c_value_ranges in c.value_ranges:
                                try:
                                    if (c_value_ranges * 0 != 0):
                                        raise TypeError("Invalid type for value_ranges in coordinate '{:s}'.".format(c.unit.name))
                                except:
                                        raise TypeError("Invalid type for value_ranges in coordinate '{:s}'.".format(c.unit.name))     
                else:
                    if (c.values is None):
                        raise ValueError("No values in non-equidistant coordinate '{:s}'.".format(c.unit.name))
                    if (not c.non_interpol(self.shape)):
                        raise ValueError("Coordinate value and data shape is inconsistent in coordinate '{:s}'.".format(c.unit.name))
                                                                    
    def coordinate_names(self):
        """ Returns a list with the coordinate names.
        """
        if (self.coordinates is not None):
            return [c.unit.name for c in self.coordinates]
        else:
            return []

    def del_coordinate(self,name):
        if (self.coordinates is None):
            raise ValueError("Coordinate '" + name + "' is not present in data object.")
        for i in range(len(self.coordinates)):
            if (self.coordinates[i].unit.name == name):
                break
        else:
            raise ValueError("Coordinate "+name+" not found in data object.")
        del self.coordinates[i]
        if (self.coordinates == []):
            self.coordinates = None

    def get_coordinate_object(self, name):
        """ Returns the Coordinate class having the given name.
        """
        if (self.coordinates is not None):
            for c in self.coordinates:
                if c.unit.name == name:
                    break
            else:
                raise ValueError("Coordinate '" + name + "' is not present in data object.")
            return c
        else:
            raise ValueError("Coordinate '" + name + "' is not present in data object.")


    def add_coordinate_object(self,coordinate,index=None):
        """ Adds a flap.Coordinate instance to the list of coordinates
            If index is not set adds to the end of the list. Otherwise adds to the position
            shown by index. (0 is beginning of list.)
        """
        if (self.coordinates is None):
            self.coordinates = [coordinate]
        else:
            if (type(self.coordinates) is not list):
                self.coordinates = [self.coordinates]
            if (index is not None):
                self.coordinates.insert(index,coordinate)
            else:
                self.coordinates.append(coordinate)

    def add_coordinate(self, coordinates=None, data_source=None, exp_id=None, options=None):
        """ This is a general coordinate conversion interface.
            Adds the requested coordinate(s) to the data_object.
        INPUT:
            coordinates: List of coordinates to add. (string array)
            data_source: Optional data_source. Use this not the one in data_object
            exp_id: Optional exp_id. Use this not the one in data_object
            options: Dictionary of options.
        Returns the modified data object. Note that the input data object remanins the same.
        """
        if (data_source is not None):
            _data_source = data_source
        else:
            _data_source = self.data_source
        if (coordinates is None):
            return self
        _coordinates = coordinates
        if (type(_coordinates) is not list):
            _coordinates = [_coordinates]
        # Checking whether the coordinates are already present
        for coord in self.coordinates:
            try:
                i = _coordinates.index(coord.unit.name)
                del _coordinates[i]
                if (len(_coordinates) == 0):
                    return _data_source
            except (ValueError, IndexError):
                pass

        # Finding the add_coordinate function for this data source
        try:
            f = get_addcoord_function(_data_source)
        except ValueError as e:
            raise e
        if (f is None):
            raise ValueError("No add_coordinate function is associated with data source "
                             + _data_source)
        try:
            d = f(self,coordinates=_coordinates,exp_id=exp_id, options=options)
        except Exception as e:
            raise e
        return d

    def coordinate_change_indices(self, name):
        """ Returns the indices to the data array for which the coordinate changes.
            The returned value is a tuple of indices, the number of elements equals the
            dimension of the data. This can be directly used to get the coordinate values
            using coordinate()
        name: Coordinate name (string)
        """
        for c in self.coordinates:
            if c.unit.name == name:
                break
        else:
            raise ValueError("Coordinate '" + name + "' is not present in data object.")
        index = [0]*len(self.data_shape)
        for i in c.dimension_list:
            index[i] = ...

    def to_intervals(self,coordinate):
        """ Create an Intervals class object from either the data error ranges or
            the coordinate value ranges.
        """
        if (self.data_unit.name != coordinate):
            try:
                coord_obj = self.get_coordinate_object(coordinate)
            except ValueError:
                raise ValueError("To use a flap.DataObject as intervals either the data_unit.name or a coordinate name should be identical to the coordinate.")
            if (coord_obj.value_ranges is None):
                raise ValueError("To use a coordinate as interval it must have ranges set.")
            if (len(coord_obj.dimension_list) > 1):
                raise ValueError("To use a coordinate as interval it must change along maximum one dimension.")
            if (coord_obj.mode.equidistant):
                if (len(coord_obj.dimension_list) == 0):
                    number = 1
                    step = None
                else:
                    number = self.shape[coord_obj.dimension_list[0]]
                    step = coord_obj.step[0]
                if (coord_obj.mode.range_symmetric):
                    intervals = Intervals(coord_obj.start - coord_obj.value_ranges,
                                          coord_obj.start + coord_obj.value_ranges,
                                          step = step,
                                          number = number
                                          )
                else:
                    intervals = Intervals(coord_obj.start - coord_obj.value_ranges[0],
                                          coord_obj.start + coord_obj.value_ranges[1],
                                          step = step,
                                          number = number
                                          )
            else:
                index = [0] * len(intervals.shape)
                index[coord_obj.dimension_list[0]] = ...
                try:
                    d, d_low, d_high = coord_obj.data(data_shape=self.data_shape,index=index)
                except Exception as e:
                    raise e
                intervals = Intervals(d_low, d_high)
        else:
            if ((self.data is None) or (self.error is None)):
                raise ValueError("To use a data object as interval it must have data and error.")
            if (type(self.error) is list):
                intervals = Intervals(self.data.flatten() - self.error[0].flatten(),
                                      self.data.flatten() + self.error[1].flatten()
                                      )
            else:
                intervals = Intervals(self.data.flatten() - self.error.flatten(),
                                      self.data.flatten() + self.error.flatten()
                                      )
        return intervals

    def slicing_to_intervals(self,slicing):
        """ Convert a multi-slicing description to an Intervals object.
            For possibilities see DataObject.slice_data().
        """
        if (type(slicing) is not dict):
            raise TypeError("slicing_to_intervals(): slicing should be a dictionary.")
        if (len(slicing.keys()) != 1):
            raise ValueError("slicing_to_intervals(): slicing dictionary should have only one element.")

        slicing_coord_name = list(slicing.keys())[0]
        slicing_coord = self.get_coordinate_object(slicing_coord_name)
        slicing_description = slicing[slicing_coord_name]

        # Getting the range of the slicing coordinate
        r, r_ext = slicing_coord.data_range(data_shape=self.shape)

        if ((type(slicing_description) is DataObject) and
            (slicing_description.data_unit.name == slicing_coord.unit.name)):
            if (type(slicing_description.error) is list):
                err1 = slicing_description.error[0].flatten()
                err2 = slicing_description.error[1].flatten()
            else:
                err1 = slicing_description.error.flatten()
                err2 = err1
            data = slicing_description.data.flatten()
            intervals = Intervals(data-err1, data+err2)
        elif ((type(slicing_description) is DataObject) and
              (slicing_description.data_unit.name != slicing_coord.unit.name)):
            try:
                coord_obj = slicing_description.get_coordinate_object(slicing_coord.unit.name)
            except Exception as e:
                raise e
            if (not coord_obj.isnumeric()):
                raise ValueError("Cannot do multi-slice with string type coordinate.")
            if (len(coord_obj.dimension_list) is not 1):
                raise ValueError("Cannot do multi-slice with coordinate changing along multiple dimensions.")
            if (coord_obj.mode.equidistant):
                if (coord_obj.mode.range_symmetric):
                    start = coord_obj.start - coord_obj.value_ranges
                    stop = coord_obj.start + coord_obj.value_ranges
                else:
                    start = coord_obj.start - coord_obj.value_ranges[0]
                    stop = coord_obj.start + coord_obj.value_ranges[1]
                intervals = Intervals(start,
                                      stop,
                                      step=coord_obj.step[0],
                                      number=slicing_description.shape[coord_obj.dimension_list[0]])
            else:
                try:
                    c, c_low, c_high = coord_obj.data(data_shape=slicing_description.shape,index='...')
                except Exception as e:
                    raise e
                intervals = Intervals(c_low, c_high)
        elif (type(slicing_description) is Intervals):
            intervals = slicing_description
        else:
            raise TypeError("Invalid slicing description.")
        return intervals

    def proc_interval_limits(self, coordinate, intervals=None):
        """ Determine processing interval limits, both in coordinates and data indices.
            This is a helper routine for all functions which do some calculation as a function
            of one coordinate and allow processing only a set of intervals instead of the whole dataset.
            INPUT:
                coordinate: Name of the coordinate along which calculation will be done. This must
                            change only along a single data dimension.
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
            Return value:
                calc_int, calc_int_ind, sel_int, sel_int_ind
                Each return value is a list of numpy arrays: [start, end]
                The calc_xxx values are for the calculation coordinate, the sel_xxx are for the
                selection coordinate. xxx_int is in coordinate values, xxx_int_ind is in data index
                values which follows the Python convention, end index is not included.
                The index start indices will be always smaller than the end indices.
        """
        # Getting the coordinate object for the processing coordinate
        try:
            coord_obj = self.get_coordinate_object(coordinate)
        except Exception as e:
            raise e
            
        if (len(coord_obj.dimension_list) != 1):
            raise ValueError("Processing coordinate may change only along one dimension.")
            
        if (intervals is None):
            _intervals = None
            sel_coordinate = coordinate
        elif (type(intervals) is dict):
            sel_coordinate = list(intervals.keys())[0]
            _intervals = intervals[sel_coordinate]
        else:
            sel_coordinate = coordinate
            _intervals = intervals
        if (sel_coordinate == coordinate):
            sel_coord_obj = coord_obj
        else:
            try:
                sel_coord_obj = self.get_coordinate_object(sel_coordinate)
            except Exception as e:
                raise e

        if (len(sel_coord_obj.dimension_list) != 1):
            raise ValueError("Selection coordinate may change only along one dimension.")

        #Converting all input interval descriptions to Interval object
        if (_intervals is None):
            input_intervals = None
        elif ((type(_intervals) is list) and (len(_intervals) == 2)):
            input_intervals = Intervals(_intervals[0], _intervals[1])
        elif (type(_intervals) is Intervals):
            input_intervals = _intervals
        elif (type(_intervals) is DataObject):
            try:
                input_intervals = _intervals.to_intervals(coordinate)
            except Exception as e:
                raise e
        else:
            raise ValueError("Invalid interval description.")
        
        # Determining selection coordinate index ranges
        if (input_intervals is None):
            sel_int_ind = [np.array([0],dtype=np.int32), 
                             np.array([self.shape[sel_coord_obj.dimension_list[0]]],dtype=np.int32)]
                        # Converting from index to coordinate
            ind = [0]*len(self.shape)
            ind[sel_coord_obj.dimension_list[0]] = np.concatenate((sel_int_ind[0], sel_int_ind[1] - 1))
            c, cl, ch = sel_coord_obj.data(data_shape=self.shape, index=ind)
            ind_first = copy.deepcopy(ind)
            ind_first[sel_coord_obj.dimension_list[0]] = 0
            ind_last = copy.deepcopy(ind)
            ind_last[sel_coord_obj.dimension_list[0]] = 1
            sel_int = [np.array([c[tuple(ind_first)]]), np.array([c[tuple(ind_last)]])]
        else:
            # Limiting the range to the coordinte values
            limits, limits_range = sel_coord_obj.data_range(data_shape=self.shape)
            # Determining interval start and end values
            l1, l2 = input_intervals.interval_limits(limits=limits,partial_intervals=True)
            sel_int = [l1,l2]
            # Converting to indices
            l = np.concatenate((l1,l2))
            ind = self.index_from_coordinate(sel_coordinate, l)
            sel_int_ind = [ind[0:len(ind)//2], ind[len(ind)//2:]]
                
        # Determining interval limits for the calculating coordinate
        if (sel_coordinate == coordinate):
            calc_int = sel_int
            calc_int_ind = sel_int_ind
        else:
            if (coord_obj.dimension_list[0] == sel_coord_obj.dimmension_list[0]):
                # If the calculating corodinate changes along the same dimension as the selection
                # then the same index ranges are used
                calc_int_ind = sel_int_ind
            else:
                # otherwise the whole calculation index range is used
                calc_int_ind = [np.array([0],dtype=np.int32), 
                                np.array([self.shape[coord_obj.dimension_list[0]]],dtype=np.int32)]
            # Converting from index to coordinate
            ind = [0]*len(self.shape)
            ind[calc_obj.dimension_list[0]] = np.concatenate((calc_int_ind[0],calc_int_ind[1]))
            c, cl, ch = coord_obj(data_shape=self.shape, index=ind)
            calc_int = [c[0:len(c)/2], c[len(c)/2:]]
        
        return calc_int, calc_int_ind, sel_int, sel_int_ind
        
    def coordinate(self, name, index=...):
        """ Returns the coordinates of a subarray of the data array.
        name: Coordinate name (string)
        index: The indexes into the data array (tuple with various elements, see Coordinate.data())

        returns 3 np.arrays:
            data: the coordinates. The number of dimension is the same as the dimension of
                  the data array, but the number  of elements are taken from index.
            data_low: low ranges, same shape as data. None if no range data is present
            data_high: high ranges, same shape as data. None if no range data is present
        """
        if (type(name) is not str):
            raise TypeError("Invalid coordinate name.")
        for c in self.coordinates:
            if c.unit.name == name:
                break
        else:
            raise ValueError("Coordinate '" + name + "' is not present in data object.")
        try:
            d,d_low,d_high = c.data(data_shape=self.shape, index=index)
        except Exception as e:
            raise e
        return d,d_low,d_high
    
    def coordinate_range(self, name, index=...):
        """ Returns the ranges of a coordinate.
        name: Coordinate name (string)

        Returns the data range and the data range with errors for the coordinate. Both are lists.
        """
        for c in self.coordinates:
            if c.unit.name == name:
                break
        else:
            raise ValueError("Coordinate '" + name + "' is not present in data object.")
        try:
            dr, dr_e = c.data_range(data_shape=self.shape)
        except Exception as e:
            raise e
        return dr, dr_e


    def index_from_coordinate(self, name, coord_values):
        """ Returns the closest data indices of the coordinate values.
            Coordinates should change only along one dimension.
            It is assumed that coordinates change monotonically.
        INPUT:
            name: Coordinate name (string)
            coord_values: The coordinate values to convert to indices.
                          (numpy array or list is scalar)
        Return value:
            Returns the indices in the same format as the input coordinates.
        """
        try:
            coord_obj = self.get_coordinate_object(name)
        except Exception as e:
            raise e
        if (len(coord_obj.dimension_list) != 1):
            raise ValueError("To convert to indices coordinate should change along one dimension.")
        if (type(coord_values) == list):
            _coord_values = np.ndarray(coord_values)
        elif (np.isscalar(coord_values)):
            _coord_values = np.array([coord_values])
        elif (type(coord_values) == np.ndarray):
            _coord_values = coord_values
        else:
            raise TypeError("Invalid type for coord_values.")

        ind = [0]*len(self.shape)
        ind[coord_obj.dimension_list[0]] = ...
        coordinate_data, dl, dh = coord_obj.data(data_shape=self.shape,index=ind)
        coordinate_data = coordinate_data.flatten()
        data_index = np.arange(self.shape[coord_obj.dimension_list[0]],dtype=np.int32)
        sort_ind = np.argsort(coordinate_data)
        coordinate_data_sorted = coordinate_data[sort_ind]
        data_index_sorted = data_index[sort_ind]
        _index_values = np.int32(np.interp(_coord_values,
                                           coordinate_data_sorted,
                                           data_index_sorted))
        if (type(coord_values) == list):
            index_values = list(_index_values)
        elif (np.isscalar(coord_values)):
            index_values = _index_values[0]
        elif (type(coord_values) == np.ndarray):
            index_values = _index_values

        return index_values

    def coordinate_change_dimensions(self,name):
        """ Return the list of dimensions of the data array along which
        the named coordinate changes
        """
        for c in self.coordinates:
            if c.unit.name == name:
                break
        else:
            raise ValueError("Coordinate '"+name+"' is not present in data object.")
        return c.change_dimensions()

    def coordinate_nochange_dimensions(self,name):
        """ Return the list of dimensions of the data array along which
        the named coordinate changes
        """
        for c in self.coordinates:
            if c.unit.name == name:
                break
        else:
            raise ValueError("Coordinate '" + name + "' is not present in data object.")
        return c.change_dimensions(self.data.shape)

    def _plot_error_ranges(self, index=None):
        """ Helper function to return error low and high limits as needed by matplotlib
        """
        if (self.error is None):
            return None
        if (index is None):
            _index = tuple([slice(0,d) for d in self.shape])
        else:
            _index = index
        if (type(self.error) is list):
            err = np.ndarray((2,self.error[0].size), dtype=self.error[0].dtype)
            err[0,:] = self.error[0][_index].flatten()
            err[1,:] = self.error[1][_index].flatten()
            return err
        else:
            return self.error[_index].flatten()

    def _plot_coord_ranges(self,coord, c_data, c_low, c_high):
        """ Helper function to return error low and high limits from coordiniate data
            in the format needed by matplotlib
        """
        if (c_low is None):
            return None
        if (coord.mode.range_symmetric):
            return c_data.flatten() - c_low.flatten()
        else:
            err  = np.ndarray((2,c_low.size),dtype=c_low.dtype)
            err[0,:] = c_data.flatten() - c_low.flatten()
            err[1,:] = c_high.flatten() - c_data.flatten()
            return err

    def plot(self,
             axes=None,
             slicing=None,
             slicing_options=None,
             summing=None,
             options=None,
             plot_type=None,
             plot_options=None,
             plot_id=None):
        """
        Plot a data object.
        axes: A list of coordinate names (strings). They should be one of the coordinate
              names in the data object or 'Data'
              They describes the axes of the plot.
              If the number of axes is less than the number required for the plot, 'Data' will be added.
              If no axes are given default will be used depending on the plot type. E.g. for
              x-y plot the default first axis is the firs coordinate, the second axis is 'Data'
        plot_type: The plot type (string)
              'xy': Simple 1D plot. Default axes are: first coordinate, Data
              'multi xy': In case of 2D data plots 1D curves with vertical shift
                          Default x axis is first coordinate, y axis is Data
                          The signals are named in the label with the 'Signal name' coordinate or the one
                          named in options['Signal name']
        plot_options: Dictionary. Will be passed over to the plot call.
        slicing, summing: arguments for slice_data. Slicing will be applied before plotting.
        options:
            Matplotlib options like xtitle, xlimit, etc
            Plot options:
                'All points' True or False
                             default is False. If True will plot all points otherwise will use
                             the sample_to_plot function
                'Error'      True: Plot all error bars (default: True)
                             False: Do not plot errors
                             number > 0: Plot this many error bars in plot
                'Y separation' Vertical separation of curves in multi xy plot. For linear scale this will
                               be added to consecutive cureves. For Log scale consecutive curves will be
                               multiplied by this.
                'Log x' : Logscale X axis
                'Log y' : Logscale Y axis
        """

        try:
            return _plot(self,
                         axes=axes,
                         slicing=slicing,
                         slicing_options=slicing_options,
                         summing=summing,
                         options=options,
                         plot_type=plot_type,
                         plot_options=plot_options,
                         plot_id=plot_id)
        except Exception as e:
            raise e
            
            
    def __moveaxis(self,source,destination):
        """ This is an internal function to move an axis in the data and error arrays.
            Does not do anything to the coordinates.
        """
        self.data = np.moveaxis(self.data,source,destination)
        if (self.error is not None):
            if (type(self.error) is list):
                for i in range(2):
                    self.error[i] = np.moveaxis(self.error[i],source,destination)
            else:
                self.error = np.moveaxis(self.error,source,destination)

    def __create_arrays(self, source_object, shape):
        """ This is an internal function to create empty data and error arrays.
        """

        self.data = np.empty(shape,dtype=source_object.data.dtype)
        if (source_object.error is not None):
            if (type(source_object.error) is list):
                self.error = []
                for i in range(2):
                    self.error.append(np.empty(shape,dtype=source_object.error[i].dtype))
            else:
                self.error = np.empty(shape,dtype=source_object.error.dtype)

    def __copy_data(self, source_object, destination_ind, source_ind):
        """ This is an internal function to copy data from one data object to the other with
            possibly different shape.
        """

        self.data[destination_ind] = source_object.data[source_ind]
        if (source_object.error is not None):
            if (type(source_object.error) is list):
                for i in range(2):
                    self.error[i][destination_ind] = source_object.error[i][source_ind]
            else:
                self.error[destination_ind] = source_object.error[source_ind]

    def __fill_nan(self, ind):
        """ This is an internal function to fill certain elements in the data and error with NaN.
        """
        if (self.data.dtype.kind is 'f'):
            self.data[ind] = np.NaN
        elif (self.data.dtype.kind is 'c'):
            self.data[ind] = complex(np.NaN,np.NaN)
        elif ((self.data.dtype.kind is 'u') or (self.data.dtype.kind is 'i')):
            self.data[ind] = 0
        elif (self.data.dtype.kind is 'O'):
            self.data[ind] = None

        if (self.error is not None):
            if (type(self.error) is list):
                for i in range(2):
                    if (self.error.dtype.kind is 'f'):
                        self.error[i][ind] = np.NaN
                    elif (self.error.dtype.kind is 'c'):
                        self.error[i][ind] = complex(np.NaN,np.NaN)
                    elif ((self.error.dtype.kind is 'u') or (self.error.dtype.kind is 'i')):
                        self.error[i][ind] = 0
                    elif (self.error.dtype.kind is 'O'):
                        self.error[i][ind] = None
            else:
                if (self.error.dtype.kind is 'f'):
                    self.error[ind] = np.NaN
                elif (self.error.dtype.kind is 'c'):
                    self.error[ind] = complex(np.NaN,np.NaN)
                elif ((self.error.dtype.kind is 'u') or (self.error.dtype.kind is 'i')):
                    self.error[ind] = 0
                elif (self.error.dtype.kind is 'O'):
                    self.error[ind] = None

    def __check_coords_after_simple_slice(self,sliced_dimensions, sliced_removed, ind_slice_coord, original_shape, 
                                          dimension_mapping, slicing_equidistant,slicing_coord,_options):
        """ This is an internal function to modify the coordinates after a simple (non-interval) slice operation
        """
        
        for check_coord in self.coordinates:
            # Do nothing for scalar dimension
            if (len(check_coord.dimension_list) == 0):
                continue

            # Determine common dimension list of this coordinate with the sliced one
            common_dims = []
            for d in check_coord.dimension_list:
                try:
                    sliced_dimensions.index(d)
                    common_dims.append(d)
                except ValueError:
                    pass

            # If there are common dimensions then must change the description
            if (len(common_dims) != 0):
                # Checking if this coordinate can be equidistant. Two cases qualify:
                # 1. If this is the slicing coordinate and the slicing is equidistant
                # 2. If this is not the slicing coordinate but both this and the slicing coordinates are equidistant
                #    along a single dimension
                if (slicing_equidistant and
                    ((check_coord == slicing_coord) 
                     or ((check_coord != slicing_coord)  
                         and (len(check_coord.dimension_list) == 1)
                         and (check_coord.mode.equidistant)
                         and (len(slicing_coord.dimension_list) == 1)
                         and slicing_coord.mode.equidistant
                        )
                     )
                    ):
                    # if this coordinate is equidistant and changes along one dimension 
                    # and slicing is also equidistant
                    if (type(ind_slice_coord) is slice):
                        check_coord.start = check_coord.start    \
                                            + ind_slice_coord.start   \
                                               * check_coord.step[0]
                        check_coord.step = check_coord.step[0] * ind_slice_coord.step
                    else:
                        check_coord.start = check_coord.start    \
                                            + ind_slice_coord[0]   \
                                               * check_coord.step[0]
                        check_coord.step = check_coord.step[0] * (ind_slice_coord[-1]  - ind_slice_coord[0]) \
                                                                 / (ind_slice_coord.size - 1)                      
                    if (sliced_removed):
                       check_coord.mode.equidistant = False
                       check_coord.values = check_coord.start
                    check_coord.shape = None
                else:
                    # This coordinate will be non-equidistant
                    # Determining dimension list containing dimensions from this coordinate
                    # and the sliced ones
                    unified_dims = unify_list(check_coord.dimension_list, sliced_dimensions)
                    # Getting coordinate data for this unified array
                    index = [0]*len(original_shape)
                    for d in unified_dims:
                        index[d] = ...
                    original_coord_data, data_low, data_high = check_coord.data(data_shape=original_shape, index=index)
                    # The shape of the unified dimension space
                    unified_shape = [original_shape[d] for d in unified_dims]
                    # Reshaping to this unified shape, other dimensions have 1 element which will be removed
                    new_coord_data = np.reshape(original_coord_data,tuple(unified_shape))
                    # Determining indices in unified dimension list which are flattened
                    flattened_in_unified = []
                    for i_d in range(len(unified_dims)):
                        try:
                            sliced_dimensions.index(unified_dims[i_d])
                            flattened_in_unified.append(i_d)
                        except ValueError:
                            pass
                    # Flattening these dimensions in the coordinate data matrix
                    new_coord_data, flat_dim_mapping = flatten_multidim(new_coord_data, flattened_in_unified)
                    # Doing the same with coordinate ranges
                    if (data_low is not None):
                        data_low = np.reshape(data_low,tuple(unified_shape))
                        data_low, xxx = flatten_multidim(data_low, flattened_in_unified)
                        if (not check_coord.mode.range_symmetric):
                           data_high = np.reshape(data_high,tuple(unified_shape))
                           data_high, xxx = flatten_multidim(data_high, flattened_in_unified)
                    # Creating a list of slices for all dimension, so as all elements are copied
                    ind_slice = [slice(0,x) for x in new_coord_data.shape]
                    if ((_options['Interpolation'] == 'Closest value') or (type(ind_slice_coord) is slice)
                        or not check_coord.isnumeric()):
                        # In the flattened dimensions inserting the slicing index array or slice
                        if (type(ind_slice_coord) is slice):
                            ind_slice[flattened_in_unified[0]] = ind_slice_coord
                        else:
                            ind_slice[flattened_in_unified[0]] = np.round(ind_slice_coord).astype(np.int32)
                        # Selecting the coordinate data for the remaining data elements
                        check_coord.values = copy.deepcopy(np.squeeze(new_coord_data[tuple(ind_slice)]))
                        if (check_coord.values.ndim == 0):
                            check_coord.values = check_coord.values.reshape([1])
                        if (data_low is not None):
                            data_low = data_low[tuple(ind_slice)]
                            if (data_low.ndim == 0):
                                data_low = data_low.reshape([1])
                            if (not check_coord.mode.range_symmetric):
                                data_high = data_high[tuple(ind_slice)]                        
                                if (data_high.ndim == 0):
                                    data_high = data_high.reshape([1])
                    elif (_options['Interpolation'] == 'Linear'):
                        # ind_slice_coord is numpy array
                        ind_slice_coord_1 = np.trunc(ind_slice_coord).astype(np.int32)
                        ind_slice_coord_2 = ind_slice_coord_1 + 1
                        ind = np.nonzero(ind_slice_coord_2 >= new_coord_data.shape[flattened_in_unified[0]])[0]
                        if (ind.size != 0):
                            ind_slice_coord_2[ind] = ind_slice_coord_1[ind] 
                        # Insert the lower slicing indices into the flattened dimension and get the two base points
                        # for interpolation
                        ind_slice_1 = copy.deepcopy(ind_slice)
                        ind_slice_2 = copy.deepcopy(ind_slice)
                        ind_slice_1[flattened_in_unified[0]] = ind_slice_coord_1
                        d1 = new_coord_data[tuple(ind_slice_1)]
                        ind_slice_2[flattened_in_unified[0]] = ind_slice_coord_2
                        d2 = new_coord_data[tuple(ind_slice_2)]
                        # Reshaping ind_slice_coord_1 and ind_slice_coord to an array with 1 elements in all
                        # dimensions except the coordinate dimension. This will broadcast with d2 and d2.
                        interplol_weight_shape = [1] * len(new_coord_data.shape)
                        interplol_weight_shape[flattened_in_unified[0]] = d1.shape[flattened_in_unified[0]]
                        ind_slice_interp_1 = ind_slice_coord_1.reshape(tuple(interplol_weight_shape))  
                        ind_slice_interp = ind_slice_coord.reshape(tuple(interplol_weight_shape))                                        
                        check_coord.values = (d2 - d1) * (ind_slice_interp - ind_slice_interp_1) + d1
                        if (data_low is not None):
                            d1 = data_low[tuple(ind_slice_1)]
                            d2 = data_low[tuple(ind_slice_2)]
                            data_low = (d2 - d1) * (ind_slice_interp - ind_slice_interp_1) + d1
                            if (not check_coord.mode.range_symmetric):
                                d1 = data_high[tuple(ind_slice_1)]
                                d2 = data_high[tuple(ind_slice_2)]
                                data_high = (d2 - d1) * (ind_slice_interp - ind_slice_interp_1) + d1
                    else:
                        raise ValueError("Internal error: cannot handle interpolation method '{:s}'".format(_options['Interpolation']))
                    if (data_low is not None):
                        if (not check_coord.mode.range_symmetric):
                           check_coord.value_ranges = [check_coord.values - data_low,
                                                       data_high - check_coord.values]
                        else:
                           check_coord.value_ranges = check_coord.values - data_low
                    check_coord.mode.equidistant = False
                    check_coord.shape = check_coord.values.shape

                #  Updating the dimension list for the case when there are common
                # dimensions with the slicing coord
                new_dimension_list = []
                for d in range(len(original_shape)):
                    try:
                        # If in old dimension list
                        check_coord.dimension_list.index(d)
                        # If it was not flattened
                        if (dimension_mapping[d] is not None):
                            new_dimension_list.append(dimension_mapping[d])
                    except ValueError:
                        pass
                    # If this is the first flattened dimension
                    # and it was not removed
                    if ((d == sliced_dimensions[0])   \
                             and not sliced_removed):
                        new_dimension_list.append(d)
            else:
                # There are no common dimensions
                new_dimension_list = \
                    [dimension_mapping[x] for x in check_coord.dimension_list]
                for d in new_dimension_list:
                    if (d is None):
                        raise("Internal error in dimension mapping. None dimension.")

            if (not check_coord.mode.equidistant and (np.isscalar(check_coord.values) or (check_coord.values.size == 1))):
                check_coord.dimension_list = []
            else:
                check_coord.dimension_list = new_dimension_list

    def __check_coords_after_multi_slice(self,slicing,
                                         original_shape,
                                         joint_slices,
                                         joint_dimension_list,
                                         dimension_mapping,
                                         slice_description,
                                         interval_dimension,
                                         in_interval_dimension,
                                         n_int,
                                         n_in_int):
        """ Internal function to modify all coordinate descriptions after a mult-slice process. This or
            __check_coordds_after_simple_slice is called after each slice operation with a coordinate.
            In the slicing process before this call the dimensions in joint_dimension_list were flattened and the data distributed
            to the interval_dimension and in_interval_dimension.
            self is a copy of the original data object before slice with the data already sliced as
            described above.
            slicing: The slincing input of self.slice_data()
            original_shape: The original data shape before this slicing operation
            joint slices: The index of the keys in slicing.keys() which were done jointly
            joint_dimension_list: List of data dimension which were flattened
            dimension_mapping: mapping from original to new dimension numbers
            slice_description: Describing the indices in the data array for the intervals.
                                There are 3 options:
                                  slice: (for 1-coordinate slice only)
                                  dictionary with 'Starts', 'Stops' keys: giving the interval
                                     start and stop indices with numpy arrays.
                                     Number of dimensions refer to the number of joint slice coordinates.
                                  list of numpy arrays: indices. If multiple coordinate multi-slice then
                                    list of lists, slicing_description[i][j][k] giving the numpy array
                                    for interval i,j,k (in this 3 with 3 joint slicing coordinates)
            interval_dimension: dimension number (in new numbering) for the intervals
            in_interval_dimension: dimension number (in new numbering) inside the intervals
            n_int: Number of intervals
            n_in_int: number of elements in intervals
        """
        slicing_coord_names = ""
        for i in joint_slices:
            coord = self.get_coordinate_object(list(slicing.keys())[i])
            if (len(slicing_coord_names) != 0):
                slicing_coord_names += ","
            slicing_coord_names += coord.unit.name
        # This will contain a list of coordinate indices which has to be deleted at the end
        del_coords = []
        for i_check_coord in range(len(self.coordinates)):
            check_coord = self.coordinates[i_check_coord]

            # Do nothing for scalar dimension
            if (len(check_coord.dimension_list) == 0):
                continue

            # Determine common dimension list of this coordinate with the sliced ones
            # and the oned which will remain after removing the sliced ones
            common_dims = []
            # the next list contains the indices to commoon_dims in check_coord.dimension_list
            common_dims_index = []
            remaining_dimension_list = []
            for i_d in range(len(check_coord.dimension_list)):
                d = check_coord.dimension_list[i_d]
                try:
                    joint_dimension_list.index(d)
                    common_dims.append(d)
                    common_dims_index.append(i_d)
                except ValueError:
                    remaining_dimension_list.append(d)

            if (len(common_dims) != 0):
                # This coordinate has common dimension with the flattened dimensions
                # This coordinate <c> will be replaced by .
                # <c> this coordinate name
                # Rel. <x> in interval
                # Start <x> in interval
                mode = check_coord.mode
                if (check_coord.isnumeric()):
                    # These coordinates make sense only for numerical values
                    # For string the original coordinate will be kept
                    cc_new_in_int = copy.deepcopy(Coordinate(name="Rel. " +
                                                             check_coord.unit.name +
                                                             " in int("+
                                                             slicing_coord_names + ")",
                                                             unit=check_coord.unit.unit,
                                                             mode=mode))
                    add_in_int_coord = True
                    cc_new_int = copy.deepcopy(Coordinate(name="Start " +
                                                          check_coord.unit.name +
                                                          " in int(" +
                                                          slicing_coord_names + ")",
                                                          unit=check_coord.unit.unit,
                                                          mode=mode))
                    add_int_coord = True
                    del_coords.append(i_check_coord)
                    change_check_coord = False
                else:
                    add_in_int_coord = False
                    add_int_coord = False
                    change_check_coord = True

                coord_int_filled = False
                coord_in_int_filled = False

                if (check_coord.mode.equidistant):
                    if ((len(joint_dimension_list) == 1)
                          and (len(joint_slices) == 1)
                          and (type(slice_description) is slice)):
                        # in this case the coordinate for the interval starts will be equidistant
                        # This means that common_dims has only 1 element. Adding the steps to the
                        # first selected element to the start value
                        cc_new_int.start = check_coord.start    \
                                            + slice_description.start  \
                                              * check_coord.step[common_dims_index[0]]
                        # Increrasing the step size
                        cc_new_int.step = [check_coord.step[common_dims_index[0]]\
                                                                * slice_description.step]
                        cc_new_int.shape = []
                        cc_new_int.value_ranges = check_coord.value_ranges
                        # The dimmension list will be the same as check_coords, except that
                        # the common_dim is replaced by the interval dimension
                        cc_new_int.dimension_list = copy.deepcopy(check_coord.dimension_list)
                        cc_new_int.dimension_list[common_dims_index[0]] = interval_dimension
                        coord_int_filled = True
                    if (((len(joint_dimension_list) == 1)
                          and (len(joint_slices) == 1)
                          and (type(slice_description) is slice)
                              or type(slice_description) is dict)):
                        # The coordinate in the intervals will be equidistant and the same in all intervals
                        cc_new_in_int.start = type(check_coord.start)(0)
                        cc_new_in_int.step = check_coord.step[common_dims_index[0]]
                        cc_new_in_int.shape = []
                        cc_new_in_int.value_ranges = check_coord.value_ranges
                        cc_new_in_int.dimension_list = [in_interval_dimension]
                        coord_in_int_filled = True

                if (not coord_int_filled or not coord_in_int_filled):
                    # We need to get the coordinate data and extract the coordinates
                    unified_dimensions = unify_list(check_coord.dimension_list, joint_dimension_list)
                    index = [0] * len(original_shape)
                    for d in unified_dimensions:
                        index[d] = ...
                    coord_data, data_low, data_high = check_coord.data(data_shape=original_shape,
                                                                       index=index)
                    # The shape of the unified dimension space
                    unified_shape = [original_shape[d] for d in unified_dimensions]
                    # Reshaping to this shape, other dimensions have 1 element which will be removed
                    coord_data = np.reshape(coord_data,tuple(unified_shape))
                    # Determining indices in unified dimension list which are flattened
                    flattened_in_unified = []
                    for i_d in range(len(unified_dimensions)):
                        try:
                            joint_dimension_list.index(unified_dimensions[i_d])
                            flattened_in_unified.append(i_d)
                        except ValueError:
                            pass
                    # Flattening these dimensions in the coordinate data matrix
                    coord_data, flat_dim_mapping = flatten_multidim(coord_data, flattened_in_unified)
                if (add_int_coord and not coord_int_filled):
                    # Filling data for the in-interval coordinate
                    # Creating a list of slices for all dimension, so as all elements are copied
                    ind = [slice(0,x) for x in coord_data.shape]
                    if (len(joint_slices) == 1):
                        # In the flattened dimensions inserting the slicing index array
                        if (type(slice_description) is slice):
                            ind[flattened_in_unified[0]] = slice_description
                        elif (type(slice_description) is dict):
                            ind[flattened_in_unified[0]] = slice_description['Starts']
                        else:
                            start_ind = np.ndarray(len(slice_description),
                                                   dtype=slice_description[0].dtype)
                            for i in range(len(slice_description)):
                                start_ind[i] = slice_description[i][0]
                            ind[flattened_in_unified[0]] = start_ind
                        cc_new_int.values = coord_data[tuple(ind)]
                        if (data_low is not None):
                            if (check_coord.mode.range_symmetric):
                                data_low = data_low[tuple(ind)]
                                cc_new_int.value_ranges = cc_new_int.values - data_low
                                np_moveaxis(cc_new_int.value_ranges,
                                            flattened_in_unified[0],
                                            cc_new_int.values.ndim-1)
                            else:
                                data_low = data_low[tuple(ind)]
                                data_low = data_low[tuple(ind)]
                                cc_new_int.value_ranges = [cc_new_int.values - data_low,
                                                           data_high - cc_new_int.values]
                                for i in range(2):
                                    cc_new_int.value_ranges[i] = np.moveaxis(cc_new_int.value_ranges[i],
                                                                             flattened_in_unified[0],
                                                                             cc_new_int.values.ndim-1)
                        cc_new_int.values = np.moveaxis(cc_new_int.values,
                                                        flattened_in_unified[0],
                                                        cc_new_int.values.ndim-1)
                        cc_new_int.shape = cc_new_int.values.shape
                        cc_new_int.dimension_list = [dimension_mapping[d] for d in remaining_dimension_list]
                        for d in cc_new_int.dimension_list:
                            if (d is None):
                                raise("Internal error in dimension mapping. None dimension.")
                        cc_new_int.dimension_list.append(interval_dimension)
                        mode = check_coord.mode
                        mode.equidistant = False
                        cc_new_int.mode = mode
                    else:
                        raise ValueError("Multi-dimension multi-slicing is not implemented yet.")
                if (add_in_int_coord and not coord_in_int_filled or change_check_coord):
                    # Here the coordinate will change in the remaining_dimensions and both the
                    # interval_dimension and in_interval_dimension
                    #  Determining the shape of the output matrix
                    new_shape = list(coord_data.shape)
                    # Removing the flattened shape
                    del new_shape[flattened_in_unified[0]]
                    # Moving the flattened dimension to the end in the existing coord data
                    coord_data = np.moveaxis(coord_data, flattened_in_unified[0], len(coord_data.shape)-1)
                    if (data_low is not None):
                        data_low = np.moveaxis(data_low, flattened_in_unified[0], len(coord_data.shape)-1)
                        data_high = np.moveaxis(data_high, flattened_in_unified[0], len(coord_data.shape)-1)
                    # Creating an index list for the source array
                    ind = [slice(0,n) for n in coord_data.shape]
                    if (len(joint_slices) == 1):
                        if (type(slice_description) is slice):
                            # Adding two new dimensions for the intervals and the data in the intervals
                            new_shape.append(n_in_int)
                            in_interval_dimension_in_coord = len(new_shape) - 1
                            new_shape.append(n_int)
                            interval_dimension_in_coord = len(new_shape) - 1
                            #Creating a new data array with this new_shape
                            new_coord_data = np.empty(new_shape, dtype = check_coord.dtype())
                            if (data_low is not None):
                                new_data_low = np.empty(new_shape, dtype = data_low.dtype)
                                new_data_high = np.empty(new_shape, dtype = data_high.dtype)
                            else:
                                new_data_low = None
                                new_data_high = None
                            # Creating an index list for the destination array
                            ind_out = [slice(0,n) for n in new_coord_data.shape]
                            # Going through the elements in the intervals
                            for i in range(n_in_int):
                                ind_out[in_interval_dimension_in_coord] = i
                                s = slice(slice_description.start + i,
                                          slice_description.stop,
                                          slice_description.step)
                                ind_out[interval_dimension_in_coord] = s
                                if (check_coord.isnumeric):
                                    new_data[tuple(ind_out)] = coord_data[tuple(ind)] - np.amin(coord_data[tuple(ind)])
                                else:
                                    new_data[tuple(ind_out)] = coord_data[tuple(ind)]
                                if (data_low is not None):
                                    new_data_low[tuple(ind_out)] = data_low[tuple(ind)]
                                    new_data_high[tuple(ind_out)] = data_high[tuple(ind)]
                                else:
                                    new_data_low = None
                                    new_data_high = None
                        else:
                            # If slice_description is dict or numpy array
                            # Adding two new dimensions for the intervals and the data in the intervals
                            new_shape.append(n_int)
                            interval_dimension_in_coord = len(new_shape) - 1
                            new_shape.append(n_in_int)
                            in_interval_dimension_in_coord = len(new_shape) - 1
                            #Creating a new data array with this new_shape
                            new_coord_data = np.empty(new_shape, dtype = check_coord.dtype())
                            if (data_low is not None):
                                new_data_low = np.empty(new_shape, dtype = data_low.dtype)
                                new_data_high = np.empty(new_shape, dtype = data_high.dtype)
                            # Creating an index list for the destination array
                            ind_out = [slice(0,n) for n in new_coord_data.shape]
                            # Going through the intervals
                            for i in range(n_int):
                                if (type(slice_description) is dict):
                                    s = slice(slice_description['Starts'][i],
                                              slice_description['Stops'][i])
                                    s_out = slice(0,s.stop-s.start)
                                else:
                                    s = slice_description[i]
                                    s_out = slice(0,s.size)
                                ind[-1] = s
                                ind_out[interval_dimension_in_coord] = i
                                ind_out[in_interval_dimension_in_coord] = s_out
                                if (check_coord.isnumeric()):
                                    new_coord_data[tuple(ind_out)] = coord_data[tuple(ind)] \
                                                                    - np.amin(coord_data[tuple(ind)])
                                else:
                                    new_coord_data[tuple(ind_out)] = coord_data[tuple(ind)]
                                if (data_low is not None):
                                    new_data_low[tuple(ind_out)] = data_low[tuple(ind)]
                                    new_data_high[tuple(ind_out)] = data_high[tuple(ind)]
                                else:
                                    new_data_low = None
                                    new_data_high = None
                                if (s_out.stop < n_in_int):
                                    ind_out[in_interval_dimension_in_coord] = slice(s_out.stop,n_in_int)
                                    if (not check_coord.isnumeric()):
                                        new_coord_data[tuple(ind_out)] = ''
                                    else:
                                        if (check_coord.dtype() is float):
                                            new_coord_data[tuple(ind_out)] = np.NaN
                                        else:
                                            # There is no NaN for integer, we put in 0. There is no other solution
                                            new_coord_data[tuple(ind_out)] = 0
                                        if (new_data_low is not None):
                                            new_data_low[tuple(ind_out)] = np.NaN
                                            new_data_high[tuple(ind_out)] = np.NaN
                            # Moving the interval dimension to the end
                            new_coord_data = np.moveaxis(new_coord_data, len(new_coord_data.shape) - 2, len(new_coord_data.shape) - 1)
                            if (data_low is not None):
                                data_low = np.moveaxis(data_low, len(data_low.shape) - 2, len(data_low.shape) - 1)
                                data_high = np.moveaxis(data_high, len(data_low.shape) - 2, len(data_high.shape) - 1)
                            interval_dimension_in_coord, in_interval_dimension_in_coord \
                                 = in_interval_dimension_in_coord, interval_dimension_in_coord
                        if (add_in_int_coord):
                            cc_new_in_int.values = new_coord_data
                            if (new_data_low is not None):
                                if (cc_new_in_int.mode.range_symmetric):
                                    cc_new_in_int.value_ranges = new_coord_data - new_data_low
                                else:
                                    cc_new_in_int.value_ranges = [new_coord_data - new_data_low,
                                                                  new_data_high - new_coord_data]
                            cc_new_in_int.shape = cc_new_in_int.values.shape
                            cc_new_in_int.dimension_list = [dimension_mapping[d] for d in remaining_dimension_list]
                            for d in cc_new_int.dimension_list:
                                if (d is None):
                                    raise("Internal error in dimension mapping. None dimension.")
                            cc_new_in_int.dimension_list.append(in_interval_dimension)
                            cc_new_in_int.dimension_list.append(interval_dimension)
                            mode = check_coord.mode
                            mode.equidistant = False
                            cc_new_in_int.mode = mode
                        if (change_check_coord):
                            check_coord.values =  new_coord_data
                            if (new_data_low is not None):
                                if (check_coord.mode.range_symmetric):
                                    check_coord.value_ranges = new_coord_data - new_data_low
                                else:
                                    check_coord.value_ranges = [new_coord_data - new_data_low,
                                                                new_data_high - new_coord_data]
                            check_coord.shape = check_coord.values.shape
                            check_coord.dimension_list = [dimension_mapping[d] for d in remaining_dimension_list]
                            for d in check_coord.dimension_list:
                                if (d is None):
                                    raise("Internal error in dimension mapping. None dimension.")
                            check_coord.dimension_list.append(in_interval_dimension)
                            check_coord.dimension_list.append(interval_dimension)
                    else:
                        raise ValueError("Multi-slice along multiple dimensions is not suported yet.")
                if (add_in_int_coord):
                    self.coordinates.append(cc_new_in_int)
                if (add_int_coord):
                    self.coordinates.append(cc_new_int)
            else:
                # There are no common dimensions
                new_dimension_list = \
                        [dimension_mapping[x] for x in check_coord.dimension_list]
                for d in new_dimension_list:
                    if (d is None):
                        raise("Internal error in dimension mapping. None dimension.")
                check_coord.dimension_list = new_dimension_list
        new_coord_list = []
        for i in range(len(self.coordinates)):
            try:
                del_coords.index(i)
            except ValueError:
                new_coord_list.append(self.coordinates[i])
        self.coordinates = copy.deepcopy(new_coord_list)
        c = copy.deepcopy(Coordinate(name="Interval(" + slicing_coord_names + ")",
                                     unit="[n.a.]",
                                     mode=CoordinateMode(equidistant=True),
                                     start = 0,
                                     step = 1,
                                     dimension_list = interval_dimension))
        self.coordinates.append(c)
        c = copy.deepcopy(Coordinate(name="Interval(" + slicing_coord_names + ") sample index",
                                     unit="[n.a.]",
                                     mode=CoordinateMode(equidistant=True),
                                     start = 0,
                                     step = 1,
                                     dimension_list = in_interval_dimension))
        self.coordinates.append(c)


    def slice_data(self,slicing=None,summing=None,options=None):
        """
        Slice (select areas) from the data object along one or more coordinates.
        Return the sliced object.
          slicing:
              Dictionary with keys referring to coordinates in the data object.
                 Values can be:
                     a:SIMPLE SLICE: cases when closest value or interpolated value is selected.
                         a1 slice objects, range objects, scalars, lists, numpy array.
                         a2 flap.DataObjects without error and with data unit.name equal to
                            the coordinate
                         a3 flap.DataObject with the name of one coordinate equal to the dictionary
                            key without having value_ranges values.
                         a4 flap.Intervals objects with one interval
                     b: MULTI SLICE: Various range selection objects. In this case ranges are
                        selected and a new dimension is added to the data array
                        (only of more than 1 interval is selected) going through the
                        intervals. If intervals are of different length the longest will be used
                        and missing elements filled with float('nan').
                        Two new coordinates are added: "<coordinate> in interval",
                        "<coordinate> interval"
                        b1 flap.Intervals objects with more than one interval
                        b2 flap.DataObjects with data unit.name equal to the slicing coordinate. The error
                           values give the intervals.
                        b3 flap.DataObject with the name of one coordinate equal to the slicing coordinate.
                           The value_ranges select the intervals.
                        If range slicing is done with multiple coordinates which have common element in the
                        dimension list they will be done in one step. Otherwise the slicing is done sequentially.
          summing:
              Summing is applied to the sliced data. It processes data along one coordinate
              and the result is a scalar. This way summing reduces the number of dimensions.
              Dictionary with keys referring to coordinates and values as processing strings. If
              the processed coordinate changes along multiple dimensions those dimensions will be flattened.

              For mean and avereage data errors are calculated as error of independent variables, that is taking the square
              root of the squared sum of errors. For coordinates the mean of the ranges is taken.

              Processing strings are the following:
                       None: Nothing to be done in this dimension
                      'Mean' : take mean of values in selection/coordinate
                      'Sum'  : take sum of values in selection/coordinate
                      'Min'  : take the minimum of the values/coordinate
                      'Max'  : take the maximum of the values/coordinate

          options: 'Partial intervals'  (bool). If true processes intervals which extend over the coordinate limits. 
                                                If false only full intervals are processed.
                   'Slice type': 'Simple': Case a above: closest or interplated values are selected, dimensions 
                                           are reduced or unchanged.
                                 'Multi': Case b above: multiple intervals are selected and their date is placed into
                                          new dimension.
                                 None: Automatically select. For slicing data in case b multi slice, otherwise simple
                   'Interpolation: 'Closest value'
                                   'Linear'
        """

        default_options = {'Partial intervals': True,
                           'Slice type': None,
                           'Interpolation': 'Closest value',
                           }
        _options = flap.config.merge_options(default_options, options, data_source=self.data_source, section='Slicing')

        if (_options['Slice type'] is not None):
            try:
                _options['Slice type'] = find_str_match(_options['Slice type'], ['Simple','Multi'])
            except ValueError as e:
                raise e
        if (_options['Interpolation'] is not None):
            try:
                _options['Interpolation'] = find_str_match(_options['Interpolation'], ['Closest value','Linear'])
            except ValueError as e:
                raise e
        
                
        def check_multi_slice(slicing, coord_dim):
            """ Determines whether this slicing operation is multi-slice
                slicing: the slicing list or dictionary element
                coord_dim: Coordinate name in case of coordinate slincing,
                           dimension index in case of dimension slicing.
            """
            if ( (type(slicing) is slice) or
                 (type(slicing) is range) or
                 np.isscalar(slicing) or
                 (type(slicing) is list) or
                 (type(slicing) is np.ndarray)):
                return False
            if (type(slicing) is DataObject):
                if (slicing.data_unit.name != coord_dim):
                    try:
                        coord_obj = slicing.get_coordinate_object(coord_dim)
                    except ValueError:
                        raise ValueError("To use a flap.DataObject for slicing either the data_unit.name or a coordinate name should be identical to the coordinate.")
                    if (coord_obj.value_ranges is None):
                        return False
                    else:
                        return True
                else:
                    if (slicing.error is None):
                        return False
                    else:
                        return True
            if (type(slicing) is Intervals):
                if (slicing.number == 1):
                    return False
                else:
                    return True
            raise TypeError("Invalid slicing description.")

        def simple_slice_index(slicing, slicing_coord, data_shape, _options):
            """ Returns index array which can be used for indexing the selected elements in the
                data array flattened in the dimensions of the coordinate.

                Assumes simple (non-range) slicing.
            """
            
            if (slicing_coord.mode.equidistant):
                # There is a chance to slice with slice object if the coordinate is equidistant
                # Creating a slice object (regular_slice) if the slicing description can be described
                # with it. Later it will be checked whether it is consistent with the equidistant coordinate.
                # regular_slice is in coordinate units.
                if ((type(slicing) is slice) or (type(slicing) is range)) :
                    if (slicing.step == None):
                        regular_slice = slice(slicing.start, slicing.stop, 1)
                    else:
                        regular_slice = slice(slicing.start, slicing.stop, slicing.step)

                if ((type(slicing) is DataObject) and
                    (slicing.data_unit.name != slicing_coord.unit.name)):
                    coord_obj = slicing.get_coordinate_object(slicing_coord.unit.name)
                    if (len(coord_obj.dimension_list) == 0):
                        # This coordinate is constant then simple slice will select this element anyway
                        return 0
                    if ((coord_obj.mode.equidistant) and (len(coord_obj.dimension_list) == 1)):
                        # This coordinate changes in one dimension equidistantly
                        regular_slice = slice(coord_obj.start,
                                              coord_obj.start + coord_obj.step[0] *
                                                 (slicing.shape[coord_obj.dimension_list[0]] - 1),
                                              coord_obj.step[0])
                if ((type(slicing) is Intervals) 
                    # Regular slice is possible only with a single interval
                    and ((slicing.step is None) and (len(slicing.start) == 1) or
                        (slicing.step is not None) and (slicing.number == 1))):
                    if (slicing_coord.step[0] > 0):
                        regular_slice = slice(slicing.start[0], slicing.stop[0], slicing_coord.step[0])
                    else:
                        regular_slice = slice(slicing.stop[0], slicing.start[0], slicing_coord.step[0])
            try:
                # If regular_slice exists
                regular_slice
                regular_slice_exists = True
            except NameError:
                regular_slice_exists = False
            if (regular_slice_exists):
                # Checking if regular_slice and coordinate have common range
                slicing_coord_stop = slicing_coord.start \
                                     + slicing_coord.step[0] \
                                       * (data_shape[slicing_coord.dimension_list[0]] - 1)
                if (regular_slice.step > 0):
                    range_slice = [regular_slice.start, regular_slice.stop]
                else:
                    range_slice = [regular_slice.stop, regular_slice.start]
                range_coord = [min([slicing_coord.start, slicing_coord_stop]),
                               max([slicing_coord.start, slicing_coord_stop])]
                if ((range_slice[0] > range_coord[1]) or
                    (range_slice[1] < range_coord[0])):
                    raise ValueError("No data in slicing range.")
                # Checking whether the step of regular_slice is nearly the integer multiple of
                # the coordinate step size
                n_step = int((min([range_slice[1], range_coord[1]])
                          - max([range_slice[0], range_coord[0]]))  \
                         / abs(regular_slice.step))
                step_diff = (abs(regular_slice.step) / abs(slicing_coord.step[0]))   \
                            - round(abs(regular_slice.step) / abs(slicing_coord.step[0]))
                if (abs(step_diff) * n_step > slicing_coord.step[0] / 10):
                    del regular_slice
                    regular_slice_exists = False
                else:
                    # If interpolation is desired and the new point don't fall onto old one we cannot use slice
                    if ((_options['Interpolation'] != 'Closest value')
                        and ((abs(regular_slice.start - slicing_coord.start) / abs(slicing_coord.step[0])) % 1 > 0.01)
                        ):
                        del regular_slice
                        regular_slice_exists = False
 
            # At this point if we have a regular_slice object than it is possible to use slicing
            # in the data array. Otherwise we will create a numpy array for indexing

            if (regular_slice_exists):
                # Limiting regular slice within the data range
                if (regular_slice.step > 0):
                    if (regular_slice.start < range_coord[0]):
                        regular_slice = slice((int((range_coord[0] - regular_slice.start)  \
                                                  / regular_slice.step) + 1) * regular_slice.step   \
                                              + regular_slice.start,
                                              regular_slice.stop,
                                              regular_slice.step)

                    if (regular_slice.stop > range_coord[1]):
                        regular_slice = slice(regular_slice.start,
                                              int((range_coord[1] - regular_slice.start)  \
                                                  / regular_slice.step) * regular_slice.step   \
                                              + regular_slice.start,
                                             regular_slice.step)

                else:
                    if (regular_slice.start > range_coord[1]):
                        regular_slice = slice((int((regular_slice.start - range_coord[1])  \
                                                  / regular_slice.step) + 1) * regular_slice.step   \
                                              + regular_slice.start,
                                              regular_slice.stop,
                                              regular_slice.step)

                    if (regular_slice.stop < range_coord[0]):
                        regular_slice = slice(regular_slice.start,
                                              int((regular_slice.start - range_coord[0])  \
                                                  / regular_slice.step) * regular_slice.step   \
                                              + regular_slice.start,
                                              regular_slice.step)

                start_index = int(round((regular_slice.start - range_coord[0]) / abs(slicing_coord.step[0])))
                step_index = int(round(regular_slice.step / abs(slicing_coord.step[0])))
                stop_index = int((regular_slice.stop - range_coord[0]) / abs(slicing_coord.step[0]))
                return slice(start_index, stop_index, step_index)
            else:
                # slice object could not be created for data array
                # Creating flattened coordinate data array
                index = [0]*len(data_shape)
                for d in slicing_coord.dimension_list:
                    index[d] = ...
                coord_data, coord_low, coord_high = slicing_coord.data(data_shape, index)
                coord_data = coord_data.flatten()
                if (type(slicing) is Intervals):
                    if (not slicing_coord.isnumeric()):
                        raise TypeError("Cannot slice string coordinate with numeric intervals.")
                    # Determining the interval limits which are within the coordinate data range 
                    dr, dre = slicing_coord.data_range(data_shape=data_shape)
                    try:
                        int_l, int_h = slicing.interval_limits(limits=dr,partial_intervals=_options['Partial intervals'])
                    except ValueError as e:
                        raise e
                    # Creating a numpy array with the indices within the intervals
                    index = np.array([])
                    for i in range(len(int_l)):
                        where_start = coord_data >= int_l[i]
                        where_stop = coord_data <= int_h[i]
                        ind = np.where(np.logical_and(where_start,where_stop))[0]
                        if (len(ind) != 0):
                            index = np.concatenate((index,ind))    
                    if (len(index) == 0):
                        raise ValueError("No elements found in slicing interval(s).")
                    return index
                else:
                    # Creating numpy array from all slicing descriptions. The values are in coordinate units
                    if ((type(slicing) is list)
                        or (type(slicing) is range)):
                        slicing_arr = np.array(slicing)
                    elif (np.isscalar(slicing)):
                        slicing_arr = np.array([slicing])
                    elif (type(slicing) is  slice):
                        if (slicing.step is None):
                            slicing_arr = np.array(slicing.start, slicing.stop, 1)
                        else:
                            slicing_arr = np.array(slicing.start, slicing.stop, slicing.step)
                    elif (type(slicing) is np.ndarray):
                        slicing_arr = copy.deepcopy(slicing)
                    elif ((type(slicing) is DataObject)
                          and (slicing.data_unit.name == slicing_coord.unit.name)):
                            slicing_arr = slicing.data.flatten()
                    elif ((type(slicing) is DataObject)
                          and (slicing.data_unit.name != slicing_coord.unit.name)):
                        coord_obj = slicing.get_coordinate_object(slicing_coord.unit.name)
                        index = [0]*len(slicing.shape)
                        for d in coord_obj.dimension_list:
                            index[d] = ...    
                        slicing_arr = coord_obj.data(slicing.shape,index)[0].flatten()

                    # Creating an ind_coord array with the indices into the data array
                    # This might be a float value as interpolation might need a non-integer index
                    if (slicing_arr.dtype.kind is 'U'):
                        if (_options['Interpolation'] != 'Closest value'):
                            raise ValueError("Cannot do interpolation with string values.")
                        # For strings requiring exact match but wildcards are allowed
                        ind_coord = np.ndarray(0, np.dtype('int64'))
                        if (coord_data.size != 1):
                            coord_data_list = list(coord_data)
                        else:
                            coord_data_list = [coord_data]
                        for i in range(len(slicing_arr)):
                            selected_str, select_ind = select_signals(coord_data_list, [slicing_arr[i]])
                            ind_coord = np.concatenate((ind_coord, np.array(select_ind)))
                        if (len(ind_coord) == 0):
                            raise ValueError("No matching elements found in slicing.")
                    else:
                        if (slicing_coord.mode.equidistant and len(slicing_coord.dimension_list) == 1):
                            ind_coord = (slicing_arr - slicing_coord.start) / slicing_coord.step
                            ind = np.nonzero(ind_coord < 0)[0]
                            if (ind.size > 0):
                                ind_coord[ind] = 0
                            ind = np.nonzero(ind_coord >= data_shape[slicing_coord.dimension_list[0]])[0]
                            if (ind.size > 0):
                                ind_coord[ind] = data_shape[slicing_coord.dimension_list[0]] - 1
                        else:
                            # Checking for monotonicity of coordinate
                            diff = (coord_data[1:] - coord_data[:-1])
                            if (np.nonzero(diff[0] * diff < 0)[0].size == 0):
                                # If monotonous interpolating the coordinate-index function
                                data_index = np.arange(coord_data.shape[0],dtype=float)
                                ind_coord = np.interp(slicing_arr,coord_data,data_index)
                            else:
                                # Otherwise looking for closest match for each element
                                if (_options['Interpolation'] != 'Closest value'):
                                    raise ValueError("Coordinate is non-monotonous, cannot interpolate.")
                                ind_coord = np.ndarray(len(slicing_arr), np.dtype('int64'))
                                for i in range(len(slicing_arr)):
                                    ind_coord [i] = np.abs(coord_data - slicing_arr[i]).argmin()
                    return ind_coord

        def check_slicing_type(slicing_description, slicing_coord):
            """ Check if the slicing description is compatible with the coordinate
                Raise TypeError is not.
                Returns the slicing type 'Numeric' or 'String'
            """

            if ((type(slicing_description) is slice)
                or (type(slicing_description) is range)):
                slicing_type = 'Numeric'
            elif (np.isscalar(slicing_description)):
                if ((type(slicing_description) is str) or (type(slicing_description) is np.str_)):
                    slicing_type = 'String'
                else:
                    slicing_type = 'Numeric'
            elif (type(slicing_description) is list):
                for i in range(1,len(slicing_description)):
                    if (type(slicing_description[i]) != type(slicing_description[0])):
                        raise TypeError("Changing types in slicing list.")
                if ((type(slicing_description[0]) is str) or (type(slicing_description[0]) is np.str_)):
                    slicing_type = 'String'
                else:
                    slicing_type = 'Numeric'
            elif ((type(slicing_description) is DataObject) and
                    (slicing_description.data_unit.name == slicing_coord.unit.name)):
                slicing_type = 'Numeric'
            elif ((type(slicing_description) is DataObject) and
                    (slicing_description.data_unit.name != slicing_coord.unit.name)):
                coord_obj = slicing_description.get_coordinate_object(slicing_coord.unit.name)
                if (not coord_obj.isnumeric()):
                    slicing_type = 'String'
                else:
                    slicing_type = 'Numeric'
            elif (type(slicing_description) is Intervals):
                slicing_type = 'Numeric'
            elif (type(slicing_description) is np.ndarray):
                dtype = type(slicing_description.flatten()[0])
                if ((dtype is str) or (dtype is np.str_)):
                    slicing_type = 'String'
                else:
                    slicing_type = 'Numeric'

            if (not slicing_coord.isnumeric()):
                coord_type = 'String'
            else:
                coord_type = 'Numeric'

            if (coord_type != slicing_type):
                    raise TypeError("Incompatible slicing and coordinate values.")

            return slicing_type


        # **************** slice starts here ***********************

        if (self.data is None):
            raise ValueError("Cannot slice data object without data.")

        try:
            partial_intervals = options['Partial intervals']
        except (KeyError, TypeError):
            partial_intervals = False

        d_slice = copy.deepcopy(self)


        if (type(slicing) is dict):
            # Checking whether these coordinates are present
            slicing_coords = []
            slicing_coord_names = []
            range_slice = []
            slicing_description = []
            for sc in slicing.keys():
                try:
                    slicing_coord_names.append(sc)
                    slicing_coords.append(d_slice.get_coordinate_object(sc))
                    # Multi-slice is called range_slice in this program
                    range_slice.append(check_multi_slice(slicing[sc], sc))
                    if (_options['Slice type'] is not None):
                        if ((range_slice[-1] is False) and (_options['Slice type'] == 'Multi')):
                            raise ("Multi slice not possible with this description.")
                        if (_options['Slice type'] == 'Simple'):
                            range_slice[-1] = False
                    slicing_description.append(slicing[sc])
                except ValueError:
                    raise ValueError("Slicing coordinate "+sc+" is not present in data object.")

            # Checking whether the same coordinate appears more than once
            for i in range(len(slicing.keys())):
                for i1 in range(i+1,len(slicing.keys())):
                    if (list(slicing.keys())[i] == list(slicing.keys())[i1]):
                        raise ValueError("Cannot slice twice with the same coordinate.")

            # Checking for common dimensions in slicing coordinates where both slicing is range type
            common_dims = np.full((len(slicing), len(slicing)), False)
            for i_sc1 in range(len(slicing_coord_names)):
                for i_sc2 in range(len(slicing_coord_names)):
                    if ((i_sc1 != i_sc2) and range_slice[i_sc1] and range_slice[i_sc2]):
                        for d in slicing_coords[i_sc1].dimension_list:
                            try:
                                slicing_coords[i_sc2].dimension_list.index(d)
                                common_dims[i_sc1,i_sc2] = True
                            except ValueError:
                                pass

            # dependent multi slices (ones with common dimension) will be processed in one step,
            # therefore we need to keep track which one was processed.
            slice_processed = [False] * len(slicing_coord_names)

            # Making a copy of options. Interpolation will be set to closest value if string slice is used
            save_options = copy.deepcopy(_options)

            for i_sc in range(len(slicing_coord_names)):
                _options = copy.deepcopy(save_options)
                
                if (slice_processed[i_sc]):
                    continue
                try:
                    st = check_slicing_type(slicing_description[i_sc],slicing_coords[i_sc])
                except TypeError as e:
                    raise e
                if (st == 'String'):
                    _options['Interpolation'] = 'Closest value'
                    

                if (not range_slice[i_sc]):
                    # This is a simple slice

                    if (len(slicing_coords[i_sc].dimension_list) == 0):
                        # This coordinate is constant
                        #raise NotImplementedError("Slicing on constant coordinates is not immplemented.")
                        return d_slice
                    try:
                        # Determine slicing index in the flattened coordinate dimensions
                        ind_slice_coord = simple_slice_index(slicing_description[i_sc],
                                                             slicing_coords[i_sc],
                                                             d_slice.shape,
                                                             _options)
                    except ValueError as e:
                        raise e
                    # Flatten the data array along the coordinate dimensions
                    d_flat, dimension_mapping = flatten_multidim(d_slice.data,
                                                                 slicing_coords[i_sc].dimension_list)
                    if (d_slice.error is not None):
                        if (type(d_slice.error) is list):
                            d_err_flat_1, xx = flatten_multidim(d_slice.error[0],
                                                                 slicing_coords[i_sc].dimension_list)
                            d_err_flat_2, xx = flatten_multidim(d_slice.error[1],
                                                                 slicing_coords[i_sc].dimension_list)
                        else:
                            d_err, xx = flatten_multidim(d_slice.error,
                                                                 slicing_coords[i_sc].dimension_list)
                    # Create a list of slices for each dimension for the whole array
                    ind_slice = [slice(0,d_flat.shape[i]) for i in range(d_flat.ndim)]
                    if ((_options['Interpolation'] == 'Closest value') or (type(ind_slice_coord) is slice)):
                        # Insert the slicing indices into the flattened dimension
                        if (type(ind_slice_coord) is slice):
                            ind_slice[slicing_coords[i_sc].dimension_list[0]] = ind_slice_coord
                        else:
                            ind_slice[slicing_coords[i_sc].dimension_list[0]] = np.round(ind_slice_coord).astype(np.int32)
                        # Slice the data with this index list
                        d_slice.data = copy.deepcopy(d_flat[tuple(ind_slice)])
                        if (d_slice.error is not None):
                            if (type(d_slice.error) is list):
                                d_err_flat_1 = copy.deepcopy(d_err_flat_1[tuple(ind_slice)])
                                d_err_flat_2 = copy.deepcopy(d_err_flat_2[tuple(ind_slice)])
                            else:
                                d_err  = copy.deepcopy(d_err[tuple(ind_slice)])                        
                    elif (_options['Interpolation'] == 'Linear'):
                        # ind_slice_coord is numpy array
                        ind_slice_coord_1 = np.trunc(ind_slice_coord).astype(np.int32)
                        ind_slice_coord_2 = ind_slice_coord_1 + 1
                        # Checking if the index is above the limit. This can happen at the end
                        ind = np.nonzero(ind_slice_coord_2 >= d_flat.shape[slicing_coords[i_sc].dimension_list[0]])[0]
                        if (ind.size != 0):
                            ind_slice_coord_2[ind] = ind_slice_coord_1[ind] 
                        # Insert the lower slicing indices into the flattened dimension and get the two base points
                        # for interpolation
                        ind_slice_1 = copy.deepcopy(ind_slice)
                        ind_slice_2 = copy.deepcopy(ind_slice)
                        ind_slice_1[slicing_coords[i_sc].dimension_list[0]] = ind_slice_coord_1
                        d1 = d_flat[tuple(ind_slice_1)]
                        ind_slice_2[slicing_coords[i_sc].dimension_list[0]] = ind_slice_coord_2
                        d2 = d_flat[tuple(ind_slice_2)]
                        # Reshaping ind_slice_coord_1 and ind_slice_coord to an array with 1 elements in all
                        # dimensions except the coordinate dimendsion. This will broadcast with d2 and d2.
                        interpol_weight_shape = [1] * len(d_flat.shape)
                        interpol_weight_shape[slicing_coords[i_sc].dimension_list[0]] \
                                                                 = d1.shape[slicing_coords[i_sc].dimension_list[0]]
                        ind_slice_interp_1 = ind_slice_coord_1.reshape(tuple(interpol_weight_shape))  
                        ind_slice_interp = ind_slice_coord.reshape(tuple(interpol_weight_shape))                                        
                        d_slice.data = (d2 - d1) * (ind_slice_interp - ind_slice_interp_1) + d1
                        if (d_slice.error is not None):
                            if (type(d_slice.error) is list):
                                d_err_flat_1_1 = d_err_flat_1[tuple(ind_slice_1)]
                                d_err_flat_1_2 = d_err_flat_1[tuple(ind_slice_2)]
                                d_err_flat_1 = (d_err_flat_1_2 - d_err_flat_1_1) * (ind_slice_interp - ind_slice_interp_1) + d_err_flat_1_1
                                d_err_flat_2_1 = d_err_flat_2[tuple(ind_slice_1)]
                                d_err_flat_2_2 = d_err_flat_2[tuple(ind_slice_2)]
                                d_err_flat_2 = (d_err_flat_2_2 - d_err_flat_2_1) * (ind_slice_interp - ind_slice_interp_1) + d_err_flat_2_1
                            else:
                                d_err_1 = d_err[tuple(ind_slice_1)]
                                d_err_2 = d_err[tuple(ind_slice_2)]
                                d_err = (d_err_2 - d_err_1) * (ind_slice_interp - ind_slice_interp_1) + d_err_1
                    else:
                        raise ValueError("Internal error: cannot handle interpolation method '{:s}'".format(_options['Interpolation']))
                    original_shape = d_slice.shape
                    # If the sliced data contains only 1 element removing this dimension
                    if (d_slice.data.shape[slicing_coords[i_sc].dimension_list[0]] == 1):
                        sliced_removed = True
                        d_slice.data = np.squeeze(d_slice.data)
                    else:
                        sliced_removed = False
                        for i in range(slicing_coords[i_sc].dimension_list[0]+1,len(original_shape)):
                            if (dimension_mapping[i] is not None):
                                dimension_mapping[i] += 1

                    sliced_dimensions = slicing_coords[i_sc].dimension_list
                    d_slice.shape = d_slice.data.shape

                    if (d_slice.error is not None):
                        if (type(d_slice.error) is list):
                            if (sliced_removed):
                                d_err_flat_1 = np.squeeze(d_err_flat_1)
                                d_err_flat_2 = np.squeeze(d_err_flat_2)
                            d_slice.error = [d_err_flat_1, d_error_flat_2]
                        else:
                            d_slice.error  = np.squeeze(d_err)
                            
                    # Determining whether the slicing description results in equidistant coordinate in the 
                    # slicing coordinate. 
                    if ((type(slicing_description[i_sc]) is slice) or 
                        (type(slicing_description[i_sc]) is range)
                        ):
                       slicing_equidistant = True
                    elif ((type(slicing_description[i_sc]) is DataObject) and \
                          (slicing_description[i_sc].data_unit.name != slicing_coord_names[i_sc])
                          ):
                        coord = slicing_description[i_sc].get_coordinate_object(slicing_coord_names[i_sc])
                        if ((len(coord.dimension_list) == 1) and coord.mode.equidistant):
                            slicing_equidistant = True
                        else:
                            slicing_equidistant = False       
                    elif ((type(slicing_description[i_sc]) is Intervals) and
                          (slicing_description[i_sc].interval_number()[0] == 1) and
                          (len(slicing_coords[i_sc].dimension_list) == 1) and
                          slicing_coords[i_sc].mode.equidistant
                          ):
                        slicing_equidistant = True
                    else:
                        slicing_equidistant = False

                    # Doing changes to all coordinates
                    d_slice.__check_coords_after_simple_slice(sliced_dimensions, 
                                                              sliced_removed, 
                                                              ind_slice_coord,
                                                              original_shape, 
                                                              dimension_mapping,
                                                              slicing_equidistant,
                                                              slicing_coords[i_sc],
                                                              _options)

                    slice_processed[i_sc] = True
                else:
                    # This is multi slice
                    joint_slices = list(np.where(common_dims[i_sc,:].flatten())[0])
                    joint_slices.append(i_sc)
                    joint_dimension_list = []
                    for i in joint_slices:
                        joint_dimension_list = unify_list(joint_dimension_list,
                                                          slicing_coords[i].dimension_list)
                    original_shape = d_slice.shape
                    #flattening data for the joint_dimensions
                    d_slice.data, dimension_mapping = flatten_multidim(d_slice.data, joint_dimension_list)
                    if (d_slice.error is not None):
                        if (type(d_slice.error) is list):
                            err_flat = []
                            for i in range(2):
                                err, xx = flatten_multidim(d_slice.error[i], joint_dimension_list)
                                err_flat.append(err)
                        else:
                            err_flat, xx = flatten_multidim(d_slice.error, joint_dimension_list)
                        d_slice.error = err_flat

                    # Converting all slicing descriptions to Intervals object
                    intervals = []
                    for i in joint_slices:
                        # Getting the range of the slicing coordinate
                        r, r_ext = slicing_coords[i].data_range(data_shape=original_shape)

                        if ((type(slicing_description[i]) is DataObject) and
                            (slicing_description[i].data_unit.name == slicing_coords[i].unit.name)):
                            if (type(slicing_description[i].error) is list):
                                err1 = slicing_description[i].error[0].flatten()
                                err2 = slicing_description[i].error[1].flatten()
                            else:
                                err1 = slicing_description[i].error.flatten()
                                err2 = err1
                            data = slicing_description[i].data.flatten()
                            intervals.append(Intervals(data-err1, data+err2))
                        elif ((type(slicing_description[i]) is DataObject) and
                              (slicing_description[i].data_unit.name != slicing_coords[i].unit.name)):
                            try:
                                coord_obj = slicing_description[i].get_coordinate_object(slicing_coords[i].unit.name)
                            except Exception as e:
                                raise e
                            if (not coord_obj.isnumeric()):
                                raise ValueError("Cannot do multi-slice with string type coordinate.")
                            if (len(coord_obj.dimension_list) is not 1):
                                raise ValueError("Cannot do multi-slice with coordinate changing along multiple dimensions.")
                            if (coord_obj.mode.equidistant):
                                if (coord_obj.mode.range_symmetric):
                                    start = coord_obj.start - coord_obj.value_ranges
                                    stop = coord_obj.start + coord_obj.value_ranges
                                else:
                                    start = coord_obj.start - coord_obj.value_ranges[0]
                                    stop = coord_obj.start + coord_obj.value_ranges[1]
                                intervals.append(Intervals(start,
                                                           stop,
                                                           step=coord_obj.step[0],
                                                           number=slicing_description[i].shape[coord_obj.dimension_list[0]]))
                            else:
                                try:
                                    c, c_low, c_high = coord_obj.data(data_shape=slicing_description[i].shape,index='...')
                                except Exception as e:
                                    raise e
                                intervals.append(Intervals(c_low, c_high))
                        elif (type(slicing_description[i]) is Intervals):
                            intervals.append(slicing_description[i])

                    # This is an index to get the data for the joined dimensions
                    index = [0]*self.data.ndim
                    for i_dim in range(len(index)):
                        try:
                            joint_dimension_list.index(i_dim)
                            index[i_dim] = ...
                        except ValueError:
                            pass

                    if (len(joint_slices) == 1):
                        # If slicing is along a single coordinatge then three types of indexing are possible:
                        # slice, interval list, index list
                        slice_type_processing = slicing_coords[i_sc].mode.equidistant \
                                                and intervals[0].regular() \
                                                and (len(slicing_coords[i_sc].dimension_list) == 1)  \
                                                and not partial_intervals
                        if (slice_type_processing):
                            # Checking whether the step size of the intervals is multiple times the step size of the coordinate
                            d_range, d_range_ext = slicing_coords[i_sc].data_range(data_shape=self.shape)
                            n_int, start_int = intervals[0].interval_number(limits=d_range,
                                                                            partial_intervals=partial_intervals)
                            if (abs(
                                    round(intervals[0].step / slicing_coords[i_sc].step[0])
                                    - (intervals[0].step / slicing_coords[i_sc].step[0])
                                    ) > slicing_coords[i_sc].step[0]*1e-5) :
                                slice_type_processing = False
                        if (slice_type_processing):
                            # In slice type processing we create a slice object
                            if (n_int == 0):
                                raise ValueError("No data in intervals")
                            n_in_int = int(round(intervals[0].stop - intervals[0].start)  \
                                                    / slicing_coords[i_sc].step[0]) + 1
                            # If the number of intervals is less than the elements in the interval it does not worth
                            # doing slice processing
                            if (n_in_int < 2):
                                    raise ValueError("Number of samples in interval is too small.")
                            if (n_in_int > n_int):
                                slice_type_processing = False
                            else:
                                slice_step = int(round(intervals[0].step / slicing_coords[i_sc].step[0]))
                                slice_start = int(round((intervals[0].start + intervals[0].step * start_int
                                               - slicing_coords[i_sc].start) / slicing_coords[i_sc].step[0]))
                                slice_stop = slice_start + n_int * slice_step
                                slice_description = slice(slice_start, slice_stop, slice_step)
                                if (slice_start < 0):
                                    raise RuntimeError("Slice start below 0.")
                        # We test again if slice type is possible as it might have been changed above
                        if (not slice_type_processing):
                            if ((slicing_coords[i_sc].mode.equidistant)  \
                                  and (len(slicing_coords[i_sc].dimension_list) == 1)):
                                # In this case it is possible to determine index ranges
                                try:
                                    d_range
                                except NameError:
                                    d_range, d_range_ext = slicing_coords[i_sc].data_range(data_shape=self.shape)
                                interval_starts, interval_stops  \
                                       = intervals[i].interval_limits(limits=d_range,  \
                                                                   partial_intervals=partial_intervals)
                                if (slicing_coords[i_sc].step[0] > 0):
                                    starts = np.round((interval_starts - slicing_coords[i_sc].start)
                                                     /slicing_coords[i_sc].step[0]
                                                      ).astype('int32')
                                    stops = np.round((interval_stops - slicing_coords[i_sc].start)
                                                     /slicing_coords[i_sc].step[0]
                                                     ).astype('int32') + 1
                                else:
                                    starts = np.round((interval_stops - slicing_coords[i_sc].start)
                                                     /slicing_coords[i_sc].step[0]
                                                      ).astype('int32')
                                    stops = np.round((interval_starts - slicing_coords[i_sc].start)
                                                     /slicing_coords[i_sc].step[0]
                                                     ).astype('int32') + 1

                                slice_description = {'Starts':starts, 'Stops':stops}
                                n_int = starts.size
                                n_in_int = np.amax(stops - starts + 1)
                                if (n_in_int < 2):
                                    raise ValueError("Number of samples in interval is too small.")
                                if (np.nonzero(starts < 0)[0].size != 0):
                                    raise RuntimeError("Internal error: negative slicing indices.")
                                if (np.nonzero(stops > d_slice.data.shape[joint_dimension_list[0]])[0].size != 0):
                                    raise RuntimeError("Internal error: Slicing indices over end of data.")
                                if (np.nonzero(stops <= starts )[0].size != 0):
                                    raise RuntimeError("Internal error: Interval slice starts after stop.")
                            else:
                                # Determining indices for each interval
                                c_data, c_low, c_high = slicing_coords[i_sc].data(data_shape=d_slice.shape, index=index)
                                d_range = [np.amin(c_data), np.amax(c_data)]
                                interval_starts, interval_stops  \
                                       = intervals[i].interval_limits(limits=d_range,  \
                                                                   partial_intervals=partial_intervals)
                                c_data = np.squeeze(c_data).flatten()
                                slice_description = []
                                n_max = 0
                                for i_int in range(interval_starts.size):
                                    ind = np.nonzero(np.logical_and(c_data >= interval_starts[i_int], \
                                                     c_data <= interval_stops[i_int]))[0]
                                    slice_description.append(ind)
                                    n = np.amax(ind) - np.amin(ind) + 1
                                    if (n > n_max):
                                        n_max = n
                                n_int = interval_starts.size
                                n_in_int = n_max
                                if (n_in_int < 2):
                                    raise ValueError("Number of samples in interval is too small.")
                            # If the interval length are very much different too much space is needed, we don't
                            # do slicing
                            if (n_in_int*n_int > d_slice.data.shape[joint_dimension_list[0]] * 3):
                                raise ValueError("Interval length too much different cannot do multi-slicing.")
                            # Creating the new data matrix
                        new_shape = list(copy.deepcopy(d_slice.data.shape))
                        # Removing the flattened dimension
                        del new_shape[joint_dimension_list[0]]
                        # Moving the flattened dimension to the end
                        d_slice.__moveaxis(joint_dimension_list[0],len(d_slice.data.shape)-1)
                        if (type(slice_description) is slice):
                            # Adding two new dimensions for the intervals and the data in the intervals
                            new_shape.append(n_in_int)
                            in_interval_dimension = len(new_shape) - 1
                            new_shape.append(n_int)
                            interval_dimension = len(new_shape) - 1
                            #Creating a new data object with this new_shape
                            new_data_object = copy.deepcopy(DataObject())
                            new_data_object.__create_arrays(d_slice, tuple(new_shape))
                            # Creating an index list for the source array
                            ind = [slice(0,n) for n in d_slice.data.shape]
                            # Creating an index list for the destination array
                            ind_out = [slice(0,n) for n in new_data_object.data.shape]
                            # Going through the elements in the intervals
                            for i in range(n_in_int):
                                s = slice(slice_description.start + i,
                                          slice_description.stop + i,
                                          slice_description.step)
                                ind[-1] = s
                                ind_out[interval_dimension] = slice(0,n_int)
                                ind_out[in_interval_dimension] = i
                                new_data_object.__copy_data(d_slice,tuple(ind_out),tuple(ind))
                        else:
                            # Here we copy interval-wise
                            # Adding two new dimensions for the intervals and the data in the intervals
                            new_shape.append(n_int)
                            interval_dimension = len(new_shape) - 1
                            new_shape.append(n_in_int)
                            in_interval_dimension = len(new_shape) - 1
                            #Creating a new data object with this new_shape
                            new_data_object = copy.deepcopy(DataObject())
                            new_data_object.__create_arrays(d_slice, tuple(new_shape))
                            # Creating an index list for the source array
                            ind = [slice(0,n) for n in d_slice.data.shape]
                            # Creating an index list for the destination array
                            ind_out = [slice(0,n) for n in new_data_object.data.shape]
                            # Going through the intervals
                            for i in range(n_int):
                                if (type(slice_description) is dict):
                                    s = slice(slice_description['Starts'][i],
                                              slice_description['Stops'][i]+1)
                                    s_out = slice(0,s.stop-s.start)
                                else:
                                    # numpy array
                                    s = slice_description[i]
                                    s_out = slice(0,s.size)
                                ind[-1] = s
                                ind_out[interval_dimension] = i
                                ind_out[in_interval_dimension] = s_out
                                if (s_out.stop > 0):
                                    new_data_object.__copy_data(d_slice,tuple(ind_out),tuple(ind))
                                if (s_out.stop < n_in_int):
                                    ind_out[in_interval_dimension] = slice(s_out.stop,n_in_int)
                                    new_data_object.__fill_nan(tuple(ind_out))
                            # Move the interval dimension to the end
                            new_data_object.__moveaxis(len(new_shape)-2,len(new_shape)-1)
                            interval_dimension, in_interval_dimension = in_interval_dimension, interval_dimension

                        d_slice.data = new_data_object.data
                        d_slice.error = new_data_object.error
                        d_slice.shape = d_slice.data.shape

                        # end if len(joint_slices) == 1
                    elif (len(joint_slices) == 2):
                        raise NotImplemented("Multi-slice for multiple dependent coordinates is not implemented yet.")
                        # Getting the data for the two coordinates
                    else :
                        raise("Cannot do range slicing with more than 2 dependent coordinates.")

                    # Doing changes to all coordinates
                    d_slice.__check_coords_after_multi_slice(slicing,
                                         original_shape,
                                         joint_slices,
                                         joint_dimension_list,
                                         dimension_mapping,
                                         slice_description,
                                         interval_dimension,
                                         in_interval_dimension,
                                         n_int,
                                         n_in_int)
        elif (slicing is None):
            pass
        else:
            raise ValueError("Wrong slicing description. Use dictionary.")


        if (type(summing) is dict):
            summing_coords = []
            summing_coord_names = []
            summing_description = []
            for sc in summing.keys():
                summing_coord_names.append(sc)
                try:
                    summing_coords.append(d_slice.get_coordinate_object(sc))
                except ValueError:
                    raise ValueError("Summing coordinate "+sc+" is not present in data object.")
                summing_description.append(summing[sc])

            for i_sc in range(len(summing_coord_names)):
                original_shape = d_slice.shape

                if (len(summing_coords[i_sc].dimension_list) == 0):
                    # Do nothing if coordinate is constant
                    continue
                
                # Flattening the data array for the dimensions
                d_flat, dimension_mapping = flatten_multidim(d_slice.data,
                                                             summing_coords[i_sc].dimension_list)
                if (d_slice.error is not None):
                    if (type(d_slice.error) is list):
                        err_flat_1, dmap = flatten_multidim(d_slice.error[0],
                                                            summing_coords[i_sc].dimension_list)
                        err_flat_2, dmap = flatten_multidim(d_slice.error[1],
                                                            summing_coords[i_sc].dimension_list)

                    else:
                        d_slice.error, dmap = flatten_multidim(d_slice.error,
                                                            summing_coords[i_sc].dimension_list)

                if (summing_description[i_sc] == 'Sum'):
                    d_slice.data = np.sum(d_flat,axis=summing_coords[i_sc].dimension_list[0])
                    if (d_slice.error is not None):
                        if (type(d_slice.error) is list):
                            err = np.maximum(err_flat_1,err_flat_2)
                            d_slice.error = np.sqrt(np.sum(err**2,axis=summing_coords[i_sc].dimension_list[0]))
                        else:
                            d_slice.error = np.sqrt(np.sum(d_slice.error**2,axis=summing_coords[i_sc].dimension_list[0]))
                elif (summing_description[i_sc] == 'Mean'):
                    d_slice.data = np.mean(d_flat,axis=summing_coords[i_sc].dimension_list[0])
                    if (d_slice.error is not None):
                        if (type(d_slice.error) is list):
                            n = err_flat1.shape[summing_coords[i_sc].dimension_list[0]]
                            err = np.maximum(err_flat_1,err_flat_2)
                            d_slice.error = np.sqrt(np.sum(err**2,
                                                           axis=summing_coords[i_sc].dimension_list[0])) / n
                        else:
                            n = d_slice.error.shape[summing_coords[i_sc].dimension_list[0]]
                            d_slice.error = np.sqrt(np.sum(d_slice.error**2,
                                                           axis=summing_coords[i_sc].dimension_list[0])) / n
                elif ((summing_description[i_sc] == 'Min') or (summing_description[i_sc] == 'Max')):
                    # Finding the appropriate indices
                    if (summing_description[i_sc] == 'Min'):
                        minmax_ind = np.argmin(d_flat,axis=summing_coords[i_sc].dimension_list[0])
                    if (summing_description[i_sc] == 'Max'):
                        minmax_ind = np.argmax(d_flat,axis=summing_coords[i_sc].dimension_list[0])
                    # Adding back the dimension which was lost in np.argmin
                    minmax_ind = np.expand_dims(minmax_ind,summing_coords[i_sc].dimension_list[0])
                    # Creating an index matrix list which indexes all elements in d_flat but
                    # in the flattened dimension indexes only one element
                    index = [np.arange(x) for x in d_flat.shape]
                    index[summing_coords[i_sc].dimension_list[0]] = np.arange(1)
                    mx_ind = list(submatrix_index(d_flat.shape,index))
                    # Replacing the indexing matrix with the index matrix to the selected elements
                    mx_ind[summing_coords[i_sc].dimension_list[0]] = minmax_ind
                    d_slice.data = np.squeeze(d_flat[tuple(mx_ind)])
                    if (d_slice.error is not None):
                        if (type(d_slice.error) is list):
                            d_slice.error[0] = np.squeeze(err_flat_1[tuple(mx_ind)])
                            d_slice.error[1] = np.squeeze(err_flat_2[tuple(mx_ind)])
                        else:
                            d_slice.error = np.squeeze(d_slice.error[tuple(mx_ind)])
                else:
                    raise ValueError("Unknown summing procedure: "+summing_description[i_sc])
                d_slice.shape = d_slice.data.shape

                # Doing changes to all coordinates
                # Saving a copy of the original summing coord structure
                summing_coords_orig = copy.deepcopy(summing_coords)
                for check_coord in d_slice.coordinates:
                    # Do nothing for scalar dimension
                    if (len(check_coord.dimension_list) == 0):
                        continue

                    # Searching for common dimensions between the summing coordinate and the check_coordinate
                    common_dims = []
                    common_in_dimlist = []
                    for i_d in range(len(check_coord.dimension_list)):
                        d = check_coord.dimension_list[i_d]
                        try:
                            summing_coords_orig[i_sc].dimension_list.index(d)
                            common_dims.append(d)
                            common_in_dimlist.append(i_d)
                        except ValueError:
                            pass

                    if (len(common_dims) != 0):
                        # If there are common dimensions:
                        if ((summing_description[i_sc] == 'Min') \
                             or (summing_description[i_sc] == 'Max') \
                             or (not check_coord.mode.equidistant)):
                            # In all these cases we must get the data for the selected elements
                            # Getting coordinate data in the unified dims
                            unified_dims = unify_list(check_coord.dimension_list,
                                                          summing_coords_orig[i_sc].dimension_list)
                            ind = [0]*len(original_shape)
                            for d in unified_dims:
                                ind[d] = ...
                            try:
                                data, data_low, data_high = check_coord.data(original_shape,ind)
                            except Exception as e:
                                raise e
                            # indices of summing dimensions in unified dimension list
                            summing_in_unified = [unified_dims.index(d) for d in summing_coords_orig[i_sc].dimension_list]
                            data = np.squeeze(data)
                            data, dims = flatten_multidim(data, summing_in_unified)
                            if ((summing_description[i_sc] == 'Sum') \
                                 or (summing_description[i_sc] == 'Mean')):
                                if (not check_coord.isnumeric()):
                                    # In case of strings it is not possible to take mean of values.
                                    # Instead using the first string and adding ,...
                                    # Getting the sub-matrix at the first element in the flattened summing dimension
                                    ind = [slice(0,d) for d in data.shape]
                                    ind[common_in_dimlist[0]] = 0
                                    data = np.squeeze(data[tuple(ind)])
                                    shape = data.shape
                                    data = data.flatten()
                                    dtype = str(data.dtype)
                                    if (dtype[0:2] == '<U'):
                                        try:
                                            l = int(dtype[2:len(dtype)])
                                            dtype_new = '<U'+str(l+4)
                                            data = data.astype(dtype_new)
                                        except Exception:
                                            pass
                                    for i in range(data.size):
                                        data[i] += ',...'
                                    data = copy.deepcopy(np.reshape(data,shape))
                                    data_low = None
                                    data_high = None
                                else:
                                    data = np.squeeze(np.mean(data,axis=summing_in_unified[0]))
                                if (data_low is not None):
                                    data_low = np.squeeze(np.mean(data_low,axis=summing_in_unified[0]))
                                    data_high = np.squeeze(np.mean(data_low,axis=summing_in_unified[0]))
                            else:
                                # Min and Max
                                # Here mx_ind is an index array list for selecting elements
                                # from d_flat. Both the number of dimensions of the matrices and the number
                                # of matrices is d_flat.ndim.
                                # We could get the coordinate for all data points, flatten it as the data and
                                # select elements with these matrices, but this would not be effective.
                                # We keep from these matrices only those dimensions where the check_coordinate
                                # changes and also keep only those matrices. We will get the coordinate data only
                                # for the dimensions where check_coord changes and flattened

                                # Original dimension numbers for the flattened array, the flattened
                                # dimension is None
                                flattened_dimlist = []
                                for i_d in range(len(dimension_mapping)):
                                    if (i_d == summing_coords_orig[i_sc].dimension_list[0]):
                                        flattened_dimlist.append(None)
                                    elif (dimension_mapping[i_d] is not None):
                                        flattened_dimlist.append(i_d)

                                # Creating an index list for cutting out dimensions from the mx_ind matrices
                                ind = [0]*len(flattened_dimlist)
                                mx_ind_shape = mx_ind[0].shape
                                for i in range(len(ind)):
                                    if (flattened_dimlist[i] is None):
                                        # This is the flattened dimension, must be kept
                                        ind[i] = slice(0,mx_ind_shape[i])
                                    else:
                                        try:
                                            check_coord.dimension_list.index(flattened_dimlist[i])
                                            ind[i] = slice(0,mx_ind_shape[i])
                                        except ValueError:
                                            pass

                                # Creating the new index matrix list
                                new_mx_ind = []
                                for i in range(len(mx_ind)):
                                    if (flattened_dimlist[i] is None):
                                        new_mx_ind.append(np.squeeze(mx_ind[i][tuple(ind)]))
                                    else:
                                        try:
                                            check_coord.dimension_list.index(flattened_dimlist[i])
                                            new_mx_ind.append(np.squeeze(mx_ind[i][tuple(ind)]))
                                        except ValueError:
                                            pass

                                # Here data should have only the dimensions where the coordinate
                                # changes plus the flattened one
                                data = data[tuple(new_mx_ind)]
                                if (data_low is not None):
                                    data_low = data_low[tuple(new_mx_ind)]
                                    data_high = data_high[tuple(new_mx_ind)]
                            # Putting data into the coordinate description
                            check_coord.values = copy.deepcopy(data)
                            check_coord.mode.equidistant = False
                            if(np.isscalar(check_coord.values)):
                                check_coord.shape = []
                            else:
                                check_coord.shape = check_coord.values.shape
                            if (data_low is not None):
                                if (check_coord.mode.range_symmetric):
                                    check_coord.value_ranges = copy.deepcopy(check_coord.data - data_low)
                                else:
                                    check_coord.value_ranges = copy.deepcopy([check_coord.data - data_low,\
                                                                data_high - check_coord.data])
                        else:
                            # If not minimum or maximum and equidistant
                            # Adding the mean of the coordinates to start
                            # In Sum and Mean this is the mean alonng the dimension
                            for i_dim in common_in_dimlist:
                                check_coord.start += (original_shape[check_coord.dimension_list[i_dim]]-1) / 2 \
                                                     * check_coord.step[i_dim]
                            # Removing steps related to summed dimensions
                            check_coord.step = del_list_elements(check_coord.step, common_in_dimlist)
                        # For both equidistant and non-equidistant removing summed dimensions from list
                        orig_dimension_list = copy.deepcopy(check_coord.dimension_list)
                        for d in common_dims:
                            i = check_coord.dimension_list.index(d)
                            del check_coord.dimension_list[i]
                        if (len(check_coord.dimension_list) == 0):
                            if (check_coord.mode.equidistant):
                                check_coord.values = check_coord.start
                            check_coord.mode.equidistant = False

                    # Mapping dimensions to the new dimension list even if there is no common dimension
                    for i in range(len(check_coord.dimension_list)):
                        check_coord.dimension_list[i] = dimension_mapping[check_coord.dimension_list[i]]
                        if (check_coord.dimension_list[i] is None):
                            raise RuntimeError("Internal error in dimension mapping: None dimension")
        elif (summing is None):
            pass
        else:
            raise TypeError("Bad summing description. Use dictionary.")

        try:
            d_slice.check()
        except Exception as e:
            raise RuntimeError("Internal error. Bad data object after slicing:{:s}".format(str(e)))

        return d_slice
    # End of slice_data

    def real(self):
        if ((self.data is None) or (self.data.dtype.kind != 'c')):
            return copy.deepcopy(self)
        d_out = copy.deepcopy(self)
        d_out.data = d_out.data.real
        if ((d_out.error is not None) and (d_out.error.dtype.kind == 'c')):
            d_out.error = np.abs(d_out.error)
        if (self.data_unit is not None):
            d_out.data_unit.name = 'Real('+self.data_unit.name+')'
        return d_out

    def imag(self):
        if (self.data is None):
            return copy.deepcopy(self)
        if (self.data.dtype.kind != 'c'):
            raise ValueError("Cannot take imageinnary part of real data.")
        d_out = copy.deepcopy(self)
        d_out.data = d_out.data.imag
        if ((d_out.error is not None) and (d_out.error.dtype.kind == 'c')):
            d_out.error = np.abs(d_out.error)
        if (self.data_unit is not None):
            d_out.data_unit.name = 'Imag('+self.data_unit.name+')'
        return d_out

    def abs_value(self):
        d_out = copy.deepcopy(self)
        if (d_out.data is not None):
            d_out.data = np.abs(self.data)
        if (self.data_unit is not None):
            d_out.data_unit.name = 'Abs('+self.data_unit.name+')'
        return d_out
    
    def phase(self):
        if (self.data is not None):
            if (self.data.dtype.kind != 'c'):
                raise TypeError("Phase can be calculated only from complex data.")
        d_out = copy.deepcopy(self)
        if (d_out.data is not None):
            d_out.data = np.angle(self.data)
        if (d_out.error is not None):
            if (type(d_out.error) is list):
                error_abs = (np.abs(d_out.error[0].flatten()) + np.abs(d_out.error[1]).flatten())/2
            else:
                error_abs = np.abs(d_out.error.flatten())
            data_abs = np.abs(d_out.data.flatten())
            error = np.empty(self.data.size,dtype=float)
            ind = np.nonzero(data_abs <= error_abs)[0]
            if (ind.size > 0):
                error[ind] = math.pi
            ind = np.nonzero(data_abs > error_abs)[0]
            if (ind.size > 0):
                error[ind] = np.arctan2(error_abs[ind],data_abs[ind])
            d_out.error = error.reshape(self.data.shape)
        else:
            d_out.error = None
        if (self.data_unit is not None):
            d_out.data_unit.name = 'Phase('+self.data_unit.name+')'
        return d_out

    def error_value(self,options=None):
        """
        Returns a data object with the error of self in it.
        options: 'High': Use high error if error is asymmetric
                 'Low': Use low error is error is asymmetric
        """
        default_options = {'High': True, 'Low': False}
        try:
            _options = flap.config.merge_options(default_options, options)
        except ValueError as e:
            raise e
        if (self.error is None):
            raise ValueError("Cannot take error of data if no error is present.")
        d_out = copy.deepcopy(self)
        d_out.error = None
        if (type(self.error) is not list):
            d_out.data = copy.deepcopy(self.error)
        else:
            if (_options['High'] and _options['Low']):
                raise ValueError("Only one of High or Low can be set to True in options.")
            if ((_options['High'] or _options['Low']) is False):
                raise ValueError("One one of High or Low can be set to True in options.")
            if (_options['High']):
                d_out.data = self.error[0]
            else:
                d_out.data = self.error[1]
        return d_out

    def apsd(self, coordinate=None, intervals=None, options=None):
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
                       If not a dictionary and not None is is interpreted as the interval
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
                               ['poly', n]: Fit an n order polynomial to the data and subtract.
                            Trend removal will be applied to each interval separately.
                 'Hanning': True/False Use a Hanning window.

        Return value: The new data object

        This method is implemented by the _apsd function in spectcral_analysis.py
        """
        try:
            return _apsd(self, coordinate=coordinate, intervals=intervals, options=options)
        except Exception as e:
            raise e

    def cpsd(self, ref=None, coordinate=None, intervals=None, options=None):
        """
            Complex Cross Power Spectrum calculation for the data object self taking d_ref as reference.
            If self is not set d is used as reference, that is all spectra are calculated within self.
            Calculates all spectra between all signals in ref and self, but not inside self and ref.
            self and ref both should have the same equidistant coordinate with equal sampling points.
            Returns a data object with dimension number self.dim+ref.dim-1. The coordinate is replaced 
            by frequency or wavenumber.
            The spectrum is calculated in multiple intervals (described by intervals)
            and the mean and variance will be returned.
    
            INPUT:
                self: A flap.DataObject.
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
                           If not a dictionary and not None is is interpreted as the interval
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
                                   ['poly', n]: Fit an n order polynomial to the data and subtract.
                                Trend removal will be applied to each interval separately.
                    'Hanning': True/False Use a Hanning window.
                    'Normalize': Calculate coherency instead of crosspower
        """
        try:
            return _cpsd(self, ref=ref, coordinate=coordinate, intervals=intervals, options=options)
        except Exception as e:
            raise e

    def ccf(self, ref=None, coordinate=None, intervals=None, options=None):
        """
            N dimensional Cross Correlation Function or covariance calculation for the data object self taking d_ref 
            as reference. If ref is not set self is used as reference, that is all CCFs are calculated 
            within self. Calculates all CCF between all signals in ref and sel, but not inside self and ref.
            Correlation is calculated along the coordinate(s) listed in coordinate which should be
            identical for the to input data objects.
            Returns a data object with dimension number self.dim+ref.dim-len(coordinate). 
            The coordinates are replaced by coordinate+' lag'.
            The CCF is calculated in multiple intervals (described by intervals)
            and the mean and variance will be returned.
    
            INPUT:
                self: A flap.DataObject.
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
                           If not a dictionary and not None is is interpreted as the interval
                               description, the selection coordinate is taken the same as
                               coordinate.
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
                                   ['poly', n]: Fit an n order polynomial to the data and subtract.
                                Trend removal will be applied to each interval separately.
                                At present trend removal can be applied to 1D CCF only.
                    'Hanning': True/False Use a Hanning window.
                    'Normalize': Normalize with autocorrelations, that is calculate correlation instead of covariance.
        """
        try:
            return _ccf(self, ref=ref, coordinate=coordinate, intervals=intervals, options=options)
        except Exception as e:
            raise e


    def detrend(self, coordinate=None, intervals=None, options=None):
        """ Trend removal.
            INPUT:
                coordinate: The x coordinate for the trend removal.
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
                options:
                  'Trend removal': Trend removal description (see also _trend_removal()). A list, string or None.
                    None: Don't remove trend.
                    Strings:
                      'Mean': subtract mean
                    Lists:
                      ['Poly', n]: Fit an n order polynomial to the data and subtract.
                            Trend removal will be applied to each interval defined by slicing
                            separately.
            Return value: The data object with the trend removed data.
       """

        if (self.data is None):
            raise ValueError("Cannot detrend without data.")
        default_options = {'Trend removal': ['poly', 2]}

        try:
            _options = flap.config.merge_options(default_options, options,
                                             data_source=self.data_source,
                                             section='Trend removal')
        except ValueError as e:
            raise e
        if (coordinate is None):
            c_names = self.coordinate_names()
            try:
                c_names.index('Time')
                _coordinate = 'Time'
            except ValueError:
                raise ValueError("No coordinate is given for detrend and no Time coordinate found.")
        else:
            _coordinate = coordinate
        try:
            coord_obj = self.get_coordinate_object(_coordinate)
        except Exception as e:
            raise e
        if (len(coord_obj.dimension_list) != 1):
            raise ValueError("Trend removal is possible only along coordinates changing along one dimension.")
        trend = _options['Trend removal']

        try:    
            calc_int, calc_int_ind, sel_int, sel_int_ind = self.proc_interval_limits(_coordinate, intervals=intervals)
        except Exception as e:
            raise e
        int_start_ind = sel_int_ind[0]
        int_end_ind = sel_int_ind[1]
        int_start = sel_int[0]
        int_end = sel_int[1]

        if (type(intervals) is dict):
            sel_coordinate = list(intervals.keys())[0]
            sel_coord_obj = self.get_coordinate_object(_coordinate)
        else:
            sel_coordinate = _coordinate
            sel_coord_obj = coord_obj


        d = copy.deepcopy(self)

        for i_int in range(len(int_start)):
            ind = [slice(0,dim) for dim in d.shape]
            ind[sel_coord_obj.dimension_list[0]] = slice(int_start_ind[i_int],int_end_ind[i_int])
            if (coord_obj.mode.equidistant):
                _trend_removal(d.data[tuple(ind)],
                               coord_obj.dimension_list[0],
                               trend)
            else:
                try:
                    index = [0]*len(d.shape)
                    index[sel_coord_obj.dimension_list[0]] = slice(int_start_ind[i_int],int_end_ind[i_int])
                    coord_data = coord_obj.data(data_shape=d.shape, index=index)[0]
                except Exception as e:
                    raise e
                _trend_removal(d.data[tuple(ind)],
                               coord_obj.dimension_list[0],
                               trend,
                               x = coord_data)
        return d


    def filter_data(self, coordinate=None, intervals=None, options=None):
        """ 1D Data filter.
            INPUT:
                coordinate: The x coordinate for the trend removal.
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
                options:
                    'Type' :
                        None: Do nothing.
                        'Int': Single term IIF filter, like RC integrator. 
                        'Diff': Single term IIF filter, like RC differentiator.
                        'Bandpass', 'Lowpass', 'Highpass': Filters designed by scipy.signal.iirdesign.
                                                           The filter type is in 'Design'
                                   Bandpass: f_low - f_high
                                   Lowpass: - f_high
                                   Highpass: - f_low
                    'Design': The design type of the bandpass, lowpass or highpass filter. 
                              ('Elliptic', 'Butterworth, 'Chebyshev I', 'Chebyshev II', 'Bessel') 
                              The numpy.iirdesign function is used for generating the filter.  
                              Setting inconsistent parameters can cause strange results. E.g. too high attenuation
                              at too low frequency relative to the smapling frequency can be a problem.
                    'f_low', 'f_high': Cut on/off frequencies. (Middle between passband and stopband edge.)
                    'Steepness': Difference between passband and stopband edge frequencies as a fraction 
                                 of the middle frequency.
                     'Loss': The maximum loss in the passband in dB
                     'Attenuation': The minimum attenuation in the stopband dB
                    'Tau': time constant for integrator/differentiator (in units of the coordinate)
                    'Power': Calculate square of the signal after filtering. (boolean)
                    'Inttime': Integration time after power calculation. (in units of oordinate)
            Return value: The data object with the filtered data.
       """

        if (self.data is None):
            raise ValueError("Cannot filter without data.")
        default_options = {'Type':None, 'Tau': None, 'Design':'Elliptic', 'f_low': None, 'f_high': None,
                           'Steepness': 0.2, 'Power': False, 'Inttime': None, 'Loss': 3, 'Attenuation':20}

        try:
            _options = flap.config.merge_options(default_options, options,
                                             data_source=self.data_source,
                                             section='Filter data')
        except ValueError as e:
            raise e
        if (coordinate is None):
            c_names = self.coordinate_names()
            try:
                c_names.index('Time')
                _coordinate = 'Time'
            except ValueError:
                raise ValueError("No coordinate is given for detrend and no Time coordinate found.")
        else:
            _coordinate = coordinate
        try:
            coord_obj = self.get_coordinate_object(_coordinate)
        except Exception as e:
            raise e
        if (len(coord_obj.dimension_list) != 1):
            raise ValueError("Filtering is possible only along coordinates changing along one dimension.")
        if (not coord_obj.mode.equidistant):
            raise ValueError("Filtering is possible only along equidistant coordinates.")
        filter_type = _options['Type']
        if ((filter_type == 'Int') or (filter_type == 'Diff')):
            tau = _options['Tau']
            if (tau is None):
                raise ValueError("Missing time constant (Tau) option for filtering.")
        elif (filter_type == 'Bandpass'):
            f_low = _options['f_low']
            if (f_low is None):
                raise ValueError("Missing f_low parameter for bandpass filter.")
            f_high = _options['f_high']
            if (f_high is None):
                raise ValueError("Missing f_high parameter for bandpass filter.")
        elif (filter_type == 'Lowpass'):
            f_high = _options['f_high']
            if (f_high is None):
                raise ValueError("Missing f_high parameter for lowpass filter.")
        elif (filter_type == 'Highpass'):
            f_low = _options['f_low']
            if (f_low is None):
                raise ValueError("Missing f_low parameter for highpass filter.")
        else:
            raise ValueError("Invalid filter type.")
            
        if (_options['Design'] == 'Elliptic'):
            design = 'ellip'
        elif (_options['Design'] == 'Butterworth'):
            design = 'butter'
        elif (_options['Design'] == 'Chebyshev I'):
            design = 'cheby1'
        elif (_options['Design'] == 'Chebyshev II'):
            design = 'cheby2'
        elif (_options['Design'] == 'Bessel'):
            design = 'bessel'
        else:
            raise ValueError("Invalid filter design.")
    
        loss = _options['Loss']
        attenuation = _options['Attenuation']
        
        try:    
            calc_int, calc_int_ind, sel_int, sel_int_ind = self.proc_interval_limits(_coordinate, intervals=intervals)
        except Exception as e:
            raise e
        int_start_ind = sel_int_ind[0]
        int_end_ind = sel_int_ind[1]
        int_start = sel_int[0]
        int_end = sel_int[1]

        if (type(intervals) is dict):
            sel_coordinate = list(intervals.keys())[0]
            sel_coord_obj = self.get_coordinate_object(sel_coordinate)
        else:
            sel_coordinate = _coordinate
            sel_coord_obj = coord_obj


        d = copy.deepcopy(self)

        for i_int in range(len(int_start)):
            ind = [slice(0,dim) for dim in d.shape]
            ind[sel_coord_obj.dimension_list[0]] = slice(int_start_ind[i_int],int_end_ind[i_int])
            ind = tuple(ind)
            if (filter_type == 'Int'):
                tau_norm = tau / float(coord_obj.step[0])
                a = np.array([1,-math.exp(-1/tau_norm)]) / (1-math.exp(-1/tau_norm))
                b = np.array([1])
                ind_0 = list(ind)
                ind_0[coord_obj.dimension_list[0]] = int_start_ind[i_int]
                zi = d.data[tuple(ind_0)]
                zi = np.expand_dims(zi,coord_obj.dimension_list[0])
                d.data[ind],zf = signal.lfilter(b, 
                                                a, 
                                                d.data[ind], 
                                                axis=coord_obj.dimension_list[0],
                                                zi=zi
                                                )   

            elif (filter_type == 'Diff'):   
                tau_norm = tau / float(coord_obj.step[0])
                a = np.array([1,-math.exp(-1/tau_norm)]) / (1-math.exp(-1/tau_norm))
                b = np.array([1])
                ind_0 = list(ind)
                ind_0[coord_obj.dimension_list[0]] = int_start_ind[i_int]
                zi = d.data[tuple(ind_0)]
                zi = np.expand_dims(zi,coord_obj.dimension_list[0])
                zi = zi
                dd,zf = signal.lfilter(b, 
                                       a, 
                                       d.data[ind], 
                                       axis=coord_obj.dimension_list[0],
                                       zi=zi)   
                d.data[ind] =  d.data[ind] - dd              
            elif (filter_type == 'Lowpass'):
                try:
                    steep = float(_options['Steepness'])
                except ValueError:
                    raise ValueError("Invalid steepness option.")
                fn = 1./coord_obj.step[0] / 2
                wp = (f_high - f_high * steep / 2) / fn
                ws = (f_high + f_high * steep / 2) / fn
                gpass = 3
                gstop = 40
                b, a = signal.iirdesign(wp,ws,loss,attenuation,ftype=design,output='ba')
                start_data = np.take(d.data, [int_start_ind[i_int]], axis=coord_obj.dimension_list[0])
                start_data = np.squeeze(start_data)
                start_data = np.expand_dims(start_data,coord_obj.dimension_list[0])
                zi = signal.lfilter_zi(b, a)
                zi_shape = [1]*start_data.ndim
                zi_shape[coord_obj.dimension_list[0]] = len(zi)
                zi = np.reshape(zi,zi_shape)
                zi = zi * start_data
                d.data[ind],zf = signal.lfilter(b,a,d.data[ind],axis=coord_obj.dimension_list[0],zi=zi)   
            elif (filter_type == 'Highpass'):
                try:
                    steep = float(_options['Steepness'])
                except ValueError:
                    raise ValueError("Invalid steepness option.")
                fn = 1./coord_obj.step[0] / 2
                wp = (f_low + f_low * steep / 2) / fn 
                ws = (f_low - f_low * steep / 2) / fn
                gpass = 3
                gstop = 40
                b, a = signal.iirdesign(wp,ws,loss,attenuation,ftype=design,output='ba')
                # preparing start of filtered signal, so as it starts the the steady part of the transfer function. 
                start_data = np.take(d.data, [int_start_ind[i_int]], axis=coord_obj.dimension_list[0])
                start_data = np.squeeze(start_data)
                start_data = np.expand_dims(start_data,coord_obj.dimension_list[0])
                zi = signal.lfilter_zi(b, a)
                zi_shape = [1]*start_data.ndim
                zi_shape[coord_obj.dimension_list[0]] = len(zi)
                zi = np.reshape(zi,zi_shape)
                zi = zi * start_data
                d.data[ind],zf = signal.lfilter(b,a,d.data[ind],axis=coord_obj.dimension_list[0],zi=zi)   
            elif (filter_type == 'Bandpass'):
                try:
                    steep = float(_options['Steepness'])
                except ValueError:
                    raise ValueError("Invalid steepness option.")
                fn = 1./coord_obj.step[0] / 2
                wp = [(f_low + f_low * steep / 2) / fn, (f_high - f_high * steep / 2) / fn]
                ws = [(f_low - f_low * steep / 2) / fn, (f_high + f_high * steep / 2) / fn]
                gpass = 3
                gstop = 40
                b, a = signal.iirdesign(wp,ws,loss,attenuation,ftype=design,output='ba')
                start_data = np.take(d.data, [int_start_ind[i_int]], axis=coord_obj.dimension_list[0])
                start_data = np.squeeze(start_data)
                start_data = np.expand_dims(start_data,coord_obj.dimension_list[0])
                zi = signal.lfilter_zi(b, a)
                zi_shape = [1]*start_data.ndim
                zi_shape[coord_obj.dimension_list[0]] = len(zi)
                zi = np.reshape(zi,zi_shape)
                zi = zi * start_data
                d.data[ind],zf = signal.lfilter(b,a,d.data[ind],axis=coord_obj.dimension_list[0],zi=zi)   
            if (_options['Power']):
                d.data[ind] = d.data[ind] ** 2
                if (_options['Inttime'] is not None):
                    try:
                        inttime = float(_options['Inttime'])
                    except ValueError:
                        raise ValueError("Invalid Inttime value.")
                tau_norm = inttime / float(coord_obj.step[0])
                a = np.array([1,-math.exp(-1/tau_norm)]) / (1-math.exp(-1/tau_norm))
                b = np.array([1])
                ind_0 = list(ind)
                ind_0[sel_coord_obj.dimension_list[0]] = int_start_ind[i_int]
                zi = d.data[tuple(ind_0)]
                zi = np.expand_dims(zi,coord_obj.dimension_list[0])
                d.data[ind],zo = signal.lfilter(b, 
                                                a, 
                                                d.data[ind], 
                                                axis=coord_obj.dimension_list[0],
                                                zi=zi
                                                )   
        return d

    def save(self,filename):
        """
        Save the data object to a binary file using pickle.
        Use load to read the object.
        """
        try:
            f = io.open(filename,"wb")
        except OSError:
            raise ValueError("Cannot open file "+filename)
        try:
            d = self
            pickle.dump(d,f)
        except pickle.PicklingError:
            raise TypeError("Error encoding data object.")
        try:
            f.close()
        except Exception:
            raise e

    def __add__(self, d1):
        """ Addition for DataObject
        """
        if (type(d1) is DataObject):
            if ((self.data is None) or (d1.data is None)):
                raise ValueError("Cannot add data objects without data.")
            if (self.data_unit.unit != d1.data_unit.unit):
                raise ValueError("Cannot add data objects with different units.")
            if (self.data_unit.name != d1.data_unit.name):
                raise ValueError("Cannot add data objects with different data names.")
            if (self.shape != d1.shape):
                raise ValueError("Cannot add data objects with different shapes.")
            if (self.error is None):
                err = d1.error
            elif (d1.error is None):
                err = self.error
            else:
                if ((type(self.error) is not list) and (type(d1.error) is not list)):
                    err = np.sqrt(self.error ** 2 + d1.error ** 2)
                elif ((type(self.error) is list) and (type(d1.error) is list)):
                    err = [np.sqrt(self.error[0] ** 2 + d1.error[0] ** 2),
                           np.sqrt(self.error[1] ** 2 + d1.error[1] ** 2)]                    
                elif (type(self.error) is list):
                    err = [np.sqrt(self.error[0] ** 2 + d1.error ** 2),
                           np.sqrt(self.error[1] ** 2 + d1.error ** 2)]
                else:
                    err = [np.sqrt(self.error ** 2 + d1.error[0] ** 2),
                           np.sqrt(self.error ** 2 + d1.error[1] ** 2)]
                                                   
            d_out = DataObject(data_array=self.data+d1.data,
                               error=err,
                               data_unit=self.data_unit)
            d_out.coordinates = []
            for coord in self.coordinates:
                try:
                    c1 = d1.get_coordinate_object(coord.unit.name)
                    if (coord != c1):
                        continue
                    d_out.coordinates.append(copy.deepcopy(coord))
                except ValueError:
                    pass
            if (len(d_out.coordinates) == 0):
                d_out.coordinates = None
            if (self.exp_id == d1.exp_id):
                d_out.exp_id = self.exp_id
            if (self.data_source == d1.data_source):
                d_out.data_source = self.data_source
        elif (np.isscalar(d1)):
            if (self.data is None):
                raise ValueError("Cannot add data objects without data.")
            if ((self.data.dtype.kind == 's') or (type(d1) is str)):
                raise ValueError("Cannot add strings.")
            d_out = copy.deepcopy(self)
            d_out.data += d1
        else:
            raise TypeError("Invalid data type for addition with flap.DataObject")

        return d_out

    def __radd__(self, d1):
         try:
             return self.__add__(d1)
         except Exception as e:
             raise e

    def __sub__(self, d1):
        """ Subtraction for DataObject
        """
        if (type(d1) is DataObject):
            if ((self.data is None) or (d1.data is None)):
                raise ValueError("Cannot subtract data objects without data.")
            if (self.data_unit.unit != d1.data_unit.unit):
                raise ValueError("Cannot subtract data objects with different units.")
            if (self.data_unit.name != d1.data_unit.name):
                raise ValueError("Cannot subtract data objects with different data names.")
            if (self.shape != d1.shape):
                raise ValueError("Cannot subtract data objects with different shapes.")
            if (self.error is None):
                err = d1.error
            elif (d1.error is None):
                err = self.error
            else:
                if ((type(self.error) is not list) and (type(d1.error) is not list)):
                    err = np.sqrt(self.error ** 2 + d1.error ** 2)
                elif ((type(self.error) is list) and (type(d1.error) is list)):
                    err = [np.sqrt(self.error[0] ** 2 + d1.error[0] ** 2),
                           np.sqrt(self.error[1] ** 2 + d1.error[1] ** 2)]                    
                elif (type(self.error) is list):
                    err = [np.sqrt(self.error[0] ** 2 + d1.error ** 2),
                           np.sqrt(self.error[1] ** 2 + d1.error ** 2)]
                else:
                    err = [np.sqrt(self.error ** 2 + d1.error[0] ** 2),
                           np.sqrt(self.error ** 2 + d1.error[1] ** 2)]
                                                   
            d_out = DataObject(data_array=self.data-d1.data,
                               error=err,
                               data_unit=self.data_unit)
            d_out.coordinates = []
            for coord in self.coordinates:
                try:
                    c1 = d1.get_coordinate_object(coord.unit.name)
                    if (coord != c1):
                        continue
                    d_out.coordinates.append(copy.deepcopy(coord))
                except ValueError:
                    pass
            if (len(d_out.coordinates) == 0):
                d_out.coordinates = None
            if (self.exp_id == d1.exp_id):
                d_out.exp_id = self.exp_id
            if (self.data_source == d1.data_source):
                d_out.data_source = self.data_source
        elif (np.isscalar(d1)):
            if (self.data is None):
                raise ValueError("Cannot add data objects without data.")
            if ((self.data.dtype.kind == 's') or (type(d1) is str)):
                raise ValueError("Cannot add strings.")
            d_out = copy.deepcopy(self)
            d_out.data -= d1
        else:
            raise TypeError("Invalid data type for subtaction from flap.DataObject")

        return d_out

    def __rsub__(self, d1):
         try:
             d = self.__sub__(d1)
             d.data = -d.data
             return d
         except Exception as e:
             raise e

    def __mul__(self, d1):
        """ Multiplication for DataObject
        """
        if (type(d1) is DataObject):
            if ((self.data is None) or (d1.data is None)):
                raise ValueError("Cannot multiply data objects without data.")
            if (self.shape != d1.shape):
                raise ValueError("Cannot multiply data objects with different shapes.")
            if ((self.error is None) or (d1.error is None)):
                err = None
            else:
                if (type(self.error) is list):
                    err1 = sqrt(self.error[0] ** 2 + self.error[1] ** 2)
                else:
                    err1 = self.error
                if (type(d1.error) is list):
                    err2 = sqrt(d1.error[0] ** 2 + d1.error[1] ** 2)
                else:
                    err2 = d1.error
                err = np.sqrt(err1 ** 2 * err2 ** 2 
                              + self.data ** 2 * err2 ** 2
                              + d1.data **2 * err1 ** 2)
            new_unit = Unit()
            if (d1.data_unit.name == self.data_unit.name):
                new_unit.name = self.data_unit.name+'^2'
            else:
                new_unit.name = self.data_unit.name+'*'+d1.unit.name
            if (d1.data_unit.unit == self.data_unit.unit):
                new_unit.unit = self.data_unit.unit+'^2'
            else:
                new_unit.unit = self.data_unit.unit+'*'+d1.unit.name
                
            d_out = DataObject(data_array=self.data * d1.data,
                               error=err,
                               data_unit=new_unit
                               )
            d_out.coordinates = []
            for coord in self.coordinates:
                try:
                    c1 = d1.get_coordinate_object(coord.unit.name)
                    if (coord != c1):
                        continue
                    d_out.coordinates.append(copy.deepcopy(coord))
                except ValueError:
                    pass
            if (len(d_out.coordinates) == 0):
                d_out.coordinates = None
            if (self.exp_id == d1.exp_id):
                d_out.exp_id = self.exp_id
            if (self.data_source == d1.data_source):
                d_out.data_source = self.data_source
        elif (np.isscalar(d1)):
            if (self.data is None):
                raise ValueError("Cannot multiply data objects without data.")
            if ((self.data.dtype.kind == 's') or (type(d1) is str)):
                raise ValueError("Cannot multiply strings.")
            d_out = copy.deepcopy(self)
            d_out.data *= d1
            if (d_out.error is not None):
                if (type(d_out.error) is list):
                    d_out.error = [d_out.error[0] * d1, d_out.error[1] * d1]
                else:
                    d_out.error *= d1
        else:
            raise TypeError("Invalid data type for multiplication with flap.DataObject")
        return d_out
             
    def __rmul__(self, d1):
         try:
             return self.__mul__(d1)
         except Exception as e:
             raise e
             
########## END of class DataObject            
            
class FlapStorage:
    """ This class is for data and data source information storage
    """
    def __init__(self):
        # List of data sources
        self.__data_sources = []
        # list of get_data function objects for each data source.
        self.__get_data_functions = []
        # list of functions for adding coordinates to a data object
        self.__add_coordinate_functions = []
        # The data objects. This is a dictionary, the keys are the data object names
        self.__data_objects = {}

    """ The methods below are interfaces for the private variables of this class
    """
    def data_sources(self):
        return self.__data_sources

    def add_data_source(self,data_source,get_data_func=None, add_coord_func=None):
        self.__data_sources.append(data_source)
        self.__get_data_functions.append(get_data_func)
        self.__add_coordinate_functions.append(add_coord_func)

    def get_data_functions(self):
        return self.__get_data_functions

    def add_coord_functions(self):
        return self.__add_coordinate_functions

    def data_objects(self):
        return self.__data_objects

    def add_data_object(self,d,name):
        _name = name + '_exp_id:' + str(d.exp_id)
        self.__data_objects[_name] = d

    def delete_data_object(self,name,exp_id=None):
        _name = name + '_exp_id:' + str(exp_id)
        del_list = []
        for n in self.__data_objects.keys():
            if (fnmatch.fnmatch(n,_name)):
                del_list.append(n)
        for n in del_list:
            del self.__data_objects[n]

    def find_data_objects(self,name,exp_id='*'):
        """" 
        Find data objects in flap storage.
        
        Returns name list, exp_ID list
        """
        _name = name + '_exp_id:' + str(exp_id)
        nlist = []
        explist = []
        for n in self.__data_objects.keys():
            if (fnmatch.fnmatch(n,_name)):
                n_split = n.split('_exp_id:')
                if (len(n_split) == 2):
                    nlist.append(n_split[0])
                    explist.append(n_split[1])
        return nlist, explist
                

    def get_data_object(self,name,exp_id='*'):
        _name = name + '_exp_id:' + str(exp_id)
        try:
            d = self.__data_objects[_name]
            return d
        except KeyError:
            nlist = []
            for n in self.__data_objects.keys():
                if (fnmatch.fnmatch(n,_name)):
                    nlist.append(n)
            if len(nlist) == 0:
                raise KeyError("Data object " + name
                               + "(exp_id:" + str(exp_id) + ") does not exists.")
            if (len(nlist) > 1):
                raise KeyError("Multiple data objects found for name "
                               + name + "(exp_id:" + str(exp_id) + ").")
            d = copy.deepcopy(self.__data_objects[nlist[0]])
            return d

    def list_data_objects(self,name, exp_id='*'):
        if (name is None):
            _name = '*'
        else:
            _name = name
        names, exps = find_data_objects(_name, exp_id=exp_id)
        if (len(names) == 0):
            return [], [], []
        
        d_list = []
        for i_names in range(len(names)):
            d_list.append(get_data_object(names[i_names], exp_id=exps[i_names]))
        return d_list, names, exps


# This is the instance of the storage class
# it will be created only if it does not exists.
global flap_storage
try:
    flap_storage
except NameError:
    try:
#        if (VERBOSE):
        print("INIT flap storage")
    except NameError:
        pass
    flap_storage = FlapStorage()


# The functions below are the flap interface
def list_data_sources():
    """ Return a list of registered data source names as a list.
    """
    global flap_storage
    return flap_storage.data_sources()


def get_data_function(data_source):
    """ Return the get data function object for a given data source
    """
    global flap_storage
    try:
        ds = flap_storage.data_sources()
        i = ds.index(data_source)
    except ValueError:
        raise ValueError("Data source "+data_source+" is unknown.")
    df = flap_storage.get_data_functions()
    return df[i]


def get_addcoord_function(data_source):
    """ Return the add_coord function object for a given data source
    """
    global flap_storage
    try:
        ds = flap_storage.data_sources()
        i = ds.index(data_source)
    except ValueError:
        raise ValueError("Data source " + data_source + " is unknown.")
    df = flap_storage.add_coord_functions()
    return df[i]

def register_data_source(name, get_data_func=None, add_coord_func=None):
    """
    Register a new data source name and the associated functions.
    """
    global flap_storage
    try:
        flap_storage.data_sources().index(name)
        return
    except ValueError:
        flap_storage.add_data_source(name,
                                     get_data_func=get_data_func,
                                     add_coord_func=add_coord_func)
        return ''


def add_data_object(d,object_name):
    """ Add a data object to the flap storage
    """
    global flap_storage
    flap_storage.add_data_object(d,object_name)


def delete_data_object(object_name,exp_id='*'):
    """ Delete a data objejct from the flap storage
    """
    global flap_storage
    try:
        flap_storage.delete_data_object(object_name, exp_id=exp_id)
    except KeyError as e:
        raise e

def find_data_objects(name,exp_id='*'):
    """
    Find data objects in flap storage.
    
    Returns list of names, list of expID.s
    """
    global flap_storage
    try:
        names, exps = flap_storage.find_data_objects(name,exp_id=exp_id)
    except Exception as e:
        raise e
    return names, exps


def get_data_object(object_name,exp_id='*'):
    """  Return a data object from the flap storage
    """
    global flap_storage
    try:
        d = flap_storage.get_data_object(object_name,exp_id=exp_id)
    except KeyError as e:
        raise e
    return d


def list_data_objects(name='*', exp_id='*', screen=True):
    """ Prepare a printout of data objects is flap storage or the listed data objects.
        name: name (with wildcards) or list of data objects
        exp_id: exp id for name
        screen: (bool) If True print to screen
        
        Return value: The text
    """
    if ((type(name) is list) or (type(name) is DataObject)):
        if (type(name) is not list):
            d_list = [name]
        else:    
            d_list = name
        names = []
        for i,d in enumerate(d_list):
            if (type(d) is not DataObject):
                raise TypeError("List of flap.DataObjects is expected.")
            names.append("<{:d}>".format(i+1))
    else:
        d_list, names, exps = flap_storage.list_data_objects(name=name, exp_id=exp_id)

    s = ''
    for i_names,d in enumerate(d_list):
        s += '\n-----------------------------\n'
        if (d.data_title is None):
            dtit = ""
        else:
            dtit = d.data_title
        if (d.exp_id is None):
            expid = ""
        else:
            expid= d.exp_id
        if (d.data_source is None):
            ds = ""
        else:
            ds= d.data_source

        s += names[i_names] + '(data_source:"'+ds+'" exp_id:"' + expid + '") data_title:"' + dtit + '"'
        s += " shape:["
        if (d.shape is None):
            data_shape = []
        else:
            data_shape = d.shape
        for i in range(len(data_shape)):
            s += str(data_shape[i])
            if (i != len(data_shape) - 1):
                s += ","
        s += "]"
        if (d.data is None):
            s += " [no data]"
        if (d.error is None):
            s += "[no error]"
        elif (type(d.error) is list):
            s += "[error asymm.]"
        else:
            s += "[error symm.]"
        if (d.data_unit is not None):
            if (d.data_unit.name is not None):
                data_name = d.data_unit.name
            else:
                data_name = ""
            if (d.data_unit.unit is not None):
                data_unit = d.data_unit.unit
            else:
                data_unit = ""
        s += '\n  Data name:"'+data_name+'", unit:"'+data_unit+'"'
        s += "\n  Coords:\n"
        c_names = d.coordinate_names()
        for i in range(len(c_names)):
            c = d.coordinates[i]
            s += "    '" + c.unit.title()
            s += "(Dims:"
            if (len(c.dimension_list) is not 0):
                try:
                    c.dimension_list.index(None)
                    print("None in dim. list!")
                except:
                    for i1 in range(len(c.dimension_list)):
                        s += str(c.dimension_list[i1])
                        if (i1 != len(c.dimension_list) - 1):
                            s += ","
            if (not c.mode.equidistant):
                s = s + ", Shape:["
                if (type(c.shape) is not list):
                    shape = [c.shape]
                else:
                    shape = c.shape
                if (len(shape) is not 0):
                    for i1 in range(len(shape)):
                        s += str(shape[i1])
                        if (i1 != len(shape) - 1):
                            s += ","
            s += "]) ["
            if (c.mode.equidistant):
                s += '<Equ.>'
            if (c.mode.range_symmetric):
                s += '<R. symm.>'
            s += '] '
            if (c.value_ranges is not None):
                s +='[Ranges set] '
            if (c.mode.equidistant):
                s += 'Start: {:10.3E}, Steps: '.format(c.start)
                for i_step in range(len(c.step)):
                   s += '{:10.3E}'.format(c.step[i_step])
                   if (i_step < len(c.step)-1):
                       s +=", "
            else:
                if (len(c.dimension_list)) is not 0:
                    ind = [0] * len(data_shape)
                    for dim in c.dimension_list:
                        ind[dim] = ...
                else:
                    ind = [0]*len(data_shape)
                c_data, c_range_l, c_range_h = c.data(data_shape=data_shape,index=ind)
                dt = c.dtype()
                try:
                    no_max = False
                    "{:10.3E}".format(np.min(c_data))
                except (TypeError, ValueError):
                    no_max = True
                if ((c_data.size <= 10) or no_max):
                    s += 'Val:'
                    n = c_data.size
                    if (n > 10):
                        n = 10
                    for j in range(n):
                        if (dt is str):
                            if (np.isscalar(c_data)):
                                s += c_data
                            else:
                                s += c_data.flatten()[j]
                        elif ((dt is Decimal) or (dt is float)):
                            s += "{:10.3E}".format(c_data.flatten()[j])
                        else:
                            s += str(c_data.flatten()[j])
                        if (j != c_data.size - 1):
                            s += ", "
                    if (c_data.size > 10):
                        s += "..."
                else:
                    s += 'Val. range: ' + "{:10.3E}".format(np.min(c_data)) + ' - '\
                         + "{:10.3E}".format(np.max(c_data))
            if (i != len(d.coordinates) - 1):
                s += "\n"
    if (screen):
        print(s)
    return s



def get_data(data_source,
             exp_id=None,
             name=None,
             no_data=False,
             options=None,
             coordinates=None,
             object_name=None):
    """ This is a general data read interface. It will call the specific data read interface
        for the registered data sources
    """
    if (name is None):
        raise ValueError("Data name is not given.")
    try:
        f = get_data_function(data_source)
    except ValueError as e:
        raise e
    if (f is None):
        raise ValueError("No get_data function is associated with data source " + data_source)
    if (type(coordinates) is dict):
        _coordinates = []
        for c_name in coordinates.keys():
            _coordinates.append(Coordinate(name=c_name,c_range=coordinates[c_name]))
    else:
        _coordinates = coordinates

    try:
        d = f(exp_id, name, no_data=no_data, options=options, coordinates=_coordinates)
    except Exception as e:
        raise e
    if (type(d) is not DataObject):
        raise ValueError("No data receiveid.")
    d.data_source = data_source
    try:
        d.check()
    except (ValueError, TypeError) as e:
        raise e
        
    if ((object_name is not None) and (d is not None)):
        add_data_object(d,object_name)
    return d


def add_coordinate(object_name,
                   coordinates=None,
                   exp_id='*',
                   options=None,
                   output_name=None):
    """ This is the function to add coordinates to a data object in flap storage.
        This function is an interface to the add_coordinate method of flap.DataObject

        INPUT:
            object_name, exp_ID: These identify the data object in the storage
            coordinates: List of new coordinates (string list)
            output_name: The name of the new data object in the storage. If None
                         the input will be overwritten
            options: Dictionary
                'exp_ID': Use this exp_id for calculating coordinates instead of the one
                          in the data object
                'data_source' : Use this data source instead of the one in the
                                data object.
                Other elements of options are passed over to flap.add_coordinate()
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        exp_id_new = options['exp_ID']
        del options['exp_ID']
    except (KeyError, TypeError):
        exp_id_new = None
    try:
        data_source_new = options['data_source']
        del options['data_source']
    except (KeyError, TypeError):
        data_source_new = None
    try:
        d.add_coordinate(coordinates=coordinates,
                         data_source=data_source_new,
                         exp_id=exp_id_new,
                         options=options)
    except Exception as e:
        raise e
    if (output_name is None):
        _output_name = object_name
    else:
        _output_name = output_name
    try:
        add_data_object(d,_output_name)
    except Exception as e:
        raise e



def plot(object_name,exp_id='*',axes=None, slicing=None, summing=None, options=None,
         plot_type=None, plot_options=None,plot_id=None,slicing_options=None):
    """
    plot function for an object in flap storage. This is a wrapper for DataObject.plot()
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        return d.plot(axes=axes,
                      slicing=slicing, 
                      slicing_options=slicing_options,
                      summing=summing, 
                      options=options,
                      plot_type=plot_type,
                      plot_options=plot_options,
                      plot_id=plot_id)
    except Exception as e:
        raise e


def slice_data(object_name,exp_id='*',output_name=None,slicing=None,summing=None,options=None):
    """
    slice function for an object in flap storage. This is a wrapper for DataObject.slicec_data()
    If output name is set the sliced object will be written back to the flap storage under this name.
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        ds=d.slice_data(slicing=slicing,summing=summing,options=options)
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(ds,output_name)
        except Exception as e:
            raise e
    return ds

def apsd(object_name,exp_id='*',output_name=None, coordinate=None, intervals=None, options=None):
    """
    Auto Power Spetctrum for an object in flap storage. This is a wrapper for DataObject.apsd()
    If output name is set the APSD object will be written back to the flap storage under this name.
    If not set the APSD object will be written back with its original name.
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        ds=d.apsd(coordinate=coordinate, intervals=intervals, options=options)
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(ds,output_name)
        except Exception as e:
            raise e
    return ds

def cpsd(object_name, ref=None, exp_id='*', output_name=None, coordinate=None, intervals=None, options=None):
    """
    Cross Power Spetctrum between two objects in flap storage. (ref can also be a data object.) 
    This is a wrapper for DataObject.cpsd()
    If output name is set the CPSD object will be written back to the flap storage under this name.
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    if (ref is not None):
        if (type(ref) is str):
            try:
                d_ref = get_data_object(ref,exp_id=exp_id)
            except Exception as e:
                raise e
        else:
            d_ref = ref
        if (type(d_ref) is not DataObject):
            raise ValueError("Invalid reference data structure. Should be string or flap.DataObject.")
    else:
        d_ref = None
    try:
        ds=d.cpsd(ref=d_ref, coordinate=coordinate, intervals=intervals, options=options)
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(ds,output_name)
        except Exception as e:
            raise e
    return ds

def ccf(object_name, ref=None, exp_id='*', ref_exp_id='*', output_name=None, coordinate=None, intervals=None, options=None):
    """
    Cross Corrrelation Function or covariance calculation between two objects in flap storage. 
    This is a wrapper for DataObject.ccf()
    If output name is set the CPSD object will be written back to the flap storage under this name.
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    if (ref is not None):
        if (type(ref) is str):
            try:
                d_ref = get_data_object(ref,exp_id=exp_id)
            except Exception as e:
                raise e
        else:
            d_ref = ref
        if (type(d_ref) is not DataObject):
            raise ValueError("Invalid reference data structure. Should be string or flap.DataObject.")
    else:
        d_ref = None
    try:
        ds=d.ccf(ref=d_ref, coordinate=coordinate, intervals=intervals, options=options)
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(ds,output_name)
        except Exception as e:
            raise e
    return ds

def detrend(object_name,exp_id='*',output_name=None, coordinate=None, intervals=None, options=None):
    """ DETREND signal(s)
        INPUT:
            object_name: Name of the object in flap storage (string) 
            exp_id: Experiment ID
            output_name: Name of the output object. If set result will be  stored under this name.
            coordinate: The coordinate for the detrend. If necessary data will be fitted with this coordinate
                        as x value.
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
            options:
              'Trend removal': Trend removal description (see also _trend_removal()). A list, string or None.
                None: Don't remove trend.
                Strings:
                  'mean': subtract mean
                Lists:
                  ['poly', n]: Fit an n order polynomial to the data and subtract.
                        Trend removal will be applied to each interval defined by slicing
                        separately.
        Return value:
            The resulting data object.
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        ds=d.detrend(coordinate=coordinate, intervals=intervals, options=options)
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(ds,output_name)
        except Exception as e:
            raise e
    return ds

def filter_data(object_name, exp_id='*', output_name=None, coordinate=None, intervals=None, options=None):
    """ 1D Data filter.
    INPUT:
        object_name: Name of the object in flap storage (string) 
        exp_id: Experiment ID
        output_name: Name of the output object. If not set object_name will be used.
        coordinate: The x coordinate for the trend removal.
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
        options:
            'Type' :
                None: Do nothing.
                'Int': Single term IIF filter, like RC integrator.
                'Diff': Single term IIF filter, like RC differentiator.
            'Tau': time constant for integrator/differentiator (in units of the coordinate)
    Return value: The data object with the filtered data.
    """
    
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        ds=d.filter_data(coordinate=coordinate, intervals=intervals, options=options)
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(ds,output_name)
        except Exception as e:
            raise e
    return ds

def save(data, filename, exp_id='*',  options=None):
    """
    Save one or more flap.DataObject-s using pickle.
    
    INPUT:
        data: If flap.DataObject than save this.
              If string or string list then find these data objects in flap storage and save them. 
              Will also use exp_id to select data objects. These data objects can be restored
              into flap storage using load.
              If any other object save it.
        exp_id: Experiment ID to use in conjuction with data of it is a string.
        options: None at present
        filename: Name of the file to save to.
    """
    # Checking for string in data name.
    # If string or list of strings data will be taken from flap storage
    if (type(data) is str):
        _data = [data]
    elif (type(data) is list):
        for d in data:
            if (type(d) is not str):
                break
        else:
            _data = data
    try:
        # If working from flap storage
        _data
        # Extending exp_id in case it is shorter than _data
        if (type(exp_id) is not list):
            _exp_id = [exp_id]*len(_data)
        else:
            _exp_id = exp_id
            if (len(_exp_id) < len(data)):
                for i in range(len(data)-len(_exp_id)):
                    _exp_id.append(_exp_id[-1])
        names = []
        exps = []
        for d,e in zip(_data,_exp_id):
            names_1, exps_1 = find_data_objects(d,exp_id=e) 
            names += names_1
            exps += exps_1
        save_data = {'FLAP storage data':True}
        for dn,en in zip(names,exps):
            try:
                d = flap.get_data_object(dn,exp_id=en)
            except Exception as err:
                raise err
            save_data[dn] = d
    except:
        save_data = {'FLAP any data':data}
    try:
        f = io.open(filename,"wb")
    except:
        raise IOError("Cannot open file: "+filename)
    try:
        pickle.dump(save_data,f)
    except Exception as e:
        raise e
    try:
        f.close()
    except Exception as e:
        raise e

        
def load(filename,options=None):
    """
    Loads data saved with flap.save()
    If the data in the file were written from the flap storage it will also be
    loaded there, unless  no_storage is set to True.
    
    INPUT:
        filename: Name of the file to read.
        options: "No storage": (bool) If True don't store data in flap storage just return a list of them.
    Return value:
        If data was not written from flap storage the original object will be returned.
        If it was written out from flap storage a list of the read objects will be returned..
    """
    default_options = {"No storage":False}
    try:
        _options = flap.config.merge_options(default_options, options,
                                             section='Save/Load')
    except ValueError as e:
        raise e
    if (type(_options['No storage']) is not bool):
        raise TypeError("Option 'No storage' should be boolean.")

    try:
        f = io.open(filename,"rb")
    except:
        raise IOError("Cannot open file: "+filename)
    try:
        save_data = pickle.load(f)
    except Exception as e:
        raise e
    try:
        f.close()
    except Exception as e:
        raise e
    if (type(save_data) is not dict):
        raise ValueError("File "+filename+" is not a flap save file.")
    if (list(save_data.keys())[0] == 'FLAP any data'):
        return save_data[list(save_data.keys())[0]]
    elif (list(save_data.keys())[0] == 'FLAP storage data'):
        output = []
        del save_data['FLAP storage data']
        for name in save_data.keys():
            output.append(save_data[name])
            if (not _options['No storage']):
                try:
                    flap.add_data_object(save_data[name],name)
                except Exception as e:
                    raise e
        return output
    else:
        raise ValueError("File "+filename+" is not a flap save file.")
        
def abs_value(object_name,exp_id='*',output_name=None):
    """
    Absolute value
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        d_out = d.abs_value()
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(d_out,output_name)
        except Exception as e:
            raise e
    return d_out

def phase(object_name,exp_id='*',output_name=None):
    """
    Phase
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        d_out = d.phase()
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(d_out,output_name)
        except Exception as e:
            raise e
    return d_out

def real(object_name,exp_id='*',output_name=None):
    """
    Real value. (Does nothing for real data)
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        d_out = d.real()
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(d_out,output_name)
        except Exception as e:
            raise e
    return d_out

               
def imag(object_name,exp_id='*',output_name=None):
    """
    Real value. (Does nothing for real data)
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        d_out = d.imag()
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(d_out,output_name)
        except Exception as e:
            raise e
    return d_out

def error_value(object_name,exp_id='*',output_name=None, options=None):
    """
        Returns a data object with the error of self in it.
        options: 'High': Use high error if error is asymmetric
                 'Low': Use low error is error is asymmetric
    """
    try:
        d = get_data_object(object_name,exp_id=exp_id)
    except Exception as e:
        raise e
    try:
        d_out = d.error_value(options=options)
    except Exception as e:
        raise e
    if (output_name is not None):
        try:
            add_data_object(d_out,output_name)
        except Exception as e:
            raise e
    return d_out
    
    
    

