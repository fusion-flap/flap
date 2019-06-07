
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:37:07 2018

@author: Sandor Zoletnik

This module creates storage and its interface for the flap package.



"""


import copy
import matplotlib.pyplot as plt
import numpy as np
from decimal import *

from .tools import *
from .coordinate import *
from .data_object import *
import flap.config

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
            return
        
        for i_names in range(len(names)):
            d = get_data_object(names[i_names], exp_id=exps[i_names])
            print('-----------------------------')
            if (d.data_title is None):
                dtit = ""
            else:
                dtit = d.data_title
            s = names[i_names] + "(exp_id:" + exps[i_names] + ') data_title:"' + dtit + '"'
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
                s += "    '" + c.unit.name + "'[" + c.unit.unit + ']'
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
            print(s)


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


def list_data_objects(name=None, exp_id='*'):
    """ Return the list of data objects in flap storage.
        The returned data is a key list.
    """
    return flap_storage.list_data_objects(name=name, exp_id=exp_id)


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
    d.data_source = data_source
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
         plot_type=None, plot_options=None,plot_id=None):
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
    
    
    
