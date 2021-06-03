# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:18:25 2019

@author: Zoletnik
@coauthor: Lampert

Tools for the FLAP module

"""
import copy
import numpy as np
import fnmatch
from flap import VERBOSE
#from decimal import Decimal                                                    #UNUSED

def del_list_elements(input_list, indices):
    """ delete elements from a list
    """
    l = copy.deepcopy(input_list)
    for i in sorted(indices, reverse=True):
        del l[i]
    return l


def unify_list(list1, list2):
    """ Returns list with elements present in any of the two lists.
        Output list is sorted.
    """
    unified_list = copy.deepcopy(list1)
    for d in list2:
        try:
            unified_list.index(d)
        except ValueError:
            unified_list.append(d)
    return sorted(unified_list)

def select_signals(signal_list, signal_spec):
    """
    Selects signals from a signal list following signal specifications.

    signal_list: List of strings of possible signal names

    signal_spec: List of strings with signal specifications including wildcards
                 Normal Unix file name wildcards are accepted and extended with
                 [<num>-<num>] type expressions so as e.g. a channel range can be selected.

    Returs select_list, select_index
       select_list: List of strings with selected signal names
       select_index: List of indices to signal list of the selected signals

    Raises ValueError if there is no match for one specification
    """

    if (type(signal_spec) is not list):
        _signal_spec = [signal_spec]
    else:
        _signal_spec = signal_spec

    if ((len(_signal_spec) == 0) or (signal_list == [])):
        raise ValueError("No signal list or signal specification.")

    select_list = []
    select_index = []
    for ch in _signal_spec:
        # This will add a list of possible channel names to the _signal_spec while [<num>-<num>} is found
        startpos = 0
        extended = False
        extended_list = []
        while 1:
            # Searching for opening and closing []
            for i1 in range(startpos,len(ch)):
                if (ch[i1] == '['):
                    break
            else:
                break
            if (i1 == len(ch)-1):
                break
            for i2 in range(i1+1,len(ch)):
                if (ch[i2] == ']'):
                    break
            else:
                break
            # found the opening and closing bracket
            # Trying to interpret the string between the brackets as <int> - <int>
            try:
                nums = ch[i1+1:i2].split('-')
                nums = [int(nums[0]), int(nums[1])]
                # Extracting the strings before and after the []
            except Exception:
                if (i2 >= len(ch)-2):
                    break
                startpos = i2+1
                continue
            if (i1 == 0):
                str1 = ""
            else:
                str1 = ch[0:i1]
            if (i2 < len(ch)-1):
                str2 = ch[i2+1:len(ch)]
            else:
                str2 = ""
            extended = True
            for i in range(nums[0],nums[1]+1):
                # Adding all the possible strings
                extended_list.append(str1+str(i)+str2)
            startpos = i2+1
            continue

        ch_match = False

        # if extended list is created not checking original name
        if (not extended):
            for i in range(len(signal_list)):
                if (fnmatch.fnmatch(signal_list[i], ch)):
                    select_list.append(signal_list[i])
                    select_index.append(i)
                    ch_match = True
        if (extended):
            for i in range(len(signal_list)):
                for che in extended_list:
                    if (fnmatch.fnmatch(signal_list[i], che)):
                        select_list.append(signal_list[i])
                        select_index.append(i)
                        ch_match = True
                        break
        if (not ch_match):
            raise ValueError("Signal name: " + ch + " is not present.")

    return select_list, select_index

def chlist(chlist=None, chrange=None, prefix='', postfix=''):
    """
    Creates a channel (signal) list name from a prefix, postfix a channel list and a list of channel
    ranges
    """
    ch = []
    if (chlist is not None):
        ch.extend(chlist)
    if (chrange is not None):
        n = int(len(chrange) / 2)
        for i in range(n):
            ch.extend(list(range(chrange[i * 2],chrange[i * 2 + 1] + 1)))
    ch_str = []
    for c in ch:
        ch_str.append(prefix + str(c) + postfix)
    return ch_str


def submatrix_index(mx_shape, index):
    """ Given an arbitrary dimension matrix with shape mx_shape the tuple to
        extract a submatrix is created and returned.
        The elements in each dimension are selected by index.

    Input:
        mx_shape: Shape of the matrix
        index: Tuple or list of 1D numpy arrays. The length should be equal to the
        length of mx_shape. Each array contains the indices for the
               corresponding dimension.
    Return value:
        A tuple of index matrices. Each index matrix has the same shape as
        described by index. Each matrix contains the indices for one dimension
        of the matrix. This tuple can be directly used for indexing the matrix.
    """

    index_arrays = []
    
#    mx_dim = len(mx_shape)
#    for i in range(mx_dim):
#        # Creating a matrix with 1 element in each direction and the
#        # number of elements in index[i] in the i-th dimension
#        shape = [1] * mx_dim
#        shape[i] = index[i].size
#        # Creating this matrix
#        ind = np.zeros(tuple(shape),dtype=int)
#        # Creating a list of indices with 0 at all places except at i where '...'
#        ind_ind = [0] * mx_dim
#        ind_ind[i] = ...
#        ind[tuple(ind_ind)] = index[i]
#        # Expanding this matrix in all other dimensions
#        for j in range(mx_dim):
#            if (j != i):
#                ind = np.repeat(ind,index[j].size,j)
#        index_arrays.append(ind)

    for i in range(len(mx_shape)):                                 #THIS IS A SOLUTION FOR LARGE MATRICES, BUT NOT COMMITED
        index_arrays.append(slice(min(index[i]),max(index[i])+1))
        
    return tuple(index_arrays)


def expand_matrix(mx,new_shape,dim_list):
    """ Insert new dimensions to a matrix so as it has <new shape> shape.
        The original dimensions are at dim_list dimensions

        Input:
            mx: The matrix with arbitrary dimensions.
            new_shape: This will be the new shape
            dim_list: This is a list of dimensions where mx is in the output
                      matrix. len(dim_list) == mx.ndim
    """

    act_dim = 0
    act_list = 0
    if (type(dim_list) is not list):
        _dim_list = dim_list
    else:
        _dim_list = dim_list
    for i in range(len(new_shape)):
        if ((act_list >= len(_dim_list)) or (act_dim < _dim_list[act_list])):
            mx = np.expand_dims(mx,act_dim)
            mx = np.repeat(mx,new_shape[act_dim],act_dim)
        else:
            act_list += 1
        act_dim += 1
    return mx
    
def flatten_multidim(mx, dim_list):
    """ Flatten the dimensions in dim_list to dim_list[0]
        Returns the modified matrix and a mapping from the original to the new dimension list.
        The mapping will be None for the flattened dimension in dim_list even if
        flattening was not done. The dimension numbers in the dimension list assume that
        the flattened dimensions are removed.
    """
    if (len(dim_list) <= 1):
        dimension_mapping = [None]*mx.ndim
        count = 0
        for i in range(mx.ndim):
            try:
                dim_list.index(i)
            except ValueError:
                dimension_mapping[i] = count
                count += 1
        return mx, dimension_mapping

    out_shape = []
    flat_size = 1
    for d in dim_list:
        flat_size *= mx.shape[d]
    #This is the mapping from the remaining dimensions to the output matrix dimensions
    out_dim_mapping = [None]*mx.ndim
    flat_dim_mapping = [None]*mx.ndim
    out_dim_counter = 0
    flat_dim_counter = 0
    for i in range(mx.ndim):
        try:
            dim_list_i = dim_list.index(i)
            if (dim_list_i == 0):
                out_shape.append(flat_size)
                flat_dim_counter += 1
        except ValueError:
            out_shape.append(mx.shape[i])
            out_dim_mapping[i] = out_dim_counter
            flat_dim_mapping[i] = flat_dim_counter
            out_dim_counter += 1
            flat_dim_counter += 1

    # Creating index matrices for each dimension of dimension list and flattening them to
    # create index
    flat_submx_shape = [ mx.shape[x] for x in dim_list]
    ind_flat_list =[]
    for i in range(len(flat_submx_shape)):
        ind = np.arange(flat_submx_shape[i])
        ind_flat_list.append(expand_matrix(ind, flat_submx_shape, [i]).flatten())
    # Creating as many index matrices as the number of dimensions of mx
    mx_list = []
    for i in range(mx.ndim):
        try:
            dim_list_i = dim_list.index(i)
            ind = ind_flat_list[dim_list_i]
            out_dim = dim_list[0]
        except ValueError:
            ind = np.arange(mx.shape[i])
            out_dim = flat_dim_mapping[i]
        mx_list.append(expand_matrix(ind, out_shape, [out_dim]))

    return  mx[tuple(mx_list)], out_dim_mapping

def multiply_along_axes(a1_orig, a2_orig, axes,keep_a1_dims=True):
    """ 
    Multiplies two arrays along given axes. 
    INPUT:
        a1_orig: Array 1.
        a2_orig: Array 2.
        axes: List of two axis numbers or list of two lists of axis numbers
        keep_1_dims: (bool) 
                     If True: The output array has dimensions of a1 followed by a2 with the common dims removed
                     If False: The output array has the a1 dimensions without common dims then the common dims
                               followed by a2 with the common dims removed
    Return values:
        a, axis_source, axis_number
            a: An array with dimension number a1.dim+a2.dim-1. 
            axis_source: List of integers telling the source array for each output axis ( 0 or 1)
            axis_number: Axis numbers in the arrays listed in axes_source
    """
    if (type(axes[0]) is not list):
        axes[0] = [axes[0]]
    if (type(axes[1]) is not list):
        axes[1] = [axes[1]]
    for i in range(len(axes[0])):       
        if (a1_orig.shape[axes[0][i]] != a2_orig.shape[axes[1][i]]):
            raise ValueError("Incompatible shapes.")

    a1 = a1_orig
    a2 = a2_orig
    a1_shape = a1.shape
    a1_axes = list(range(a1.ndim))
    a2_shape = a2.shape
    a2_axes = list(range(a2.ndim))
    for i in range(len(axes[0])):
        # Finding the axis
        ind = a1_axes.index(axes[0][i])
        # Move from a1 the processing axis to the end
        a1 = np.moveaxis(a1,ind,-1)
        # Following this change in the axis list
        del a1_axes[ind]
        a1_axes.append(axes[0][i])
        # Move from a2 the processing axis to the front
        ind = a2_axes.index(axes[1][i])
        a2 = np.moveaxis(a2,ind,i)
        del a2_axes[ind]
        a2_axes.insert(i,axes[1][i])
    out_shape = list(a1.shape) + list(a2.shape)[len(axes[0]):]
    for i in range(len(out_shape)-len(a1_shape)):
        a1 = np.expand_dims(a1,-1)
    for i in range(len(out_shape)-len(a2_shape)):
        a2 = np.expand_dims(a2,0)    
    r = a1 * a2
    if (keep_a1_dims):
        # Moving the processing axes back where they were in the original array
        # We have to move the axis in increasing destination order
        sort_axes = axes[0]
        sort_axes.sort()
        for i in range(len(sort_axes)):
            ind = a1_axes.index(sort_axes[i])
            r = np.moveaxis(r, ind, sort_axes[i])
            del a1_axes[ind]
            a1_axes.insert(sort_axes[i],sort_axes[i])
    axis_source = [0]*a1_orig.ndim + [1]*(a2_orig.ndim - len(axes[0]))
    axis_number = a1_axes + a2_axes[len(axes[1]):]
    return r, axis_source, axis_number

def move_axes_to_end(mx_orig,axes):
    """ Moves the listed axes to the end.
    """
    mx = mx_orig
    mx_axes = list(range(mx.ndim))
    for i in range(len(axes)):
        # Finding the axis
        ind = mx_axes.index(axes[i])
        # Move from to the end
        mx = np.moveaxis(mx,ind,-1)
        # Following this change in the axis list
        del mx_axes[ind]
        mx_axes.append(axes[i])
    return mx, mx_axes
    
def move_axes_to_start(mx_orig,axes):
    """ Moves the listed axes to the start axes.
    """
    mx = mx_orig
    mx_axes = list(range(mx.ndim))
    for i in range(len(axes)):
        # Finding the axis
        ind = mx_axes.index(axes[i])
        # Move from to the end
        mx = np.moveaxis(mx,ind,0)
        # Following this change in the axis list
        del mx_axes[ind]
        mx_axes.insert(i,axes[i])
    return mx, mx_axes
    
def find_str_match(value, options):
    """
    Given value string and a list of possibilities in the list of strings option
    find matches assuming value is an abbreviation. Return ValueError if no match 
    or multiple match is found.
    If one match is found return the matching string
    """
    if (type(value) is not str):
        raise TypeError("Invalid value.")
    matches = []
    for s in options:
        if (value == s[0:min([len(value),len(s)])]):
            if (len(matches) != 0):
                raise ValueError("Too short abbreviation: "+value)
            matches.append(s)
    if (len(matches) == 0):
        raise ValueError("No match for "+value)
    return matches[0]

def grid_to_box(xdata,ydata):
    """
    Given 2D x and y coordinate matrices create box coordinates around the points as
    needed by matplotlib.pcolomesh.
    xdata: X coordinates. 
    ydata: Y coordinates. 
    In both arrays x direction is along first dimension, y direction along second dimension.
    Returns xbox, ybox.
    """
    xdata = np.transpose(xdata.astype(float))
    xbox_shape = list(xdata.shape)
    xbox_shape[0] += 1
    xbox_shape[1] += 1
    xbox = np.empty(tuple(xbox_shape),dtype=xdata.dtype)
    xbox[1:,1:-1] = (xdata[:,:-1] + xdata[:,1:]) / 2 
    xbox[1:-1,1:-1] = (xbox[2:,1:-1] + xbox[1:-1,1:-1]) / 2
    xbox[1:-1,0] = ((xdata[1:,0] + xdata[:-1,0]) / 2 - xbox[1:-1,1]) * 2 + xbox[1:-1,1]
    xbox[1:-1,-1] = ((xdata[1:,-1] + xdata[:-1,-1]) / 2 - xbox[1:-1,-2]) * 2 + xbox[1:-1,-2]
    xbox[0,1:-1] = ((xdata[0,:-1] + xdata[0,1:]) / 2 - xbox[1,1:-1]) * 2 + xbox[1,1:-1]
    xbox[-1,1:-1] = ((xdata[-1,:-1] + xdata[-1,1:]) / 2 - xbox[-2,1:-1]) * 2 + xbox[-2,1:-1]
    xbox[0,0] = xbox[1,1] + (xbox[0,1] - xbox[1,1]) + (xbox[1,0] - xbox[1,1])
    xbox[-1,-1] = xbox[-2,-2] + (xbox[-1,-2] - xbox[-2,-2]) + (xbox[-2,-1] - xbox[-2,-2])
    xbox[0,-1] = xbox[1,-2] + (xbox[0,-2] - xbox[1,-2]) + (xbox[1,-1] - xbox[1,-2]) 
    xbox[-1,0] = xbox[-2,1] + (xbox[-1,1] - xbox[-2,1]) + (xbox[-2,0] - xbox[-2,1])
      
    ydata = np.transpose(ydata.astype(float))
    ybox_shape = list(ydata.shape)
    ybox_shape[0] += 1
    ybox_shape[1] += 1
    ybox = np.empty(tuple(ybox_shape),dtype=ydata.dtype)
    ybox[1:,1:-1] = (ydata[:,:-1] + ydata[:,1:]) / 2 
    ybox[1:-1,1:-1] = (ybox[2:,1:-1] + ybox[1:-1,1:-1]) / 2
    ybox[1:-1,0] = ((ydata[1:,0] + ydata[:-1,0]) / 2 - ybox[1:-1,1]) * 2 + ybox[1:-1,1]
    ybox[1:-1,-1] = ((ydata[1:,-1] + ydata[:-1,-1]) / 2 - ybox[1:-1,-2]) * 2 + ybox[1:-1,-2]
    ybox[0,1:-1] = ((ydata[0,:-1] + ydata[0,1:]) / 2 - ybox[1,1:-1]) * 2 + ybox[1,1:-1]
    ybox[-1,1:-1] = ((ydata[-1,:-1] + ydata[-1,1:]) / 2 - ybox[-2,1:-1]) * 2 + ybox[-2,1:-1]
    ybox[0,0] = ybox[1,1] + (ybox[0,1] - ybox[1,1]) + (ybox[1,0] - ybox[1,1])
    ybox[-1,-1] = ybox[-2,-2] + (ybox[-1,-2] - ybox[-2,-2]) + (ybox[-2,-1] - ybox[-2,-2])
    ybox[0,-1] = ybox[1,-2] + (ybox[0,-2] - ybox[1,-2]) + (ybox[1,-1] - ybox[1,-2]) 
    ybox[-1,0] = ybox[-2,1] + (ybox[-1,1] - ybox[-2,1]) + (ybox[-2,0] - ybox[-2,1])
    
    return xbox,ybox

def time_unit_translation(time_unit=None,max_value=None):
    
    """
    This tool returns the numerical pre-fix of the time_unit. It is almost
    obsolate due to the existance of the methos named unit_conversion, but
    this script can provide backwards conversion (used by flap_mdsplus.py)
    time_unit: can be a string e.g. ms or a number e.g. 1e-3
    max_value: returns a suspected time unit based on this input
                e.g.: max_value=1000. 
    """
    
    if (str(type(time_unit)) == 'str' or 
       str(type(time_unit)) == "<class 'numpy.str_'>"):
        _time_unit=time_unit.lower()
    else:
        _time_unit=time_unit
    if ((_time_unit == ' ') or (_time_unit is None)) and (max_value is not None):
        #Raise awareness:
        if VERBOSE:
            print('Time unit: \''+str(_time_unit)+'\'')
            print('Time unit translation based on values only works for shots under 1000s.')
        value_translation=[[0.,1e3,1e6,1e9,1e12],
                           ['s','ms','us','ns','ps']]
        for i in range(len(value_translation[0])-1):
            if (max_value > value_translation[0][i] and max_value < value_translation[0][i+1]):
                _time_unit=value_translation[1][i]
            elif max_value > value_translation[0][4]:
                _time_unit=value_translation[1][4]
    translation={'seconds':1,
                 'second':1,
                 's':1,
                 'milliseconds':1e-3,
                 'millisecond':1e-3,
                 'ms':1e-3,
                 'microseconds':1e-6,
                 'microsecond':1e-6,
                 'us':1e-6,
                 'nanoseconds':1e-9,
                 'nanosecond':1e-9,
                 'ns':1e-9,
                 'picoseconds':1e-12,
                 'picosecond':1e-12,
                 'ps':1e-12
                 }
    if (_time_unit in translation.keys()):
        return translation[_time_unit]
    else:
        if type(_time_unit) is not str:
            backwards_translation=[[1,1e-3,1e-6,1e-9,1e-12],
                                   ['s','ms','us','ns','ps']]
            for i in range(len(backwards_translation[0])):
                if backwards_translation[0][i] == _time_unit:
                    return backwards_translation[1][i]
        else:
            print(_time_unit+' was not found in the translation. Returning 1.')
            return 1
    
def unit_conversion(original_unit=None,
                    new_unit=None
                    ):
    
    #The code provides a unit conversion between any unit types for most
    #of the prefixes. 
    #There are certain limitations:
    #   The unit compatibility is not checked (e.g. mm-->MegaHertz is allowed)
    
    known_conversions_full={'Terra':1e12,
                            'Giga':1e9,
                            'Mega':1e6,
                            'kilo':1e3,
                            'milli':1e-3,
                            'micro':1e-6,
                            'nano':1e-9,
                            'pico':1e-12,
                            }
    
    known_conversions_short={'T':1e12,
                             'G':1e9,
                             'M':1e6,
                             'k':1e3,
                             'm':1e-3,
                             'u':1e-6,
                             'n':1e-9,
                             'p':1e-12
                             }
    
    original_unit_translation=None
    new_unit_translation=None
    
    #Trying to find the long unit names in the inputs
    
    for keys_full in known_conversions_full:
        if keys_full in original_unit:
            original_unit_translation=known_conversions_full[keys_full]
        if keys_full in new_unit:
            new_unit_translation=known_conversions_full[keys_full]
            
    if original_unit_translation is None:
        if ((len(original_unit) == 1) or (len(original_unit) > 3)): # SI units are longer than 3 if using the full name
            original_unit_translation=1.
        else:
            for keys_short in known_conversions_short:
                if keys_short == original_unit[0]:
                    original_unit_translation=known_conversions_short[keys_short]
                    
    if new_unit_translation is None:                    
        if ((len(new_unit) == 1) or (len(new_unit) > 3)):
            new_unit_translation=1.
        else:
            for keys_short in known_conversions_short:
                if keys_short == new_unit[0]:
                    new_unit_translation=known_conversions_short[keys_short]
                    
    if original_unit_translation is None: 
        print('Unit translation cannot be done for the original unit, '+original_unit+'. Returning 1.')
        if VERBOSE:
            if len(original_unit) > 3:
                print('Known conversion units are:')
                print(known_conversions_full)
                print('\n')
            else:
                print('Known conversion units are:')
                print(known_conversions_short)
                print('\n')
        original_unit_translation=1.
        
    if new_unit_translation is None:
        print('Unit translation cannot be done for the new unit, '+new_unit+'. Returning 1.')
        if VERBOSE:
            if len(original_unit) > 3:
                print('Known conversion units are:')
                print(known_conversions_full)
                print('\n')
            else:
                print('Known conversion units are:')
                print(known_conversions_short)
                print('\n')
            new_unit_translation=1.
        
    return original_unit_translation/new_unit_translation

def polyfit_2D(x=None,
               y=None,
               values=None,
               sigma=None,
               order=None,
               irregular=False,
               return_covariance=False,
               return_fit=False):
    
    if sigma is None:
        sigma=np.zeros(values.shape)
        sigma[:]=1.
    else:
        if sigma.shape != sigma.shape:
            raise ValueError('The shape of the errors do not match the shape of the values!')
    
    if not irregular:
        if len(values.shape) != 2:
            raise ValueError('Values are not 2D')
        if x is not None and y is not None:
            if x.shape != values.shape or y.shape != values.shape:
                raise ValueError('There should be as many points as values and their shape should match.')
        if order is None:
            raise ValueError('The order is not set.')
        if (x is None and y is not None) or (x is not None and y is None):
            raise ValueError('Either both or neither x and y need to be set.')
        if x is None and y is None:
            polynom=np.asarray([[i**k * j**l / sigma[i,j] for k in range(order+1) for l in range(order-k+1)] for i in range(values.shape[0]) for j in range(values.shape[1])]) #The actual polynomial calculation
        else:
            polynom=np.asarray([[x[i,j]**k * y[i,j]**l / sigma[i,j] for k in range(order+1) for l in range(order-k+1)] for i in range(values.shape[0]) for j in range(values.shape[1])]) #The actual polynomial calculation
            
        original_shape=values.shape
        values_reshape=np.reshape(values/sigma, values.shape[0]*values.shape[1])
        
        covariance_matrix=np.linalg.inv(np.dot(polynom.T,polynom))
        
        coefficients=np.dot(np.dot(covariance_matrix,polynom.T),values_reshape) #This performs the linear regression
        
        if not return_fit:
            if return_covariance:
                return (coefficients, covariance_matrix)
            else:
                return coefficients 
        else:
            return np.reshape(np.dot(polynom,coefficients),original_shape)
    else:
        if x.shape != y.shape or x.shape != values.shape:
            raise ValueError('The points should be an [n,2] vector.')
        if len(x.shape) != 1 or len(y.shape) != 1 or len(values_reshape.shape) != 1:
            raise ValueError('x,y,values should be a 1D vector when irregular is set.')
        if order is None:
            raise ValueError('The order is not set.')
        polynom=np.asarray([[x[i]**k * y[i]**l for k in range(order+1) for l in range(order-k+1)] for i in range(values.shape[0])]) #The actual polynomial calculation
        if not return_fit:
            return np.dot(np.dot(np.linalg.inv(np.dot(polynom.T,polynom)),polynom.T),values) #This performs the linear regression
        else:
            return np.dot(polynom,np.dot(np.dot(np.linalg.inv(np.dot(polynom.T,polynom)),polynom.T),values))
        
def reorder_2d_ccf_indices(res,             #Original result of the ccf calculation from the fft
                           cd,              #Correlation dimensions
                           ind_in1,         #Input indices 1 from 1D
                           ind_in2,         #Input indices 2 from 1D
                           ind_out1,        #Output indices 1 from 1D
                           ind_out2):       #Output indices 2 from 1D

    """
    This helper function reorganizes the 2D cross-correlation or auto-correlation
    functions in order to have the maximum in the middle of the matrix and
    not somewhere around the corner. Used in _ccf in spectral_analysis.py
    """
    
    
    corr = np.empty(res.shape,dtype=res.dtype)
    
    ind_out_1=ind_out1[:]
    ind_out_2=ind_out1[:]
    ind_out_3=ind_out1[:]
    ind_out_4=ind_out1[:]

    ind_out_1[cd[0]]=ind_out1[cd[0]]
    ind_out_1[cd[1]]=ind_out1[cd[1]]
    ind_out_2[cd[0]]=ind_out2[cd[0]]
    ind_out_2[cd[1]]=ind_out1[cd[1]]
    ind_out_3[cd[0]]=ind_out2[cd[0]]
    ind_out_3[cd[1]]=ind_out2[cd[1]]
    ind_out_4[cd[0]]=ind_out1[cd[0]]
    ind_out_4[cd[1]]=ind_out2[cd[1]]
    
    ind_in_1=ind_in1[:]
    ind_in_2=ind_in1[:]
    ind_in_3=ind_in1[:]
    ind_in_4=ind_in1[:]
    
    ind_in_1[cd[0]]=ind_in1[cd[0]]
    ind_in_1[cd[1]]=ind_in1[cd[1]]
    ind_in_2[cd[0]]=ind_in2[cd[0]]
    ind_in_2[cd[1]]=ind_in1[cd[1]]
    ind_in_3[cd[0]]=ind_in2[cd[0]]
    ind_in_3[cd[1]]=ind_in2[cd[1]]
    ind_in_4[cd[0]]=ind_in1[cd[0]]
    ind_in_4[cd[1]]=ind_in2[cd[1]]
    
    corr[tuple(ind_out_1)] = res[tuple(ind_in_1)] 
    corr[tuple(ind_out_2)] = res[tuple(ind_in_2)] 
    corr[tuple(ind_out_3)] = res[tuple(ind_in_3)]
    corr[tuple(ind_out_4)] = res[tuple(ind_in_4)]
    
    return corr