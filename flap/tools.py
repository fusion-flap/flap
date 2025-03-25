# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:18:25 2019

Tools for the FLAP module

@author: Sandor Zoletnik  (zoletnik.sandor@ek-cer.hu)
Centre for Energy Research

"""
import copy
import numpy as np
import fnmatch
from flap import VERBOSE
import configparser
#from decimal import Decimal                                                    #UNUSED

def del_list_elements(input_list, indices):
    """ delete elements from a list
    """
    l = copy.deepcopy(input_list)
    for i in sorted(indices, reverse=True):
        del l[i]
    return l


def unify_list(list1, list2):
    """ Return list with elements present in any of the two lists.
        Output list is sorted.
    """
    unified_list = copy.deepcopy(list1)
    for d in list2:
        try:
            unified_list.index(d)
        except ValueError:
            unified_list.append(d)
    return sorted(unified_list)

def select_signals(signal_list, signal_spec, no_check=False):
    """
    Selects signals from a signal list following signal specifications.

    Parameters
    -----------
    signal_list: list
        List of strings of possible signal names. The signals will be selected from this. If this is
        [] on None no selection will be done. Only bracket extension like
        CH[1-12] will be accepted. 
        [1-4] will result in 1,2,3,4
        [01-04] will result in 01,02,03,04

    signal_spec: list or str 
        List of strings with signal specifications including wildcards
        Normal Unix file name wildcards are accepted and extended with
        [<num>-<num>] type expressions so as e.g. a channel range can be selected.
        
    no_check: bool
        If True no error is raised when a signal is not present in the signals list.
        The signal list is used only for * regular expression in this case.
        If False an error is raised when a signal is not presen in the signal list.

    Raises
    ------
    ValueError: No signal spec

    Returns
    -------
    list: 
        List of strings with selected signal names
    list:
        List of indices to signal list of the selected signals. If signal_list is None or [] None.

    """

    if (type(signal_spec) is not list):
        _signal_spec = [signal_spec]
    else:
        _signal_spec = signal_spec

    if ((signal_list is None) or signal_list == []):
        no_signal_list = True
    else:
        no_signal_list = False
    if (len(_signal_spec) == 0):
        raise ValueError("No signal sspecification.")

    while (True):
        select_list = []
        select_index = []
        for i_ch,ch in enumerate(_signal_spec):
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
                    numstr1 = nums[0].strip()
                    numstr2 = nums[1].strip()
                    nums = [int(nums[0]), int(nums[1])]
                    for zero1 in range(len(numstr1)):
                        if (numstr1[zero1] != '0'):
                            break
                    for zero2 in range(len(numstr2)):
                        if (numstr2[zero2] != '0'):
                            break    
                except Exception:
                    # Trying as a list
                    name_list = ch[i1+1:i2].split(',')
                    if (len(name_list) > 1):
                        # Extracting the strings before and after the []
                        if (i1 == 0):
                            str1 = ""
                        else:
                            str1 = ch[0:i1]
                        if (i2 < len(ch)-1):
                            str2 = ch[i2+1:len(ch)]
                        else:
                            str2 = ""
                        extended = True
                        for s in name_list:
                            extended_list.append(str1+s+str2)
                        startpos = i2+1
                        break                       
                    if (i2 >= len(ch)-2):
                        break
                    startpos = i2+1
                    continue
                # Extracting the strings before and after the []
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
                    if (zero1 != 0):
                        strlen = max([len(numstr1),len(numstr2)])
                        format_str = ":0{:d}d".format(strlen)
                        format_str = "{" + format_str + "}"
                        s = format_str.format(i)
                    else:
                        s = str(i)
                    extended_list.append(str1+s+str2)
                startpos = i2+1
                break
            
            if (extended):
                del _signal_spec[i_ch]
                _signal_spec += extended_list
                break
                
            if (no_signal_list):
                select_index = None
                if (extended):
                    select_list += extended_list
                else:
                    select_list.append(ch)
            else:    
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
                    if (no_check):
                        select_list.append(ch)
                        select_index.append(None)
                    else:
                        raise ValueError("Signal name: " + ch + " is not present.")
        if (not extended):
            break
    return select_list, select_index


class SignalList():
    """ Class to contain a list of identical signal names suitable for inclusion in 
        a DataObject.
        
        data_type: str
           'real': data_list contains a list of valid signal names
           'complex' : data_list contains a list of 2-element listst.
                       Each element contains a final webapi name. The two-element lists
                       are interpreted as complex signals.    
        signal_list: list 
                   List of signal names or complex signal descriptions
                   Complex signal is desribed by a list of two signal names.
        invert: list or None
            If list: It should have the same length as signal_list. 
                     The elemenets should be boolean. If True the respective signal is inverted after reading from the database or cache.
            If None: Nothing to do with the signals.
    """
    def __init__(self,data_type=None,signal_list=[],invert=None):
        self.data_type = data_type
        if (self.data_type is not None and (self.data_type.lower() == 'complex')):
            for sig in signal_list:
                if (len(sig != 2)):
                    raise ValueError ('Complex signal description should have two elements.')
        self.signal_list = signal_list
        if (invert is not None):
            if (len(invert) != len(self.signal_list)):
                raise ValueError("In invert is given in signal description it should have same lenght as signal list.")
        self.invert = invert
        
        
    def add(self,s,pos=None):
        """
        Adds a SignalList to an existging one at a position in the list. The type (real,complex) of the two
        lists should be identical.

        Parameters
        ----------
        s : class SignalDesc
            The list to add.
        pos : int, optional
            The position in the list to add. If None will add to the end of the list.
            If not None will remove that element in the list and add the new list there.
            First element is 0.

        Raises
        ------
            TypeError: If the two lists have different type.
            ValueError: Unknown signal type
            
        Returns
        -------
        None.

        """
        if ((self.data_type is None) and (len(self.signal_list) == 0)):
            self.data_type = s.data_type
            self.signal_list = s.signal_list
            self.invert = s.invert
            return
        if ((s.data_type is None) and (len(s.signal_list) == 0)):
            return
        if ((self.data_type is None) or (s.data_type is None)):
            raise ValueError("Unknown data type for SignalList.")
        if (((pos is None) or (len(self.signal_list) != 1)) and (self.data_type != s.data_type)):
            raise TypeError("Cannot combine two signals")
        if ((pos is None) or (pos >= len(self.signal_list))):
            # Adding to end
            self.signal_list += s.signal_list
            if (self.invert is not None) or (s.invert is not None):
                if (self.invert is None) and (s.invert is not None):
                    self.invert = [False] * len(self.signal_list)
                if (self.invert is not None) and (s.invert is None):
                    self.invert += [False] * len(s.signal_list)
                else:
                    self.invert += s.invert
            return
        if ((pos < 0) and ((len(self.signal_list) == 0) or (-pos > len(self.signal_list)))):
            raise ValueError("Invalid position for inserting SignalList.")   
        if (pos >= 0):
            _pos = pos
        else:
            _pos = len(self.signal_list) + pos 
            self.signal_list = self.signal_list[:_pos] + s.signal_list + self.signal_list[_pos+1:]
            self.data_type = s.data_type
            if (self.invert is not None) or (s.invert is not None):
                if (self.invert is None) and (s.invert is not None):
                    self.invert = [False] * len(self.signal_list)
                if (self.invert is not None) and (s.invert is None):
                    self.invert = self.invert[:_pos] + [False] * len(s.signal_list) + self.invert[_pos+1:]
                else:
                    self.invert = self.invert[:_pos] + s.invert + self.invert[_pos+1:]         
        return
        
        
def interpret_signals(signals,exp_id=None,virtual_signal_file=None,signal_list=None):
    """ 
    Interpret a list of signal names by processing regular expressions and interpreting signals
    using a virtual signal file. The result is a single SignalList class. Possible signal names
    can be listed in <signal_list>. This function will be called recursively until all names found
    are interpreted as a valid signal or a complex signal composed of two valid signals.
    
    A virtual signal file is a text file with format:
        [Virtual names]
        <virtual name>(<start expID> - <end expID) = <description>
        ...
        
        <start expID> and <end expID> set a range of expIDs for which the description applies. 
        This element is optional, if missing the description is valid for all expIDs.
        <virtual name> which can be interpreted.
        <description> can be the following:
            1. A signal name. It can be real signal name, a virtual name, a signal name containing 
               extended regular expressions (see select_signals())
            2. A list of signal names in the form (name1,name3,....)
               Each name can be any of the one in point 1.
            3. A complex signal composed of two signals: complex(signal1,signal2)
               The two signal names should interpret to real signals.
            4. An inverted signal: invert(signal description). Signal description is a single signal or list. 
               After reading data the signals will be inverted.
    Parameters
    ----------
    signals : str
        The signal names to interpret.
    exp_id : anything convertable to int, optional
        The experiment ID. The default is None.
    virtual_signal_file : str, optional
        The name of the virtual signal file. The default is None, in which case only regular 
        expressions will be processed.
    signal_list: list, optional
        The list of possible signal names. The default is None, in which case only bracked regular 
        expressions will be processed no *. E.g. CH[1-3] is valid, but CH* is valid only if the signal_list is present.

    Raises
    ------
    TypeError
        exp_id should be sting or int.
    ValueError
        Various virtual signal file format problems
    ValueErrorr
        DESCRIPTION.

    Returns
    -------
    list
        A SignalList.

    """

    def convert_exp_id(exp_id):
        """ Convert an exp_id to int or float """
        try:
            return int(exp_id)
        except ValueError:
            pass
        try:
            return float(exp_id)
        except ValueError:
            return None

    if ((exp_id is not None) and (type(exp_id) is not str) and (type(exp_id) is not int)):
        raise TypeError("exp_id should be a string or int.")
    if (type(signals) is not list):
        _signals = [signals]
    else:
        _signals = signals

    # Handling regular expressions, creating a signal list after reguler expression processing
    proc_signals = []
    for i in range(len(_signals)):     
        reg_signals, signal_index = select_signals(signal_list, _signals[i],no_check=True)
        proc_signals += reg_signals
    if (len(proc_signals) == 0):
        return SignalList()
    
    if (virtual_signal_file is not None):    
        # Reading the virtual signal file                
        config = configparser.ConfigParser()
        config.optionxform = str
        read_ok = config.read(virtual_signal_file)
        if (read_ok == []):
            raise ValueError("Cannot open virtual signal file "+virtual_signal_file) 
        try:
            entries = config.items('Virtual names')
        except Exception as e:
            print(str(e))
            raise ValueError("Invalid virtual signal file "+virtual_signal_file)
        entry_descr = [e[0] for e in entries]
        entry_values = [e[1] for e in entries]
 
        slist = SignalList()   
        for i_proc,sig in enumerate(proc_signals):
            # These will contain the selected entry names and values for this signal
            values = []
            entry_names = []
            for e,ev in zip(entry_descr, entry_values):
                # Searching for an entry for this name
                start_brace = e.find('(')
                stop_brace = e.find(')')
                if (start_brace * stop_brace < 0):
                    raise ValueError("Invalid entry '{:s}' in virtual signal file: {:s}".format(e, virtual_signal_file))
                if ((start_brace > 0) and (sig == e[:start_brace]) or ((start_brace < 0) and (sig == e))):
                    if ((exp_id is None) and (start_brace > 0)):
                        raise ValueError("Signal entry '{:s}' in virtual name file: {:s} has expID range, but expID is not given.".format(e, virtual_signal_file))                                 
                    if (start_brace > 0):
                        # ExpID range given
                        exp_id_range = e[start_brace+1:stop_brace].split('-')
                        if (len(exp_id_range) != 2):
                            raise ValueError("Invalid exp_id range in entry '{:s}' in virtual name file: {:s}".format(e, virtual_signal_file))
                        if (exp_id_range[0].strip() == ''):
                            exp_id_start = None
                        else:
                            exp_id_start = convert_exp_id(exp_id_range[0])
                            if (exp_id_start is None):
                                 raise ValueError("Invalid exp_id start in entry '{:s}' in virtual name file: {:s}".format(e, virtual_signal_file))
                        if (exp_id_range[1].strip() == ''):
                            exp_id_stop = None
                        else:
                            exp_id_stop = convert_exp_id(exp_id_range[1])
                            if (exp_id_stop is None):
                                raise ValueError("Invalid exp_id stop in entry '{:s}' in virtual name file: {:s}".format(e, virtual_signal_file))
                        exp_id_num = convert_exp_id(exp_id)
                        if (exp_id_num is None):
                            raise ValueError("exp_id is invalid.")
                        if ((exp_id_start is not None) and (exp_id_num < exp_id_start) or \
                            (exp_id_stop is not None) and (exp_id_num > exp_id_stop)) :
                            continue        
                        entry_names.append(e[:start_brace])
                    else:
                        entry_names.append(e)
                    values.append(ev)
            if (len(entry_names) == 0):
                # This was not interpreted in the virtual signal file
                slist.add(SignalList(data_type='real',signal_list=[sig]))
                continue
            if (len(entry_names) != 1):
                raise ValueError("Multiple interpretations for signal name '{:s} in file {:s}".format(sig,virtual_signal_file))
            descr = values[0].strip()
            start_brace = descr.find('(')
            stop_brace = descr.find(')')
            if (start_brace * stop_brace < 0):
                raise ValueError("Invalid value '{:s}' in virtual signal file: {:s}".format(descr, virtual_signal_file))
            if (start_brace < 0):
                # This is a simple name
                sl = interpret_signals(descr,exp_id=exp_id,virtual_signal_file=virtual_signal_file,signal_list=signal_list)
                # Replacing this signal name with its interpretation
                slist.add(sl)
                continue
            if (start_brace >= 0):
                signals_to_interpret = descr[start_brace+1:stop_brace].split(',')
                data_type = descr[:start_brace].lower().strip()
                if (data_type == 'complex'):
                    sl = interpret_signals(signals_to_interpret,exp_id=exp_id,virtual_signal_file=virtual_signal_file,signal_list=signal_list)
                    if ((len(sl.signal_list) != 2) or (sl.data_type != 'real')):
                        raise ValueError("Invalid complex signal description for signal '{:s}' in file {:s}".format(sig,virtual_signal_file))
                    sl.data_type = 'complex'
                    sl.signal_list = [sl.signal_list]
                    slist.add(sl)
                if ((data_type == 'invert') or (data_type == '')):
                    # This is a list of names
                    sl = interpret_signals(signals_to_interpret,exp_id=exp_id,virtual_signal_file=virtual_signal_file,signal_list=signal_list)
                    if (data_type == 'invert'):
                        sl.invert = [True] * len(sl.signal_list)
                    slist.add(sl)
                    continue
                else:
                    raise ValueError("Invalid signal description for signal '{:s}' in file {:s}".format(sig,virtual_signal_file))
    else:
        slist =  SignalList(data_type='real',signal_list=proc_signals)                
    return slist        


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
    mx_dim = len(mx_shape)
    for i in range(mx_dim):
        # Creating a matrix with 1 element in each direction and the
        # number of elements in index[i] in the i-th dimension
        shape = [1] * mx_dim
        shape[i] = index[i].size
        # Creating this matrix
        ind = np.zeros(tuple(shape),dtype=int)
        # Creating a list of indices with 0 at all places except at i where '...'
        ind_ind = [0] * mx_dim
        ind_ind[i] = ...
        ind[tuple(ind_ind)] = index[i]
        # Expanding this matrix in all other dimensions
        for j in range(mx_dim):
            if (j != i):
                ind = np.repeat(ind,index[j].size,j)
        index_arrays.append(ind)

#    for i in range(len(mx_shape)):                                 #THIS IS A SOLUTION FOR LARGE MATRICES, BUT NOT COMMITED
#        index_arrays.append(slice(min(index[i]),max(index[i])+1))  #DUE TO BEING UNTESTED. NEEDS TO BE UNCOMMENTED IF ONE WANTS TO USE IT
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
        value_translation=[[1,1e3,1e6,1e9,1e12],
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

def spatial_unit_translation(spatial_unit=None):
    _spatial_unit=spatial_unit.lower()
    translation={'meters':1,
                 'meter':1,
                 'm':1,
                 'millimeters':1e-3,
                 'millimeter':1e-3,
                 'mm':1e-3,
                 'micrometers':1e-6,
                 'micrometer':1e-6,
                 'um':1e-6,
                 'nanometers':1e-9,
                 'nanometer':1e-9,
                 'nm':1e-9,
                 'picometers':1e-12,
                 'picometer':1e-12,
                 'pm':1e-12,
                 }
    if (_spatial_unit in translation.keys()):
        return translation[_spatial_unit]
    else:
        print(_spatial_unit+' was not found in the translation. Returning 1.')
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
        if len(original_unit) == 1 or len(original_unit) > 3 : # SI units are longer than 3 if using the full name
            original_unit_translation=1.
        else:
            for keys_short in known_conversions_short:
                if keys_short == original_unit[0]:
                    original_unit_translation=known_conversions_short[keys_short]

    if new_unit_translation is None:
        if len(new_unit) == 1 or len(new_unit) > 3:
            new_unit_translation=1.
        else:
            for keys_short in known_conversions_short:
                if keys_short == new_unit[0]:
                    new_unit_translation=known_conversions_short[keys_short]

    if original_unit_translation is None:
        print('Unit translation cannot be done for the original unit. Returning 1.')
        if VERBOSE:
            if len(original_unit) > 3:
                print('Known conversion units are:')
                print(known_conversions_full)
            else:
                print('Known conversion units are:')
                print(known_conversions_short)
        original_unit_translation=1.

    if new_unit_translation is None:
        print('Unit translation cannot be done for the new unit. Returning 1.')
        if VERBOSE:
            if len(original_unit) > 3:
                print('Known conversion units are:')
                print(known_conversions_full)
            else:
                print('Known conversion units are:')
                print(known_conversions_short)
            new_unit_translation=1.

    return original_unit_translation/new_unit_translation






#import matplotlib.pyplot as plt

#plt.clf()
#ydata, xdata = np.meshgrid(np.arange(10),np.arange(20))
#xdata = xdata.astype(float)
#ydata = ydata.astype(float)
#xdata += ydata*0.1
#ydata += xdata*0.2
#xbox, ybox = grid_to_box(xdata,ydata)
#data =  (xdata + ydata)
#plt.pcolormesh(xbox,ybox,np.transpose(data),cmap='Greys')
#plt.scatter(xdata.flatten(), ydata.flatten())
