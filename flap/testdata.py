# -*- coding: utf-8 -*-
"""
This is a test data source for flap

It is assumed that measurement channels collect temporal data on a 2D mesh in the physical space
The measurement channels are named as TEST-<row>-<colunn>
<row>:  1....15
<column>:  1...10
Sample time is 1 MHz between 0,1 seconds. The signals are sine waves with 1 kHz frequency
The amplitude has Gaussian distribution around the channel matrix center.
Signal unit is Volt, signal amplititude max is 1V, ADC digit is 1 mV

The row/column locations relative to the array corner are:
    xr = (<column>-1)*0.5 [cm]
    yr = 0
    zr = (<row>-1)*0.4 [cm]

Conversion to device coordinates is:
alpha = 15 degree
  x = xr*cos(alpha) - zr*sin(alpha)
  z = xr*sin(alpha) + zr*cos(alpha)
  y = yr

The Device R,Z,phi coordinate system origin in at x,y,z = 0
R is along x
Z is z

Created on Thu Jan 17 10:29:56 2019

@author: Zoletnik
"""

import math
import os.path
import random

import flap
from decimal import *
import numpy as np
import copy

# These are the measurement parameters
ROW_NUMBER = 10
COLUMN_NUMBER = 15
meas_ADC_step = 0.001
alpha = 18. # deg

def testdata_get_data(exp_id=None, data_name='*', no_data=False,
                      options=None, coordinates=None):
    """ Data read function for flap test data source
        Channel names: TEST-col-row
        options:
            'Scaling': 'Volt', 'Digit'
            'Signal' : 'Sin'  Sine signals
                      'Const.' : Constant values with row*COLUMN_NUMBER+column
                      'Complex-Sin': Same as Sine but an imaginary cosine is added
                      'Random': Random (normal dist.) signal in all channels
            'Frequency' : <number> Fixed frequency
                        : [f2,f2]: Changes from channel-to-channel between these frequencies
            'Length': Length in second. The sample rate is 1 MHz
            'Samplerate': Sample rate [Hz]
    """

    default_options = {'Signal': 'Sin',
                       'Scaling': 'Volt',
                       'Frequency': 1E3,
                       'Length': 0.1,
                       'Samplerate':1e6
                       }
    _options = flap.config.merge_options(default_options,options,data_source='TESTDATA')

    meas_timerange = [0, _options['Length']]
    meas_sampletime = 1./_options['Samplerate']
    meas_sample = np.rint(float((meas_timerange[1]-meas_timerange[0])/meas_sampletime))+1
    # creating a list of signal names
    signal_list = []
    for row in range(ROW_NUMBER):
        for column in range(COLUMN_NUMBER):
            signal_list.append('TEST-'+str(column+1)+'-'+str(row+1))

    # Selecting the desired channels
    try:
        signal_select, signal_index = flap.select_signals(signal_list,
                                                          data_name)
    except ValueError as e:
        raise e

    read_range = None
    read_samplerange = None
    # Checking whether time or sample range selection is present
    if (coordinates is not None):
        if (type(coordinates) is not list):
            _coordinates = [coordinates]
        else:
            _coordinates = coordinates
        for coord in _coordinates:
            if (type(coord) is not flap.Coordinate):
                raise TypeError("Coordinate description should be flap.Coordinate.")
            if (coord.unit.name is 'Time'):
                if (coord.mode.equidistant):
                    read_range = [float(coord.c_range[0]),
                                  float(coord.c_range[1])]
                else:
                    raise NotImplementedError("Non-equidistant Time axis is not implemented yet.")
                break
            if coord.unit.name is 'Sample':
                if (coord.mode.equidistant):
                    read_samplerange = coord.c_range
                else:
                    raise NotImplementedError("Non-equidistant Sample axis is not implemented yet.")
                break

    # If no sample or time range is given read all data
    if ((read_range is None) and (read_samplerange is None)):
        read_samplerange = [0, meas_sample]

    # Determining the read_samplerange and read_range
    if (read_range is not None):
        read_range = np.array(read_range)
    if (read_samplerange is None):
        read_samplerange = np.rint(read_range/float(meas_sampletime)).astype(int)
    else:
        read_samplerange = np.array(read_samplerange).astype(int)
    if ((read_samplerange[1] < 0) or (read_samplerange[0] >= meas_sample)):
        raise ValueError("No data in time range.")
    if (read_samplerange[0] < 0):
        read_samplerange[0] = 0
    if (read_samplerange[1] >= meas_sample):
        read_samplerange[1] = meas_sample-1
    if (read_range is None):
        read_range = read_samplerange*meas_sampletime+meas_timerange[0]
    ndata = int(read_samplerange[1]-read_samplerange[0]+1 )

    # Checking whether scaling to Volt is requested
    if (_options['Scaling'] is 'Volt'):
        scale_to_volts = True
        dtype = float
        data_unit = flap.Unit(name='Signal', unit='Volt')
    else:
        scale_to_volts = False
        dtype = np.int16
        data_unit = flap.Unit(name='Signal', unit='Digit')


    # Arranging the signals in 2D
    # These will collect the row and column coordinatees for all signals
    row_list = []
    column_list = []
    # In these we collect how many channels are active in each row and
    # column
    rows_act = np.array([0]*ROW_NUMBER)
    columns_act = np.array([0]*COLUMN_NUMBER)
    for i in range(len(signal_select)):
        s_split = signal_select[i].split('-')
        c = int(s_split[1])
        r = int(s_split[2])
        row_list.append(r)
        column_list.append(c)
        rows_act[r-1] += 1
        columns_act[c-1] += 1

    if (len(signal_select) is 1):
        twodim = False
    else:
        # Determining whether the channel matrix fills a rectangular area
        # indices where rows and columns have channels
        ind_rows_act_n0 = np.where(rows_act != 0)[0]
        ind_columns_act_n0 = np.where(columns_act != 0)[0]
        twodim = ((np.min(rows_act[ind_rows_act_n0])
                   == np.max(rows_act[ind_rows_act_n0]) )
                  and (np.min(columns_act[ind_columns_act_n0])
                       == np.max(columns_act[ind_columns_act_n0])))
        if ((len(ind_rows_act_n0)*len(ind_columns_act_n0) != len(signal_select)) \
            or (len(ind_rows_act_n0) is 1) or (len(ind_columns_act_n0) is 1)):
            twodim = False

    data_arr = None

    if (len(signal_select) is not 1):
        if (twodim):
            row_coords = np.arange(ROW_NUMBER, dtype=int)[ind_rows_act_n0]+1
            column_coords = np.arange(COLUMN_NUMBER, dtype=int)[ind_columns_act_n0]+1
            data_shape = (len(column_coords), len(row_coords), ndata)
#            if (no_data is False):
#                data_arr = np.empty(data_shape, dtype=dtype)
            signal_matrix = np.empty((len(column_coords), len(row_coords)),
                                     dtype=np.dtype(object))
        else:
            data_shape = (len(signal_select), ndata)
 #           if (no_data is False):
 #               data_arr = np.empty(data_shape, dtype=dtype)
    else:
        data_shape = [ndata]
    # Reading the signals

    for i in range(len(signal_select)):
        # Determining row and column numbers from
        r = row_list[i]
        c = column_list[i]
        signal_type = _options['Signal']
        if ((signal_type == 'Sin') or (signal_type == 'Complex-Sin')):
            amp = (r-float(ROW_NUMBER)/2)**2 / (2*(float(ROW_NUMBER)/4)**2) \
                   * (c-float(COLUMN_NUMBER)/2)**2 / (2*(float(COLUMN_NUMBER)/4)**2)
            amp = math.exp(-amp)
            s = np.linspace(float(meas_timerange[0]), float(meas_timerange[1]),
                            num=meas_sample)
            if (type(_options['Frequency']) is list):
                flist = _options['Frequency']
                f = float(flist[1] - flist[0])/(len(signal_select)-1)*i + flist[0]
            else:
                f = float(_options['Frequency'])
            if (signal_type == 'Sin'):
                signal = np.sin(s*2*math.pi*f)*amp
                signal = np.rint(signal/meas_ADC_step)
            if (signal_type == 'Complex-Sin'):
                signal_re = np.sin(s*2*math.pi*f)*amp
                signal_re = np.rint(signal_re/meas_ADC_step)
                signal_im = np.cos(s*2*math.pi*f)*amp
                signal_im = np.rint(signal_im/meas_ADC_step)
                signal = np.empty(signal_re.shape,dtype=complex)
                signal.real = signal_re
                signal.imag = signal_im
        elif (signal_type == 'Const.'):
            signal = np.full(int(meas_sample),float(r-1)*COLUMN_NUMBER + float(c-1))
        elif (signal_type == 'Random'):
            signal = np.random.randn(int(meas_sample)) + 10.
        else:
            raise ValueError("Signal type '"+signal_type+"' is not understood.")

        # The actual signal read. Excepts are not necessary here only if
        # file read is involved.
        if (len(signal_select) is 1):
            try:
                if (no_data is False):
                    data_arr = signal[int(read_samplerange[0]):int(read_samplerange[1])+1]
            except Exception as e:
                raise IOError("Error reading from file.")
        else:
            try:
                if (no_data is False):
                    d = signal[int(read_samplerange[0]):int(read_samplerange[1])+1]
            except Exception as e:
                raise IOError("Error reading from file.")
            if (twodim):
                 if (no_data is False):
                    if (data_arr is None):
                         data_arr = np.empty(data_shape, dtype=d.dtype)
                    data_arr[list(column_coords).index(c),
                             list(row_coords).index(r), :] = d
                 signal_matrix[list(column_coords).index(c),
                              list(row_coords).index(r)] = signal_select[i]
            else:
                if (no_data is False):
                    if (data_arr is None):
                        data_arr = np.empty(data_shape, dtype=d.dtype)
                    data_arr[i, :] = d
    if (scale_to_volts):
        if (no_data is False):
            data_arr = data_arr*meas_ADC_step

    # Adding coordinates: Sample, Time, Signal name, Column, Row
    coord = [None]*5

    # Sample and Time
    c_mode = flap.CoordinateMode(equidistant=True)
    if (twodim):
        dimension_list = [2]
    else:
        if (len(signal_select) is 1):
            dimension_list = [0]
        else:
            dimension_list = [1]
    coord[0] = copy.deepcopy(flap.Coordinate(name='Time',
                                             unit='Second',
                                             mode=c_mode,
                                             shape=ndata,
                                             start=read_range[0],
                                             step=meas_sampletime,
                                             dimension_list=dimension_list)
                                            )
    coord[1] = copy.deepcopy(flap.Coordinate(name='Sample',
                                             unit='n.a.',
                                             mode=c_mode,
                                             shape=ndata,
                                             start=read_samplerange[0],
                                             step=1,
                                             dimension_list=dimension_list))
    # Signal name
    c_mode = flap.CoordinateMode(equidistant=False)
    if (twodim):
        dimension_list = [0, 1]
        value = signal_matrix
        shape = signal_matrix.shape
    else:
        value = signal_select
        if (len(signal_select) is 1):
            dimension_list = []
            shape = []
        else:
            dimension_list = [0]
            shape = len(signal_select)
    coord[2] = copy.deepcopy(flap.Coordinate(name='Signal name',
                                             unit='n.a.',
                                             mode=c_mode,
                                             shape=shape,
                                             values=np.array(value),
                                             dimension_list=dimension_list))

    # Column
    c_mode = flap.CoordinateMode(equidistant=False)
    if (twodim):
        dimension_list = [0]
        values = column_coords
        shape = len(values)
    else:
        values = column_list
        if (len(values) is 1):
            dimension_list = []
            shape = []
        else:
            dimension_list = [0]
            shape = len(values)
    coord[3] = copy.deepcopy(flap.Coordinate(name='Column',
                                             unit='n.a.',
                                             mode=c_mode,
                                             shape=shape,
                                             values=values,
                                             dimension_list=dimension_list))
    # Row
    c_mode = flap.CoordinateMode(equidistant=False)
    if (twodim):
        dimension_list = [1]
        values = row_coords
        shape = len(values)
    else:
        values = row_list
        if (len(values) is 1):
            dimension_list = []
            shape = shape
        else:
            dimension_list = [0]
            shape = len(values)
    coord[4] = copy.deepcopy(flap.Coordinate(name='Row',
                                             unit='n.a.',
                                             mode=c_mode,
                                             shape=shape,
                                             values=values,
                                             dimension_list=dimension_list))

    data_title = "Test data"
    if (len(signal_select) is 1):
        data_title += " ("+signal_select[0]+")"
    d = flap.DataObject(data_array=data_arr, data_unit=data_unit,
                        coordinates=coord, exp_id=exp_id,
                        data_title=data_title, data_shape=data_shape)
    return d


def add_coordinate(data_object,
                   coordinates=None,
                   exp_id=None,
                   options=None):
    """ This is the registered function for adding coordinates to the
        data object.

        handled coordinates:
            Device x, Device z, Device r

    """
    for new_coord in coordinates:
        if (new_coord == 'Device y'):
            # Adding a constant 0 as the z coordinate is always 0
            data_object.add_coordinate_object(copy.deepcopy(
                    flap.Coordinate(name='Device y',
                                    unit='cm',
                                    mode=flap.CoordinateMode(equidistant=False),
                                    shape=[],
                                    values=float(0),
                                    dimension_list=[]))
                                       )
            continue

        if ((new_coord == 'Device x') or (new_coord == 'Device z')):
            # Checking whether the new coordinates have already been calculated
            try:
                Device_x
                Device_z
            except NameError:
                # If not calculating both

                # We need to find on which dimensions the row and column coordinate changes
                # x and y will change on both
                # Getting the flap.Coordinate instances of the Row and Column coordinates
                try:
                    c_row = data_object.get_coordinate_object('Row')
                    c_column = data_object.get_coordinate_object('Column')
                except Exception as e:
                    raise e

                # Creating the dimension list of the new coordinate which will be the combined
                # list of Row and Column
                dimension_list = copy.copy(c_column.dimension_list)
                if (type(dimension_list) is not list):
                    dimension_list = [dimension_list]
                dimlist_row = copy.copy(c_row.dimension_list)
                if (type(dimlist_row) is not list):
                    dimlist_row = [dimlist_row]
                for d in dimlist_row:
                    try:
                        dimension_list.index(d)
                    except (ValueError,IndexError):
                        dimension_list.append(d)
                # Creating an index list where these dimensions are fully indexed, the rest is 0
                index = [0]*len(data_object.shape)
                if (len(dimension_list) is not 0):
                    # If at least one of the coordinates is not scalar
                    dimension_list.sort()
                    for i in dimension_list:
                        index[i] = ...
                # Getting data for this data points
                try:
                    row, row_l, row_h = data_object.coordinate('Row',index=index)
                    column, column_l, column_h = data_object.coordinate('Column',index=index)
                except Exception as e:
                    raise e
                if (len(dimension_list) is not 0):
                    sh = []
                    for i in dimension_list:
                        sh.append(data_object.shape[i])
                    row = np.reshape(row,tuple(sh))
                    column = np.reshape(column,tuple(sh))
                # Calculating the new coordinates
                xr = (column - 1) * 0.5
                zr = (row -1) *0.4
                alpha_rad = alpha/180.*math.pi
                Device_x = xr * math.cos(alpha_rad) - zr * math.sin(alpha_rad)
                Device_z = xr * math.sin(alpha_rad) + zr * math.cos(alpha_rad)
                if (Device_x.size is 1):
                    shape = []
                else:
                    shape = Device_x.shape

            # Adding the new coordinate
            if (new_coord == 'Device x'):
                data = Device_x
            else:
                data = Device_z
            cc = flap.Coordinate(name=new_coord,
                                    unit='cm',
                                    mode=flap.CoordinateMode(equidistant=False),
                                    shape=shape,
                                    values=data,
                                    dimension_list=dimension_list)
            data_object.add_coordinate_object(copy.deepcopy(cc))


def register():
    flap.register_data_source('TESTDATA',
                          get_data_func=testdata_get_data,
                          add_coord_func=add_coordinate)
