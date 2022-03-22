# -*- coding: utf-8 -*-
"""
This is a test data source for flap

It is assumed that measurement channels collect temporal data on a 2D mesh in the physical space
The measurement channels are named as TEST-<row>-<colunn>
<row>:  1....15
<column>:  1...10

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
#import os.path                                                                 #UNUSED
#import random                                                                  #UNUSED

import flap
#from decimal import *
import numpy as np
import copy

# These are the measurement parameters
ROW_NUMBER = 10
COLUMN_NUMBER = 15
meas_ADC_step = 0.001  # ADC resolution in Volt
alpha = 18. # degree angle of measurement matrix

def testdata_get_data(exp_id=None, data_name='*', no_data=False,
                      options=None, coordinates=None, data_source=None):
    """ Data read function for flap test data source
        Channel names: TEST-col-row: Signals on a 15x5 spatial matrix
                       VIDEO: A test image with timescale as for the signals.
                       
        options:
            'Scaling': 'Volt', 'Digit'
            'Signal' :'Sin'  Sine signals
                      'Const.' : Constant values with row*COLUMN_NUMBER+column
                      'Complex-Sin': Same as Sine but an imaginary cosine is added
                      'Random': Random (normal dist.) signal in all channels
            'Row number': Number of rows for signal matrix
            'Coumn number': Number of columns for signal matrix
            'Matrix angle': The angle [deg] of the signal matrix
            'Image' : 'Gauss' Gaussian spot
                      'Random'  Random int16 values between 0 and 4095
            'Spotsize': Full width at half maximum of spot in pixels
            'Width'   : Image width in pixels (x size)
            'Height'  : Image height in pixels (y size)
            'Frequency' : <number> Fixed frequency of signals [Hz]
                        : [f2,f2]: Changes from channel-to-channel between these frequencies [Hz]
                        : data object: should be data object with Time coordinate and Frequency as data
                                       If has one channel describes frequency vs time for all channels
                                       If multiple signals (coordinate Signal name describes signals) describes frequency
                                          vs time for channels. Should have the same number of channels as for the generation.
                                          
            'Length': Length in second. The sample rate is 1 MHz
            'Samplerate': Sample rate [Hz]
    """
    if (data_source is None ):
        data_source = 'TESTDATA'

    default_options = {'Row number': 10,
                       'Column number': 15,
                       'Matrix angle': 0.0,
                       'Scaling': 'Volt',
                       'Signal': 'Sin',
                       'Image':'Gauss',
                       'Spotsize':10,
                       'Width': 640,
                       'Height': 480,
                       'Frequency': 1e3,
                       'Length': 0.1,
                       'Samplerate':1e6
                       }
    _options = flap.config.merge_options(default_options, options, data_source=data_source)

    ROW_NUMBER = _options['Row number']
    COLUMN_NUMBER = _options['Column number']
    alpha = _options['Matrix angle']
    # creating a list of signal names
    signal_list = []
    for row in range(ROW_NUMBER):
        for column in range(COLUMN_NUMBER):
            signal_list.append('TEST-'+str(column+1)+'-'+str(row+1))
    signal_list.append('VIDEO')

    # Selecting the desired channels
    try:
        signal_select, signal_index = flap.select_signals(signal_list,
                                                          data_name)
    except ValueError as e:
        raise e

    try:
        signal_select.index('VIDEO')
        test_image = True
    except:
        test_image = False
    if (test_image and (len(signal_select) != 1)):
            raise ValueError("VIDEO data cannot be read together with test signals.")

    meas_timerange = [0, _options['Length']]
    meas_sampletime = 1./_options['Samplerate']
    meas_sample = int(np.rint(float((meas_timerange[1]-meas_timerange[0])/meas_sampletime))+1)
    
    read_range = None
    read_samplerange = None
    # Checking whether time or sample range selection == present
    if (coordinates is not None):
        if (type(coordinates) != list):
            _coordinates = [coordinates]
        else:
            _coordinates = coordinates
        for coord in _coordinates:
            if (type(coord) != flap.Coordinate):
                raise TypeError("Coordinate description should be flap.Coordinate.")
            if (coord.unit.name == 'Time'):
                if (coord.mode.equidistant):
                    read_range = [float(coord.c_range[0]),
                                  float(coord.c_range[1])]
                else:
                    raise NotImplementedError("Non-equidistant Time axis is not implemented yet.")
                break
            if coord.unit.name == 'Sample':
                if (coord.mode.equidistant):
                    read_samplerange = coord.c_range
                else:
                    raise NotImplementedError("Non-equidistant Sample axis is not implemented yet.")
                break

    # If no sample or time range is given read all data
    if ((read_range is None ) and (read_samplerange is None )):
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
    if (_options['Scaling'] == 'Volt'):
        scale_to_volts = True
        #dtype = float                                                          #UNUSED
        data_unit = flap.Unit(name='Signal', unit='Volt')
    else:
        scale_to_volts = False
        #dtype = np.int16                                                       #UNUSED
        data_unit = flap.Unit(name='Signal', unit='Digit')

    if (not test_image):
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
            if (len(s_split) != 3):
                continue
            c = int(s_split[1])
            r = int(s_split[2])
            row_list.append(r)
            column_list.append(c)
            rows_act[r-1] += 1
            columns_act[c-1] += 1
    
        if (len(signal_select) == 1):
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
                or (len(ind_rows_act_n0) == 1) or (len(ind_columns_act_n0) == 1)):
                twodim = False
    
        data_arr = None
    
        if (len(signal_select) != 1):
            if (twodim):
                row_coords = np.arange(ROW_NUMBER, dtype=int)[ind_rows_act_n0]+1
                column_coords = np.arange(COLUMN_NUMBER, dtype=int)[ind_columns_act_n0]+1
                data_shape = (len(column_coords), len(row_coords), ndata)
    #            if (no_data == False):
    #                data_arr = np.empty(data_shape, dtype=dtype)
                signal_matrix = np.empty((len(column_coords), len(row_coords)),
                                         dtype=np.dtype(object))
            else:
                data_shape = (len(signal_select), ndata)
     #           if (no_data == False):
     #               data_arr = np.empty(data_shape, dtype=dtype)
        else:
            data_shape = [ndata]
        # Reading the signals
    
        for i in range(len(row_list)):
            # Determining row and column numbers
            r = row_list[i]
            c = column_list[i]
            signal_type = _options['Signal']
            if ((signal_type == 'Sin') or (signal_type == 'Complex-Sin')):
                amp = (r-float(ROW_NUMBER)/2)**2 / (2*(float(ROW_NUMBER)/4)**2) \
                       * (c-float(COLUMN_NUMBER)/2)**2 / (2*(float(COLUMN_NUMBER)/4)**2)
                amp = math.exp(-amp)
                n_sample = round(_options['Samplerate'] * _options['Length'])
                # Sample times for all signal 
                t = np.arange(meas_sample) * meas_sampletime
                if (type(_options['Frequency']) == list):
                    flist = _options['Frequency']
                    f = float(flist[1] - flist[0])/(len(signal_select)-1)*i + flist[0]
                    ph = t * 2 * math.pi * f
                elif (type(_options['Frequency']) == flap.DataObject):
                    if (len(_options['Frequency'].shape) == 1):
                        tf,t_low,t_high = _options['Frequency'].coordinate('Time',options={'Change only':True})
                        f = _options['Frequency'].data
                        if ((np.amin(tf) > 0) or (np.amax(tf) < _options['Length'])):
                            raise ValueError('Frequency time evolution is not available for sample range [0,{:f}]s.'.format(_options['Length']))
                        fi = np.interp(t,tf,f)
                        ph = np.cumsum(2 * math.pi * fi /_options['Samplerate'])
                    else:
                        raise NotImplementedError("Variable frequency for multiple channels not supported yet.")
                    
                else:
                    f = float(_options['Frequency'])
                    ph = t * 2 * math.pi * f
                if (signal_type == 'Sin'):
                    signal = np.sin(ph) * amp
                    signal = np.rint(signal/meas_ADC_step)
                if (signal_type == 'Complex-Sin'):
                    signal_re = np.sin(ph)*amp
                    signal_re = np.rint(signal_re/meas_ADC_step)
                    signal_im = np.cos(ph)*amp
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
    
            if (len(signal_select) == 1):
                if (no_data == False):
                    data_arr = signal[int(read_samplerange[0]):int(read_samplerange[1])+1]
            else:
                if (no_data == False):
                    d = signal[int(read_samplerange[0]):int(read_samplerange[1])+1]
                if (twodim):
                     if (no_data == False):
                        if (data_arr is None ):
                             data_arr = np.empty(data_shape, dtype=d.dtype)
                        data_arr[list(column_coords).index(c),
                                 list(row_coords).index(r), :] = d
                     signal_matrix[list(column_coords).index(c),
                                  list(row_coords).index(r)] = signal_select[i]
                else:
                    if (no_data == False):
                        if (data_arr is None ):
                            data_arr = np.empty(data_shape, dtype=d.dtype)
                        data_arr[i, :] = d
        
        if (scale_to_volts):
            if (no_data == False):
                data_arr = data_arr*meas_ADC_step
    
        # Adding coordinates: Sample, Time, Signal name, Column, Row
        coord = [None]*5
    
        # Sample and Time
        c_mode = flap.CoordinateMode(equidistant=True)
        if (twodim):
            dimension_list = [2]
        else:
            if (len(signal_select) == 1):
                dimension_list = [0]
            else:
                dimension_list = [1]
        coord[0] = copy.deepcopy(flap.Coordinate(name='Time',
                                                 unit='Second',
                                                 mode=c_mode,
                                                 shape=[],
                                                 start=read_range[0],
                                                 step=meas_sampletime,
                                                 dimension_list=dimension_list)
                                                )
        coord[1] = copy.deepcopy(flap.Coordinate(name='Sample',
                                                 unit='',
                                                 mode=c_mode,
                                                 shape=[],
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
            if (len(signal_select) == 1):
                dimension_list = []
                shape = []
            else:
                dimension_list = [0]
                shape = len(signal_select)
        coord[2] = copy.deepcopy(flap.Coordinate(name='Signal name',
                                                 unit='',
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
            if (len(values) == 1):
                dimension_list = []
                shape = []
            else:
                dimension_list = [0]
                shape = len(values)
        coord[3] = copy.deepcopy(flap.Coordinate(name='Column',
                                                 unit='',
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
            if (len(values) == 1):
                dimension_list = []
                shape = shape
            else:
                dimension_list = [0]
                shape = len(values)
        coord[4] = copy.deepcopy(flap.Coordinate(name='Row',
                                                 unit='',
                                                 mode=c_mode,
                                                 shape=shape,
                                                 values=values,
                                                 dimension_list=dimension_list))
    
        data_title = "Test data"
        if (len(signal_select) == 1):
            data_title += " ("+signal_select[0]+")"
        d = flap.DataObject(data_array=data_arr, data_unit=data_unit,
                            coordinates=coord, exp_id=exp_id,
                            data_title=data_title, data_shape=data_shape)
    
    else:
        # VIDEO
        image_xsize = _options['Width']
        image_ysize = _options['Height']
        spotwidth = _options['Spotsize']
        if (no_data == False):
            if (_options['Image'] == 'Gauss'):
                f = float(_options['Frequency'])
                t = np.arange(ndata,dtype=float) * meas_sampletime
                amp = np.sin(t * 4.5 * math.pi * f) ** 2 * 3000 + 1000
                center_x = image_xsize/2 + (np.sin(t*2*math.pi*f) * image_xsize / 4) * (t / t[-1] + 0.2)
                center_y = image_ysize/2 + (np.cos(t*2*math.pi*f) * image_ysize / 4) * (t / t[-1] + 0.2)
                x,y = np.meshgrid(range(0,image_xsize), range(0,image_ysize))
                data_arr = np.empty((x.shape[0], x.shape[1], t.size),dtype=np.int16)
                for it in range(len(t)):
                    data_arr[:,:,it] = (np.exp(-((x - center_x[it]) ** 2 + (y - center_y[it]) ** 2)
                                       /2/(spotwidth ** 2 + spotwidth  ** 2)) * amp[it]).astype(np.int16)
            elif(_options['Image'] == 'Random'):
                data_arr = np.random.randint(0,high=4095,size=(image_xsize,image_ysize,ndata),dtype=np.int16)                
        coord = [None]*4
        coord[0] = copy.deepcopy(flap.Coordinate(name='Time',
                                                 unit='Second',
                                                 mode=flap.CoordinateMode(equidistant=True),
                                                 shape=[],
                                                 start=read_range[0],
                                                 step=meas_sampletime,
                                                 dimension_list=[2])
                                                )
        coord[1] = copy.deepcopy(flap.Coordinate(name='Sample',
                                                 unit='',
                                                 mode=flap.CoordinateMode(equidistant=True),
                                                 shape=[],
                                                 start=read_samplerange[0],
                                                 step=1,
                                                 dimension_list=[2]))
        coord[2] = copy.deepcopy(flap.Coordinate(name='Image x',
                                                 unit='Pixel',
                                                 mode=flap.CoordinateMode(equidistant=True),
                                                 shape=[],
                                                 start=0,
                                                 step=1,
                                                 dimension_list=[1]))
        coord[3] = copy.deepcopy(flap.Coordinate(name='Image y',
                                                 unit='Pixel',
                                                 mode=flap.CoordinateMode(equidistant=True),
                                                 shape=[],
                                                 start=0,
                                                 step=1,
                                                 dimension_list=[0]))
        data_title = "Test images"
        d = flap.DataObject(data_array=data_arr, data_unit=flap.Unit(name='Image',unit='Digit'),
                            coordinates=coord, exp_id=exp_id,
                            data_title=data_title)

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
                if (type(dimension_list) != list):
                    dimension_list = [dimension_list]
                dimlist_row = copy.copy(c_row.dimension_list)
                if (type(dimlist_row) != list):
                    dimlist_row = [dimlist_row]
                for d in dimlist_row:
                    try:
                        dimension_list.index(d)
                    except (ValueError,IndexError):
                        dimension_list.append(d)
                # Creating an index list where these dimensions are fully indexed, the rest is 0
                index = [0]*len(data_object.shape)
                if (len(dimension_list) != 0):
                    # If at least one of the coordinates != scalar
                    dimension_list.sort()
                    for i in dimension_list:
                        index[i] = ...
                # Getting data for this data points
                try:
                    row, row_l, row_h = data_object.coordinate('Row',index=index)
                    column, column_l, column_h = data_object.coordinate('Column',index=index)
                except Exception as e:
                    raise e
                if (len(dimension_list) != 0):
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
                if (Device_x.size == 1):
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


def register(data_source=None):
    if (data_source is None ):
        data_source = 'TESTDATA'
    flap.register_data_source('TESTDATA',
                          get_data_func=testdata_get_data,
                          add_coord_func=add_coordinate)
