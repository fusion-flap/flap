# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:37:06 2019

@author: Zoletnik
"""
""" Notes on how to use subplots.
    plt.gca() returns the current subplot
    to create sub-sub plots use
        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0)
    Then create the figures (axes):
        ax1 = plt.subplot(gs[0, :])
    Use ax1.plot and other calls to plot onto the figure.
    
    Note on FLAP plotting mechanism.
    - The axes paramater of plot() can be the following:
        string: Name of a coordinate or __Data__. This latter means the data in the data object is on the axis.
        float value: Constant
        data object: Enables plotting one data object as a function of another
    - When a plot is made parameters are recorded in PlotID. This later enables overplotting
      using the same axes. When the axis are incompatble in overplotting they can be forced. As 
      a link to the data objects involved in the plot are also recorded it is also possible
      to regenerate the plot when megnified and 'All points' is False. 
    - If axis are forced or no axis name and unit is avalilable (like in constant) overplotting is possible
      without forcing. If forced the name and unit of the incompatible axes will be set to ''.
    - Constants are considered as unit-less, therefore they don't need forcing.
    
"""    

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.animation as animation
import numpy as np
import copy
from enum import Enum
import math
import time

import flap.config
from .coordinate import *
from .tools import *

global act_plot_id, gca_invalid

class PddType(Enum):
    Coordinate = 0
    Constant = 1
    Data = 2
    
class PlotDataDescription:
    """ Plot axis description for use in PlotID() and plot(). 
        data_object: The data object from which the data for this coordinate originates. This may
                     be None if the data is constant.
        data_type: PddType
                   PddType.Coordinate: A coordinate in data_object. 
                                 self.value is a flap.Coordinate object. 
                   PddType.Constant: A float constant, stored in value.
                   PddType.Data: The data in self.data_object. 
        value: Value, see above
    """
    def __init__(self, data_type=None, data_object=None, value=None):
        self.data_type = data_type
        self.data_object = data_object
        self.value = value
        
def axes_to_pdd_list(d,axes):
    """ Convert a plot() axes parameter to a list of PlotAxisDescription and axes list for PlotID
    d: data object
    axes: axes parameter of plot()
    
    return pdd_list, ax_list
    """
    if (axes is None):
        return [],[]
    if (type(axes) is not list):
        _axes = [axes]
    else:
        _axes = axes
    
    pdd_list = []
    ax_list = []
    for ax in _axes:
        if (type(ax) is str):
            if (ax != '__Data__'):
                try:
                    coord = d.get_coordinate_object(ax)
                except ValueError as e:
                    raise e
                pdd = PlotDataDescription(data_type=PddType.Coordinate,
                                          data_object=d,
                                          value=coord
                                          )
                axx = coord.unit
            else:
                if (d.data is None):
                    raise ValueError("No data available for plotting.")
                pdd = PlotDataDescription(data_type=PddType.Data,
                                          data_object=d
                                          )
                axx = d.data_unit
        elif (type(ax) is type(d)):
            if (ax.data is None):
                raise ValueError("No data available for axis.")
            pdd = PlotDataDescription(data_type=PddType.Data,
                                      data_object=ax
                                      )  
            axx = ax.data_unit
        else:
            try:
                val = float(ax)
            except ValueError:
                raise ValueError("Invalid axis description.")
            pdd = PlotDataDescription(data_type=PddType.Constant,
                                      value=val
                                      )
            axx = Unit()
        pdd_list.append(pdd)
        ax_list.append(axx)
    return pdd_list,ax_list

class PlotID:
    def __init__(self):
        # The figure number where the plto resides
        self.figure = None
        # The subplot containing the whole plot
        self.base_subplot = None
        # The plot type string 
        self.plot_type = None
        # Subtype is dependent on the plot type. It marks various versions, e.g. real-complex
        self.plot_subtype = None
        # Number of plot calls which generated this
        self.number_of_plots = 0
        # The axes list. Each element is a flap.Unit and describes one axis of the plot.
        self.axes = None
        # The description of the axis data. This is a list of self.number_of_plots lists. Each inner list 
        # is a list of PlotDataDescriptions. 
        self.plot_data = []
        # The list of Axes which can be used for plotting into the individual plots. If there is only a single
        # plot self.base_subplot is the same as plt_axis_list[0]
        self.plt_axis_list = None
        # These are a list of the options to plot
        self.options = []

    def clear(self):
        """ Clears the parameters of the plot, but does not clear the
        base_subplot and figure
        """
        self.plot_type = None
        self.plot_subtype = None
        self.number_of_plots = 0
        self.axes = None
        self.plot_data = []
        self.plt_axis_list = None
        self.options = []
        

#    def __del__(self):
#        if (self.plt_axis_list is not None):
#            for ax in self.plt_axis_list:
#                ax.remove()
                
    def check_axes(self,d, axes, clear=True, default_axes=None, force=False):
        """ Checks whether the required plot axes are correct, present and compatible with the self PlotID. 
            In case of problems raises ValueError.
        INPUT:
            d: data object
            axes: List of the required axes or None as input to plot() 
            clear: (bool) If True the plot will be cleared, therefore the axes in the PlotID are irrelevant.
            default_axes: The default axes desired for this type of plot. (strings)
            force: (bool) Force accept incompatibe axes
        Return value:
            pdd_list, ax_list
            pdd_list: list of PlotDataDescription objects which can be used to generate the plot.
            ax_list: axis list which can be put into axes in self.  
        """
        if (axes is not None):
            if (type(axes) is not list):
                _axes = [axes]
            else:
                _axes = axes
        else:
            _axes = []
        
        # Converting to PDD and axes list
        try:
            pdd_list,ax_list = axes_to_pdd_list(d,_axes)
        except ValueError as e:
            raise e
        # Adding default axes if not all are specified   
        if (len(_axes) < len(default_axes)):    
            # Determining default axes for the ones which are not in _axes:
            #  either the ones in the plot or defaults
            if (not clear and (self.axes is not None)):
                # Using the axis in the plot
                plot_axes = [''] * (len(default_axes) - len(_axes))
                for i,ax in enumerate(self.axes[len(_axes):len(default_axes)]):
                    # If axis in plot has no name then using default axis else the plot axis
                    if (ax.name == ''):
                        plot_axes[i] = default_axes[i]
                    else:
                        if (ax.name == d.data_unit.name):
                            plot_axes[i] = '__Data__'
                        else:
                            plot_axes[i] = ax.name
                try:            
                    pdd_def,plot_def_axes = axes_to_pdd_list(d,plot_axes)
                except ValueError as e:
                    for axname in plot_axes:
                        if (axname in str(e)):
                            raise ValueError("Plot coordinate '"+axname+"' not found neither as coordinate nor data name. Must specifiy axes and use option={'Force axes':True} to overplot.")
            else:
                # If overplotting or no plot is available then using the default
                try:
                    pdd_def, plot_def_axes = axes_to_pdd_list(d,default_axes[len(_axes):])   
                except ValueError as e:
                    raise e
            ax_list += plot_def_axes
            pdd_list += pdd_def
                     
                    
        ax_out_list = []
        # Checking whether the units of this plot are consistent with the present axes                            
        if (not clear and (self.axes is not None)):
            for ax_in,ax_plot in zip(ax_list, self.axes):
                # If none of the plot and the new axis has name and ot unit, the new axis will not have it either
#                if ((ax_plot.name == '') or (ax_plot.unit == '') or 
#                    (ax_in.name == '') or (ax_in.unit == '')):
#                    ax_out_list.append(Unit())
                if ((ax_in.name != ax_plot.name) or (ax_in.unit != ax_plot.unit)):
                    if (force):
                        u = Unit()
                        if (ax_in.name == ax_plot.name):
                            u.name = ax_in.name
                        elif ((ax_in.name is None) or (ax_in.name is '')):
                            u.name = ax_plot.name 
                        elif ((ax_plot.name is None) or (ax_plot.name is '')):
                            u.name = ax_in.name    
                        if (ax_in.unit == ax_plot.unit):
                            u.unit = ax_in.unit
                        elif ((ax_in.unit is None) or (ax_in.unit is '')):
                            u.unit = ax_plot.unit 
                        elif ((ax_plot.unit is None) or (ax_plot.unit is '')):
                            u.unit = ax_in.unit        
                        ax_out_list.append(u)
                        continue
                    raise ValueError("Axis '"+ax_in.name+"' is incompatible with plot. Use option={'Force axes':True}")
                else:
                    ax_out_list.append(ax_in)
        else:
            ax_out_list = ax_list
        return pdd_list, ax_out_list
    
def set_plot_id(plot_id):
    """
    Set the current plot.
    """
    global act_plot_id, gca_invalid

    if ((plot_id is not None) and (type(plot_id) is not PlotID)):
        raise ValueError("flap.set_plot_id should receive a valid flap PlotID or None")
    act_plot_id = plot_id
    if (act_plot_id is None):
        gca_invalid = False
    else:
        if (plot_id.plt_axis_list is None):
            gca_invalid = False
        else:
            if (plt.gca() != plot_id.plt_axis_list[-1]):
                gca_invalid = True
            else:
                gca_invalid = False
            
def get_plot_id():
    """
    Return the current PlotID or None if no act plot.
    """
    global act_plot_id, gca_invalid
    try:
        return act_plot_id
    except NameError:
        gca_invalid = False
        return None
    
def __get_gca_invalid():
    """ This returns the global gca_invalid flag.
        This flag is set invalid when set_plot changes the default plot and this way
        the Matplotlib current axis (get_gca()) becomes invalid for this plot.
    """
    try:
        gca_invalid
    except NameError:
        gca_invalid = False
    return gca_invalid
        
   
def sample_for_plot(x,y,x_error,y_error,n_points):
    """
    Resamples the y(x) function to np points for plotting.
    This is useful for plotting large arrays in a way that short outlier pulses
    are still indicated.
    The original function is divided into np equal size blocks in x and in each block
    the minimum and maximum is determined. The output will contain 2*np number
    or points. Each consecutive point pair contains the minimum and maximum in
    a block, the time is the centre time of the block.

    x: Input x array.
    y: input y array.
    np: Desired number of blocks. This would be a number larger than the number
        of pixels in the plot in the horizontal direction.

    Return values:
        x_out, y_out
        If the box length is less than 5 the original data will be returned.
    """

    binlen = int(len(x) / n_points)
    if (binlen < 5):
        return x,y,x_error,y_error
    nx = int(len(x) / binlen)
    _y = y[0:nx * binlen]
    _y = np.reshape(_y, (nx,binlen))
    y_bin = np.empty(nx * 4, dtype=type(y))
    ind = np.arange(nx) * 4
    y_bin[ind + 1] = np.amin(_y, 1)
    y_bin[ind + 2] = np.amax(_y, 1)
    y_bin[ind] = np.mean(_y, 1)
    y_bin[ind + 3] = y_bin[ind]
    x_bin = np.empty(nx * 4, dtype=type(x))
    ind1 = np.arange(nx) * binlen + int(binlen / 2)
    x_bin[ind] = x[ind1]
    x_bin[ind + 1] = x[ind1]
    x_bin[ind + 2] = x[ind1]
    x_bin[ind + 3] = x[ind1]

    if ((x_error is not None) or (y_error is not None)):
        ind = np.arange(nx,dtype=np.int32) * binlen + binlen // 2
    if (x_error is not None):
        if (x_error.ndim == 1):
            x_error_bin = np.empty(x_bin.size,dtype=x_error.dtype)
            for i in range(4):
                x_error_bin[slice(i, nx * 4 + i, 4)] = x_error[ind]
        else:
            x_error_bin = np.empty((2,x_bin.size),dtype=x_error.dtype)
            for i in range(4):
                x_error_bin[0,slice(i, nx * 4 + i, 4)] = x_error[0,ind]
                x_error_bin[1,slice(i, nx * 4 + i, 4)] = x_error[1,ind]
    else:
        x_error_bin = None

    if (y_error is not None):
        if (y_error.ndim == 1):
            y_error_bin = np.empty(y_bin.size,dtype=y_error.dtype)
            for i in range(4):
                y_error_bin[slice(i, nx * 4 + i, 4)] = y_error[ind]
        else:
            y_error_bin = np.empty((2,x_bin.size),dtype=y_error.dtype)
            for i in range(4):
                y_error_bin[0,slice(i, nx * 4 + i, 4)] = y_error[0,ind]
                y_error_bin[1,slice(i, nx * 4 + i, 4)] = y_error[1,ind]
    else:
        y_error_bin = None

    return x_bin, y_bin, x_error_bin, y_error_bin

def _plot(data_object, 
          axes=None,
          slicing=None,
          summing=None,
          slicing_options=None,
          options=None,
          plot_type=None,
          plot_options={},
          plot_id=None):
    """
    Plot a data object.
    axes: A list of coordinate names (strings). They should be one of the coordinate
          names in the data object or 'Data'
          They describe the axes of the plot.
          If the number of axes is less than the number required for the plot, '__Data__' will be added.
          If no axes are given default will be used depending on the plot type. E.g. for
          x-y plot the default first axis is the firs coordinate, the second axis is '__Data__'
    slicing, summing: arguments for slice_data. Slicing will be applied before plotting.
    slicing_options: options for slicing. See slice_data()
    plot_type: The plot type (string). Can be abbreviated.
          'xy': Simple 1D plot. Default axes are: first coordinate, Data. For complex signals this
                produces two plots, see option "Complex mode'.
          'multi xy': In case of 2D data plots 1D curves with vertical shift
                      Default x axis is first coordinate, y axis is Data
                      The signals are named in the label with the 'Signal name' coordinate or the one
                      named in options['Signal name']
          'image': Plots a 2D data matrix as an image. Options: Colormap, Data range, ...
          'contour': Contour plot
    plot_options: Dictionary or list of dictionories. Will be passed over to the plot call. For plots with multiple subplots this can be
                  a list of dictionaries, each for one subplot.
    plot_id: A PlotID object is the plot should go into an existing plot.
    options:
        'Error'      True: Plot all error bars (default: True)
                     False: Do not plot errors
                     number > 0: Plot this many error bars in plot
        'Y separation' Vertical separation of curves in multi xy plot. For linear scale this will
                       be added to consecutive cureves. For Log scale consecutive curves will be
                       multiplied by this.
        'Log x' : Logscale X axis
        'Log y' : Logscale Y axis
        'All points' True or False
                     Default is False. If True will plot all points otherwise will plot only a reduced 
                     number of points (see maxpoints). Each plotted point will be the mean of data
                     in a box and a vertical bar will indicate the value range in each box. 
        'Maxpoints': The maximum number of data points plotted. Above this only this many points will 
                     be plotted if "All points" is False.
        'Complex mode': 'Amp-phase': Plot amplitude and phase 
                        'Real-imag': Plot real and imaginary part
        'X range','Y range': Axes ranges. (List of two numbers.)
        'Z range': Range of the vertical axis. (List of two numbers.)
        'Colormap': Cmap name for image and contour plots.
        'Levels': Number of contour levels or array of levels.
        'Aspect ratio': 'equal', 'auto' or float. (See imshow)
        'Waittime' : Time to wait [Seconds] between two images in anim-... type plots
        'Video file': Name of output video file for anim-... plots
        'Video framerate':  Frame rate for output video.
        'Clear': Boolean. If True don't use the existing plots, generate new. (No overplotting.)
        'Force axes': Force overplotting even if axes are incpomatible
        'Colorbar': Boolelan. Switch on/off colorbar
        'Nan color': The color to use in image data plotting for np.nan (not-a-number) values
        'Interpolation': Interpolation method for image plot.
        'Language': Language of certain standard elements in the plot. ('EN', 'HU')
        'animation.ffmpeg_path': Path for the ffmpeg program (with program name) for video encoding.
                                 This can also be set in plt.rcParams['animation.ffmpeg_path']
    """
    
    default_options = {'All points': False, 'Error':True, 'Y separation': None,
                       'Log x': False, 'Log y': False, 'Log z': False, 'maxpoints':4000, 'Complex mode':'Amp-phase',
                       'X range':None, 'Y range': None, 'Z range': None,'Aspect ratio':'auto',
                       'Clear':False,'Force axes':False,'Language':'EN','Maxpoints': 4000,
                       'Levels': 10, 'Colormap':None, 'Waittime':1,'Colorbar':True,'Nan color':None,
                       'Interpolation':'bilinear','Video file':None, 'Video framerate': 20,
                       'animation.ffmpeg_path':None}
    _options = flap.config.merge_options(default_options, options, data_source=data_object.data_source, section='Plot')

    if (plot_options is None):
        _plot_options = {}
    else:
        _plot_options = plot_options
    if (type(_plot_options) is not list):
        _plot_options = [_plot_options]


    if (type(_options['Clear']) is not bool):
        raise TypeError("Invalid type for option Clear. Should be boolean.")
    if (type(_options['Force axes']) is not bool):
        raise TypeError("Invalid type for option 'Force axes'. Should be boolean.")    

    if ((slicing is not None) or (summing is not None)):
        d = data_object.slice_data(slicing=slicing, summing=summing, options=slicing_options)
    else:
        d = data_object        
    
    # Determining a PlotID:
    # argument, actual or a new one
    if (type(plot_id) is PlotID):
        _plot_id = plot_id
    else:
        _plot_id = get_plot_id()
        if (_plot_id is None): 
            # If there is no actual plot we create a new one 
            _plot_id = PlotID()
            _plot_id.figure = plt.gcf().number
            _plot_id.base_subplot = plt.gca()
        if (_plot_id.plt_axis_list is not None):    
            if ((_plot_id.plt_axis_list[-1] != plt.gca()) or (_plot_id.figure != plt.gcf().number)):
                # If the actual subplot is not the one in the plot ID then either the subplot was
                # changed to a new one or the plot_ID changed with set_plot. 
                if (not __get_gca_invalid()):
                    # This means the plot ID was not changed, the actual plot or axis was changed.
                    # Therefore we need to use the actual values         
                    _plot_id = PlotID()
                    _plot_id.figure = plt.gcf().number
                    _plot_id.base_subplot = plt.gca()  
                    plt.cla()
    if (_options['Clear'] ):
        _plot_id = PlotID()
#        _plot_id.clear()
        if (_plot_id.figure is not None):
            plt.figure(_plot_id.figure)
        else:
            _plot_id.figure = plt.gcf().number
        if (_plot_id.base_subplot is not None):
            plt.subplot(_plot_id.base_subplot)
        else:
            _plot_id.base_subplot = plt.gca()          
        plt.cla()
            
    # Setting plot type
    known_plot_types = ['xy','scatter','multi xy', 'image', 'anim-image','contour','anim-contour']
    if (plot_type is None):
        if (len(d.shape) == 1):
            _plot_type = 'xy'
        elif (len(d.shape) == 2):
            _plot_type = 'multi xy'
        elif (len(d.shape) == 3):
            _plot_type = 'anim-image'
        else:
            raise ValueError("No default plot type for this kind of data, set plot_type.")
    else:
        try:
            _plot_type = find_str_match(plot_type,known_plot_types)
        except TypeError:
            raise TypeError("Invalid type for plot_type. String is expected.")
        except ValueError:
            raise ValueError("Unknown plot type or too short abbreviation")    

    # Processing some options
    if ((_plot_type == 'xy') or (_plot_type == 'multi xy')):
        all_points = _options['All points']
        if (type(all_points) is not bool):
            raise TypeError("Option 'All points' should be boolean.")
    else:
        all_points = True
    plot_error = _options['Error']
    if (type(plot_error) is not bool):
        try:
            if (int(plot_error) <= 0):
                raise ValueError("Invalid number of error bars in plot (option: Error).")
            errorbars = int(plot_error)
            plot_error = True
        except:
            raise ValueError("Invalid 'Error' option.")
    else:
        errorbars = -1
    # The maximum number of points expected in the horizontal dimension of a plot
    try:
        maxpoints = int(_options['Maxpoints'])
    except ValueError:
        raise ValueError("Invalid maxpoints setting.")

    try:
        compt = find_str_match(_options['Complex mode'], ['Amp-phase','Real-imag'])
    except:
        raise ValueError("Invalid 'Complex mode' option:" +_options['Complex mode'])
    if (compt == 'Amp-phase'):
        comptype = 0
    elif (compt == 'Real-imag'):    
        comptype = 1
    if (_plot_id.number_of_plots > 0):
        if (_plot_id.options[-1]['Complex mode'] != _options['Complex mode']):
            raise ValueError("Different complex plot mode in overplot.")

    language = _options['Language']
       
    # X range and Z range is processed here, but Y range not as it might have multiple entries for some plots
    xrange = _options['X range']
    if (xrange is not None):
        if ((type(xrange) is not list) or (len(xrange) != 2)):
            raise ValueError("Invalid X range setting.")
    zrange = _options['Z range']
    if (zrange is not None):
        if ((type(zrange) is not list) or (len(zrange) != 2)):
            raise ValueError("Invalid Z range setting.")
     
    cmap = _options['Colormap']
    if ((cmap is not None) and (type(cmap) is not str)):
        raise ValueError("Colormap should be a string.")
    
    contour_levels = _options['Levels']    
        
    # Here _plot_id is a valid (maybe empty) PlotID

    if ((_plot_type == 'xy') or (_plot_type == 'scatter')):
        # 1D plots: xy, scatter and complex versions
        if (len(d.shape) > 1):
            raise ValueError("xy plot is applicable only to 1D data. Use slicing.")
        # Checking whether oveplotting into the same plot type
        if ((d.data is not None) and (d.data.dtype.kind == 'c')): 
            subtype = 1
        else:
            subtype = 0
        if (not _options['Clear']):
            if ((_plot_id.plot_type is not None) and ((_plot_id.plot_type != 'xy') and (_plot_id.plot_type != 'scatter')) 
                or (_plot_id.plot_subtype is not None) and (_plot_id.plot_subtype != subtype)):
                raise ValueError("Overplotting into different plot type. Use option={'Clear':True} to erase first.")
        # Processing axes
        default_axes = [d.coordinates[0].unit.name, '__Data__']
        try:
            pdd_list, ax_list = _plot_id.check_axes(d, 
                                                    axes, 
                                                    clear=_options['Clear'], 
                                                    default_axes=default_axes, 
                                                    force=_options['Force axes'])
        except ValueError as e:
            raise e
        # Preparing data    
        plotdata = [0]*2
        plotdata_low = [0]*2
        plotdata_high = [0]*2
        ploterror = [0]*2
        for ax_ind in range(2):
            if (pdd_list[ax_ind].data_type == PddType.Data):
                plotdata[ax_ind] = pdd_list[ax_ind].data_object.data.flatten()
                if (plot_error):
                    ploterror[ax_ind] = pdd_list[ax_ind].data_object._plot_error_ranges()
                else:
                    ploterror[ax_ind] = None
            elif (pdd_list[ax_ind].data_type == PddType.Coordinate):
                if (not pdd_list[ax_ind].value.isnumeric()):
                    raise ValueError("Coordinate is not of numeric type, cannot plot.")
                pdata, pdata_low, pdata_high = \
                                    pdd_list[ax_ind].value.data(data_shape=pdd_list[ax_ind].data_object.shape)
                plotdata[ax_ind] = pdata.flatten()
                if (pdata_low is not None):
                    pdata_low = pdata_low.flatten()
                if (pdata_high is not None):
                    pdata_high = pdata_high.flatten()
                if (plot_error):
                    ploterror[ax_ind]  = pdd_list[ax_ind].data_object._plot_coord_ranges(pdd_list[ax_ind].value,
                                                                                         pdata, 
                                                                                         pdata_low, 
                                                                                         pdata_high
                                                                                         )
                else:
                    ploterror[ax_ind] = None
            elif (pdd_list[ax_ind].data_type == PddType.Constant):
                plotdata[ax_ind] = pdd_list[ax_ind].value
                ploterror[ax_ind] = None
            else:
                raise RuntimeError("Internal error, invalid PlotDataDescription.")
        if (_plot_id.number_of_plots != 0):
            _options['Log x'] = _plot_id.options[-1]['Log x']
            _options['Log y'] = _plot_id.options[-1]['Log y']
            

        xdata = plotdata[0]
        xerror = ploterror[0]

        if (subtype == 0):     
            yrange = _options['Y range']
            if (yrange is not None):
                if ((type(yrange) is not list) or (len(yrange) != 2)):
                    raise ValueError("Invalid Y range setting.")

            ax = _plot_id.base_subplot
            _plot_id.plt_axis_list = [ax]
    
            ydata = plotdata[1]
            yerror = ploterror[1]
            # Checking for constants, converting to numpy array
            if (np.isscalar(ydata)):
                if (np.isscalar(xdata)):
                    ydata = np.full(1, ydata)
                else:
                    ydata = np.full(xdata.size, ydata)
                yerror = None
            if (np.isscalar(xdata)):
                if (np.isscalar(ydata)):
                    xdata = np.full(1, xdata)
                else:
                    xdata = np.full(ydata.size, xdata)
                xerror = None
    
            if (all_points is True):
                x = xdata
                y = ydata
                xerr = xerror
                yerr = yerror
            else:
                x,y,xerr,yerr = sample_for_plot(xdata,ydata,xerror,yerror,maxpoints)
            if (errorbars < 0):
                errorevery = 1
            else:
                errorevery = int(round(len(x)/errorbars))
                if (errorevery == 0):
                    errorevery = 1

            _plot_opt = _plot_options[0]
            if (type(_plot_opt) is not dict):
                raise ValueError("Plot options should be a dictionary or list of dictionaries.")
   
            if (_plot_type == 'xy'):                
                if (plot_error):
                    ax.errorbar(x,y,xerr=xerr,yerr=yerr,errorevery=errorevery,**_plot_opt)
                else:
                    ax.plot(x,y,**_plot_opt)
            else:
                if (plot_error):
                    ax.errorbar(x,y,xerr=xerr,yerr=yerr,errorevery=errorevery,fmt='o',**_plot_opt)
                else:
                    ax.scatter(x,y,**_plot_opt)
                    
            ax.set_xlabel(ax_list[0].title(language=language))
            ax.set_ylabel(ax_list[1].title(language=language))
            if (_options['Log x']):
                ax.set_xscale('log')
            if (_options['Log y']):
                ax.set_yscale('log')
            if (xrange is not None):
                ax.set_xlim(xrange[0],xrange[1])
            if (yrange is not None):
                ax.set_ylim(yrange[0],yrange[1])
                
            title = ax.get_title()
            if (title is None):
                title = ''
            if (title[-3:] != '...'):
                newtitle = ''
                if (d.exp_id is not None):
                    newtitle += str(d.exp_id) + ' '
                    if (d.data_title is not None):
                        newtitle += d.data_title
                if (len(newtitle) != 0):
                    if (len(title + newtitle) < 40):
                        if (title != ''):
                            title += ','+newtitle
                        else:
                            title = newtitle
                    else:
                        title += ',...'
            ax.set_title(title)

            _plot_id.plot_subtype = 0 # real xy plot 
            # End of real xy and scatter plot
        else:
            # Complex xy plot
            yrange = _options['Y range']
            if (yrange is not None):
                if ((type(yrange) is not list) or (len(yrange) != 2)):
                    raise ValueError("Invalid Y range setting.")
                if ((type(yrange[0]) is list) and (type(yrange[1]) is list) and
                    ((len(yrange[0]) == 2)) and (len(yrange[1]) == 2)):
                        pass
                else:
                    try:
                        yrange = [float(yrange[0]), float(yrange[1])]
                    except ValueError:
                        raise ValueError("Invalid Y range setting. For complex xy plot either a list or list of two lists can be used.")
                    yrange = [yrange,yrange]

            # Complex xy and scatter plot
            # Creating two sublots if this is a new plot
            if (_plot_id.number_of_plots == 0):
                gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=_plot_id.base_subplot)
                _plot_id.plt_axis_list = []
                _plot_id.plt_axis_list.append(plt.subplot(gs[0,0]))
                _plot_id.plt_axis_list.append(plt.subplot(gs[1,0],sharex=_plot_id.plt_axis_list[0]))
            # Taking the mean of the complex errors
            if (ploterror[1] is not None):
                if (type(ploterror[1]) is list):
                    yerror_abs = (np.abs(ploterror[1][0]) + np.abs(ploterror[1][1]))  / 2
                else:
                    yerror_abs = np.abs(ploterror[1]) 
            else:
                yerror_abs = None
            for i_comp in range(2):            
                ax = _plot_id.plt_axis_list[i_comp]
                if (comptype == 0):
                    # amp-phase
                    if (i_comp == 0):
                        ydata = np.abs(plotdata[1])
                        yerror = yerror_abs
                    else:
                        ydata_abs = np.abs(plotdata[1])
                        ydata = np.angle(plotdata[1])
                        if (yerror_abs is not None):
                            yerror = np.empty(ydata.shape,dtype=float)
                            ind = np.nonzero(ydata_abs <= yerror_abs)[0]
                            if (ind.size > 0):
                                yerror[ind] = math.pi
                            ind = np.nonzero(ydata_abs > yerror_abs)[0]
                            if (ind.size > 0):
                                yerror[ind] = np.arctan2(yerror_abs[ind],ydata_abs[ind])
                        else:
                            yerror = None
                else:
                    # real-imag
                    if (i_comp == 0):
                        ydata = np.real(plotdata[1])
                        yerror = yerror_abs
                    else:    
                        ydata = np.imag(plotdata[1])
                        yerror = yerror_abs

                # Checking for constants, converting to numpy array
                if (np.isscalar(ydata)):
                    if (np.isscalar(xdata)):
                        ydata = np.full(1, ydata)
                    else:
                        ydata = np.full(xdata.size, ydata)
                    if (plot_error):
                        yerror = d._plot_coord_ranges(c, ydata, ydata_low, ydata_high)
                    else:
                        yerror = None
                if (np.isscalar(xdata)):
                    if (np.isscalar(ydata)):
                        xdata = np.full(1, xdata)
                    else:
                        xdata = np.full(ydata.size, xdata)
                    if (plot_error):
                        xerror = d._plot_coord_ranges(c, xdata, xdata_low, xdata_high)
                    else:
                        xerror = None
        
                if (all_points is True):
                    x = xdata
                    y = ydata
                    xerr = xerror
                    yerr = yerror
                else:
                    x,y,xerr,yerr = sample_for_plot(xdata,ydata,xerror,yerror,maxpoints)
                if (errorbars < 0):
                    errorevery = 1
                else:
                    errorevery = int(round(len(x)/errorbars))
                    if (errorevery == 0):
                        errorevery = 1

                _plot_opt = _plot_options[0]
                if (type(_plot_opt) is list):
                    _plot_opt[i_comp]
                if (type(_plot_opt) is not dict):
                    raise ValueError("Plot options should be a dictionary or list of dictionaries.")
       
                if (_plot_type == 'xy'):                
                    if (plot_error):
                        ax.errorbar(x,y,xerr=xerr,yerr=yerr,errorevery=errorevery,**_plot_opt)
                    else:
                        ax.plot(x,y,**_plot_opt)
                else:
                    if (plot_error):
                        ax.errorbar(x,y,xerr=xerr,yerr=yerr,errorevery=errorevery,fmt='o',**_plot_opt)
                    else:
                        ax.scatter(x,y,**_plot_opt)

                # Setting axis labels
                ax.set_xlabel(ax_list[0].title(language=language))
                ax.set_ylabel(ax_list[1].title(language=language,complex_txt=[comptype,i_comp])) 
    
                if (_options['Log x']):
                    ax.set_xscale('log')
                if (_options['Log y']):
                    ax.set_yscale('log')
                if (xrange is not None):
                    ax.set_xlim(xrange[0],xrange[1])
                if (yrange is not None):
                    ax.set_ylim(yrange[i_comp][0],yrange[i_comp][1])
            title = ax.get_title()
            if (title is None):
                title = ''
            if (title[-3:] != '...'):
                newtitle = ''
                if (d.exp_id is not None):
                    newtitle += str(d.exp_id) + ' '
                    if (d.data_title is not None):
                        newtitle += d.data_title
                if (len(newtitle) != 0):
                    if (len(title + newtitle) < 40):
                        if (title != ''):
                            title += ','+newtitle
                        else:
                            title = newtitle
                    else:
                        title += ',...'
            ax.set_title(title)

            _plot_id.plot_subtype = 1
            # End of complex xy and scatter plot
        
    elif (_plot_type == 'multi xy'):
        if (len(d.shape) > 2):
            raise TypeError("multi x-y plot is applicable to 2D data only. Use slicing.")
        if (len(d.shape) != 2):
            raise TypeError("multi x-y plot is applicable to 2D data only.")
        if (d.data.dtype.kind == 'c'):
            raise TypeError("multi x-y plot is applicable only to real data.")
        if ((axes is not None) and (type(axes) is list) and (len(axes) > 1)):
            raise ValueError("For multi xy plot only one axis can be given.")

        yrange = _options['Y range']
        if (yrange is not None):
            if ((type(yrange) is not list) or (len(yrange) != 2)):
                raise ValueError("Invalid Y range setting.")

        # Processing axes
        default_axes = [d.coordinates[0].unit.name, '__Data__']
        try:
            pdd_list, ax_list = _plot_id.check_axes(d, 
                                                    axes, 
                                                    clear=_options['Clear'], 
                                                    default_axes=default_axes, 
                                                    force=_options['Force axes'])
        except ValueError as e:
            raise e
        
        if (pdd_list[0].data_type == PddType.Coordinate):
            coord = pdd_list[0].value
            if (len(coord.dimension_list) != 1):
                raise ValueError("Plotting coordinate for multi xy plot should change only along 1 dimension.")
        axis_index = coord.dimension_list[0]
        if (not ((pdd_list[1].data_type == PddType.Data) and (pdd_list[1].data_object == d))):
            raise ValueError("For multi xy plot only data can be plotted on the y axis.")
            
        # Trying to get signal names
        try:
            signals_coord = d.get_coordinate_object('Signal name')
        except ValueError:
            signals_coord = None
        if((signals_coord is not None) and (len(signals_coord.dimension_list) == 1)\
           and (signals_coord.dimension_list[0] != axis_index)):
               if (axis_index == 0):
                   index = [0,...]
               else:
                   index = [...,0]
               signal_names, l, h = signals_coord.data(data_shape=d.shape,index=index)
               signal_names = signal_names.flatten()
        else:
            signal_names = None

        # Getting x data    
        if (pdd_list[0].data_type == PddType.Data):
            xdata = pdd_list[0].data_object.data.flatten()
            if (plot_error):
                xerror = pdd_list[0].data_object._plot_error_ranges()
            else:
                xerror = None
        elif (pdd_list[0].data_type == PddType.Coordinate):
            if (not pdd_list[0].value.isnumeric()):
                raise ValueError("Coordinate is not of numeric type, cannot plot.")
            index = [0]*2
            index[axis_index] = ...
            xdata, xdata_low, xdata_high = \
                                pdd_list[0].value.data(data_shape=pdd_list[0].data_object.shape,index=index)
            xdata = xdata.flatten()
            if (xdata_low is not None):
                xdata_low  = xdata_low.flatten()
            if (xdata_high is not None):
                xdata_high = xdata_high.flatten()
            if (plot_error):
                xerror  = pdd_list[0].data_object._plot_coord_ranges(pdd_list[0].value,
                                                                     xdata, 
                                                                     xdata_low, 
                                                                     xdata_high
                                                                     )
            else:
                xerror = None
        elif (pdd_list[0].data_type == PddType.Constant):
            xdata = pdd_list[ax_ind].value
            xerror = None
        else:
            raise RuntimeError("Internal error, invalid PlotDataDescription.")

        if (_plot_id.number_of_plots != 0):
            _options['Log x'] = _plot_id.options[-1]['Log x']
            _options['Log y'] = _plot_id.options[-1]['Log y']
            
        ax = _plot_id.base_subplot
        _plot_id.plt_axis_list = [ax]
        
        ysep = _options['Y separation']
        if (ysep == None):
            if (_plot_id.number_of_plots != 0):
                ysep = _plot_id.options[-1]['Y separation']
            else:
                if (_options['Log y']):
                    maxval = np.nanmax(d.data)
                    minval = np.nanmin(d.data)
                    if ((minval <= 0) or (minval == maxval)):
                        ysep = 1
                    else:
                        ysep = math.sqrt(maxval/minval)
                else:
                    ysep = float(np.nanmax(d.data))
        else:
            try:
                ysep = float(ysep)
            except ValueError:
                raise ValueError("Invalid Y separation option.") 
        _options['Y separation'] = ysep
                
        # Determining Y range
        if (axis_index == 0):
            if (_options['Log y']):
                ax_min = np.nanmin(d.data[:,0])
                ax_max = np.nanmax(d.data[:,-1]) * ysep ** (d.data.shape[1]-1)
            else:
                ax_min = np.nanmin(d.data[:,0])
                ax_max = np.nanmax(d.data[:,-1]) + ysep * (d.data.shape[1]-1)
            signal_index = 1
        else:
            if (_options['Log y']):
                ax_min = np.nanmin(d.data[0,:])
                ax_max = np.nanmax(d.data[-1,:]) * ysep ** (d.data.shape[0]-1)
            else:
                ax_min = np.nanmin(d.data[0,:])
                ax_max = np.nanmax(d.data[-1,:]) + ysep * (d.data.shape[0]-1)
            signal_index = 0
        # If overplot then taking min and max of this and previous plots    
        if (_plot_id.number_of_plots != 0):
            ax_min = min(ax.get_ylim()[0], ax_min)
            ax_max = max(ax.get_ylim()[1], ax_max)
        if (ax_max <= ax_min):
            ax_max += 1
        ax.set_ylim(ax_min, ax_max)

        legend = []
        if (signal_names is not None):
            for i in range(d.shape[signal_index]):
                legend.append(signal_names[i])
            ax.legend(legend)
         
        _plot_opt = _plot_options[0]
        for i in range(d.shape[signal_index]):
            index = [...] * 2
            index[signal_index] = i
            yerror = d._plot_error_ranges(index=tuple(index))
            if (_options['Log y']):
                ydata = d.data[tuple(index)].flatten() * (ysep ** i)
                if (yerror is not None):
                    yerror *= (ysep ** i)
            else:
                ydata = d.data[tuple(index)].flatten() + ysep * i
            if (all_points is True):
                x = xdata
                y = ydata
                xerr = xerror
                yerr = yerror
            else:
                x,y,xerr,yerr = sample_for_plot(xdata,ydata,xerror,yerror,maxpoints)
            if (errorbars < 0):
                errorevery = 1
            else:
                errorevery = int(round(len(x)/errorbars))
                if (errorevery < 1):
                    errorevery = 1
            if (plot_error):
                ax.errorbar(x,y,xerr=xerr,yerr=yerr,errorevery=errorevery,**_plot_opt)
            else:
                ax.plot(x,y,**_plot_opt)

        if (xrange is not None):
            ax.set_xlim(xrange[0],xrange[1])
        if (yrange is not None):
            ax.set_ylim(yrange[0],yrange[1])        
        ax.set_xlabel(ax_list[0].title(language=language))
        ax.set_ylabel(ax_list[1].title(language=language))
        if (_options['Log x']):
            ax.set_xscale('log')
        if (_options['Log y']):
            ax.set_yscale('log')
        title = ax.get_title()
        if (title is None):
            title = ''
        if (title[-3:] != '...'):
            newtitle = ''
            if (d.exp_id is not None):
                newtitle += str(d.exp_id) + ' '
                if (d.data_title is not None):
                    newtitle += d.data_title
            if (len(newtitle) != 0):
                if (len(title + newtitle) < 40):
                    if (title != ''):
                        title += ','+newtitle
                    else:
                        title = newtitle
                else:
                    title += ',...'
                ax.set_title(title)        
        _plot_id.plot_subtype = 0 # real multi xy plot 

    elif ((_plot_type == 'image') or (_plot_type == 'contour')):
        if (d.data is None):
            raise ValueError("Cannot plot DataObject without data.")
        if (len(d.shape) != 2):
            raise TypeError("Image/contour plot is applicable to 2D data only. Use slicing.")
        if (d.data.dtype.kind == 'c'):
            raise TypeError("Image/contour plot is applicable only to real data.")
        # Checking for numeric type
        try:
            d.data[0,0] += 1
        except TypeError:
            raise TypeError("Image plot is applicable only to numeric data.")
        
        yrange = _options['Y range']
        if (yrange is not None):
            if ((type(yrange) is not list) or (len(yrange) != 2)):
                raise ValueError("Invalid Y range setting.")

        # Processing axes
        # Although the plot will be cleared the existing plot axes will be considered
        default_axes = [d.coordinates[0].unit.name, d.coordinates[1].unit.name, '__Data__']
        try:
            pdd_list, ax_list = _plot_id.check_axes(d, 
                                                    axes, 
                                                    clear=_options['Clear'], 
                                                    default_axes=default_axes, 
                                                    force=_options['Force axes'])
        except ValueError as e:
            raise e

        # No overplotting is possible for this type of plot, erasing and restarting a Plot_ID
        if (not _options['Clear']):
            plt.subplot(_plot_id.base_subplot)
            plt.cla()
        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=_plot_id.base_subplot)
        _plot_id.plt_axis_list = []
        _plot_id.plt_axis_list.append(plt.subplot(gs[0,0]))
        ax = _plot_id.plt_axis_list[0]
        
        pdd_list[2].data_type = PddType.Data
        pdd_list[2].data_object = d
        if ((pdd_list[0].data_type != PddType.Coordinate) or (pdd_list[1].data_type != PddType.Coordinate)) :
            raise ValueError("X and y coordinates of image plot type should be coordinates.")

        coord_x = pdd_list[0].value
        coord_y = pdd_list[1].value
        if (_plot_type == 'image'):
            if ((coord_x.mode.equidistant) and (len(coord_x.dimension_list) == 1) and
                (coord_y.mode.equidistant) and (len(coord_y.dimension_list) == 1)):
                # This data is image-like with data points on a rectangular array
                image_like = True
            elif ((len(coord_x.dimension_list) == 1) and (len(coord_y.dimension_list) == 1)):
                if (not coord_x.isnumeric()):
                    raise ValueError('Coordinate '+coord_x.unit.name+' is not numeric.')
                if (not coord_y.isnumeric()):
                    raise ValueError('Coordinate '+coord_y.unit.name+' is not numeric.')
                index = [0] * len(d.shape)
                index[coord_x.dimension_list[0]] = ...
                xdata,xdata_low,xdata_high = coord_x.data(data_shape=d.shape,index=index)
                xdata = xdata.flatten()
                dx = xdata[1:] - xdata[:-1]
                index = [0] * len(d.shape)
                index[coord_y.dimension_list[0]] = ...
                ydata,ydata_low,ydata_high = coord_y.data(data_shape=d.shape,index=index)
                ydata = ydata.flatten()
                dy = ydata[1:] - ydata[:-1]
                if ((np.nonzero(np.abs(dx - dx[0]) / math.fabs(dx[0]) > 0.01)[0].size == 0) and
                    (np.nonzero(np.abs(dy - dy[0]) / math.fabs(dy[0]) > 0.01)[0].size == 0)):
                    # Actually the non-equidistant coordinates are equidistant
                    image_like = True
                else:
                    image_like = False
        else:
            image_like = False
        if (image_like):        
            xdata_range = coord_x.data_range(data_shape=d.shape)[0]   
            ydata_range = coord_y.data_range(data_shape=d.shape)[0]   
        else:
            ydata,ydata_low,ydata_high = coord_y.data(data_shape=d.shape)
            xdata,xdata_low,xdata_high = coord_x.data(data_shape=d.shape)
            
        if (zrange is None):
            vmin = np.nanmin(d.data)
            vmax = np.nanmax(d.data)
        else:
            vmin = zrange[0]
            vmax = zrange[1]

        if (vmax <= vmin):
            raise ValueError("Invalid z range.")
            
        if (_options['Log z']):
            if (vmin <= 0):
                raise ValueError("z range[0] cannot be negative or zero for logarithmic scale.")
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            locator = ticker.LogLocator(subs='all')
        else:
            norm = None
            locator = None
                
        if (contour_levels is None):
            contour_levels = 255
            
        _plot_opt = _plot_options[0]

        try:
            cmap_obj = plt.cm.get_cmap(cmap)
            if (_options['Nan color'] is not None):
                cmap_obj.set_bad(_options['Nan color'])
        except ValueError:
            raise ValueError("Invalid color map.")

        if (image_like):
            try: 
                if (coord_x.dimension_list[0] == 0):
                    img = ax.imshow(np.clip(np.transpose(d.data),vmin,vmax),extent=xdata_range + ydata_range,norm=norm,
                                    cmap=cmap_obj,vmin=vmin,aspect=_options['Aspect ratio'],interpolation=_options['Interpolation'],
                                    vmax=vmax,origin='lower',**_plot_opt)
                else:
                    img = ax.imshow(np.clip(d.data,vmin,vmax),extent=xdata_range + ydata_range,norm=norm,
                                    cmap=cmap_obj,vmin=vmin,aspect=_options['Aspect ratio'],interpolation=_options['Interpolation'],
                                    vmax=vmax,origin='lower',**_plot_opt)
                    
            except Exception as e:
                raise e
        else:
            if (_plot_type == 'image'):
                xgrid, ygrid = grid_to_box(xdata,ydata)
                try:
                    img = ax.pcolormesh(xgrid,ygrid,np.clip(np.transpose(d.data),vmin,vmax),norm=norm,cmap=cmap,vmin=vmin,
                                        vmax=vmax,**_plot_opt)
                except Exception as e:
                    raise e
            else:
                try:
                    img = ax.contourf(xdata,ydata,np.clip(d.data,vmin,vmax),contour_levels,norm=norm,
                                      origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,**_plot_opt)
                except Exception as e:
                    raise e
                
        if (_options['Colorbar']):
            cbar = plt.colorbar(img,ax=ax)
            cbar.set_label(d.data_unit.name)
        
        if (xrange is not None):
            ax.set_xlim(xrange[0],xrange[1])
        if (yrange is not None):
            ax.set_ylim(yrange[0],yrange[1])        
        ax.set_xlabel(ax_list[0].title(language=language))
        ax.set_ylabel(ax_list[1].title(language=language))
        if (_options['Log x']):
            ax.set_xscale('log')
        if (_options['Log y']):
            ax.set_yscale('log')
        title = ax.get_title()
        if (title is None):
            title = ''
        if (title[-3:] != '...'):
            newtitle = ''
            if (d.exp_id is not None):
                newtitle += str(d.exp_id)
                if (d.data_title is not None):
                    newtitle += ' ' + d.data_title
            if (len(newtitle) != 0):
                if (len(title + newtitle) < 40):
                    if (title != ''):
                        title += ','+newtitle
                    else:
                        title = newtitle
                else:
                    title += ',...'
        ax.set_title(title)   
                
    elif ((_plot_type == 'anim-image') or (_plot_type == 'anim-contour')):
        if (d.data is None):
            raise ValueError("Cannot plot DataObject without data.")
        if (len(d.shape) != 3):
            raise TypeError("Animated image plot is applicable to 3D data only. Use slicing.")
        if (d.data.dtype.kind == 'c'):
            raise TypeError("Animated image plot is applicable only to real data.")
        # Checking for numeric type
        try:
            d.data[0,0] += 1
        except TypeError:
            raise TypeError("Animated image plot is applicable only to numeric data.")
        
        yrange = _options['Y range']
        if (yrange is not None):
            if ((type(yrange) is not list) or (len(yrange) != 2)):
                raise ValueError("Invalid Y range setting.")

        # Processing axes
        # Although the plot will be cleared the existing plot axes will be considered
        default_axes = [d.coordinates[0].unit.name, d.coordinates[1].unit.name,d.coordinates[2].unit.name,'__Data__']
        try:
            pdd_list, ax_list = _plot_id.check_axes(d, 
                                                    axes, 
                                                    clear=_options['Clear'], 
                                                    default_axes=default_axes, 
                                                    force=_options['Force axes'])
        except ValueError as e:
            raise e
        
        if (not ((pdd_list[3].data_type == PddType.Data) and (pdd_list[3].data_object == d))):
            raise ValueError("For anim-image/anim-contour plot only data can be plotted on the z axis.")
        if ((pdd_list[0].data_type != PddType.Coordinate) or (pdd_list[1].data_type != PddType.Coordinate)) :
            raise ValueError("X and y coordinates of anim-image/anim-contour plot type should be coordinates.")
        if (pdd_list[2].data_type != PddType.Coordinate) :
            raise ValueError("Time coordinate of anim-image/anim-contour plot should be coordinate.")

        coord_x = pdd_list[0].value
        coord_y = pdd_list[1].value
        coord_t = pdd_list[2].value
        
        if (len(coord_t.dimension_list) != 1):
            raise ValueError("Time coordinate for anim-image/anim-contour plot should be changing only along one dimension.")
        try:
            coord_x.dimension_list.index(coord_t.dimension_list[0])
            badx = True
        except:
            badx = False
        try:
            coord_y.dimension_list.index(coord_t.dimension_list[0])
            bady = True
        except:
            bady = False
        if (badx or bady):
            raise ValueError("X and y coordinate for anim-image plot should not change in time dimension.")
            
        index = [0] * 3
        index[coord_t.dimension_list[0]] = ...
        tdata = coord_t.data(data_shape=d.shape,index=index)[0].flatten()
        if (not coord_y.isnumeric()):
            raise ValueError('Coordinate '+coord_y.unit.name+' is not numeric.')

        if ((coord_x.mode.equidistant) and (len(coord_x.dimension_list) == 1) and
            (coord_y.mode.equidistant) and (len(coord_y.dimension_list) == 1)):
            # This data is image-like with data points on a rectangular array
            image_like = True
        elif ((len(coord_x.dimension_list) == 1) and (len(coord_y.dimension_list) == 1)):
            if (not coord_x.isnumeric()):
                raise ValueError('Coordinate '+coord_x.unit.name+' is not numeric.')
            if (not coord_y.isnumeric()):
                raise ValueError('Coordinate '+coord_y.unit.name+' is not numeric.')
            index = [0] * len(d.shape)
            index[coord_x.dimension_list[0]] = ...
            xdata,xdata_low,xdata_high = coord_x.data(data_shape=d.shape,index=index)
            xdata = xdata.flatten()
            dx = xdata[1:] - xdata[:-1]
            index = [0] * len(d.shape)
            index[coord_y.dimension_list[0]] = ...
            ydata,ydata_low,ydata_high = coord_y.data(data_shape=d.shape,index=index)
            ydata = ydata.flatten()
            dy = ydata[1:] - ydata[:-1]
            if ((np.nonzero(np.abs(dx - dx[0]) / math.fabs(dx[0]) > 0.01)[0].size == 0) and
                (np.nonzero(np.abs(dy - dy[0]) / math.fabs(dy[0]) > 0.01)[0].size == 0)):
                # Actually the non-equidistant coordinates are equidistant
                image_like = True
            else:
                image_like = False
        else:
            image_like = False
        if (image_like):        
            xdata_range = coord_x.data_range(data_shape=d.shape)[0]   
            ydata_range = coord_y.data_range(data_shape=d.shape)[0]   
        else:
            ydata,ydata_low,ydata_high = coord_y.data(data_shape=d.shape)
            xdata,xdata_low,xdata_high = coord_x.data(data_shape=d.shape)
            
        try:
            cmap_obj = plt.cm.get_cmap(cmap)
            if (_options['Nan color'] is not None):
                cmap_obj.set_bad(_options['Nan color'])
        except ValueError:
            raise ValueError("Invalid color map.")

        if (_options['Video file'] is not None):
            anim_ims = []
        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=_plot_id.base_subplot)
        ax=plt.plot()
        _plot_id.plt_axis_list = []
        _plot_id.plt_axis_list.append(plt.subplot(gs[0,0]))
        plt.subplot(_plot_id.base_subplot)
        plt.plot()
        plt.cla()
        ax=plt.gca()
        for it in range(len(tdata)):
            if (_options['Video file'] is None):
                #Erasing plot if not making video
#                plt.subplot(_plot_id.base_subplot)
                plt.cla()
            time_index = [slice(0,dim) for dim in d.data.shape]
            time_index[coord_t.dimension_list[0]] = it
            time_index = tuple(time_index)
    
            if (zrange is None):
                vmin = np.nanmin(d.data[time_index])
                vmax = np.nanmax(d.data[time_index])
            else:
                vmin = zrange[0]
                vmax = zrange[1]
    
            if (vmax <= vmin):
                raise ValueError("Invalid z range.")
                
            if (_options['Log z']):
                if (vmin <= 0):
                    raise ValueError("z range[0] cannot be negative or zero for logarithmic scale.")
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
                locator = ticker.LogLocator(subs='all')
            else:
                norm = None
                locator = None
                    
            if (contour_levels is None):
                contour_levels = 255
                
            _plot_opt = _plot_options[0]
    
            if (image_like):
                try: 
                    if (coord_x.dimension_list[0] == 0):
                        im = np.clip(np.transpose(d.data[time_index]),vmin,vmax)
                        img = plt.imshow(im,extent=xdata_range + ydata_range,norm=norm,
                                        cmap=cmap_obj,vmin=vmin,aspect=_options['Aspect ratio'],interpolation=_options['Interpolation'],
                                        vmax=vmax,origin='lower',**_plot_opt)
                    else:
                        im = np.clip(d.data[time_index],vmin,vmax)
                        img = plt.imshow(im,extent=xdata_range + ydata_range,norm=norm,
                                        cmap=cmap_obj,vmin=vmin,aspect=_options['Aspect ratio'],interpolation=_options['Interpolation'],
                                        vmax=vmax,origin='lower',**_plot_opt)            
                        if (_options['Video file'] is None):
                            del im
                except Exception as e:
                    raise e
            else:
                if (_plot_type == 'anim-image'):
                    xgrid, ygrid = grid_to_box(xdata,ydata)
                    im = np.clip(np.transpose(d.data[time_index]),vmin,vmax)
                    try:
                        img = plt.pcolormesh(xgrid,ygrid,im,norm=norm,cmap=cmap,vmin=vmin,
                                          origin='lower',vmax=vmax,**_plot_opt)
                    except Exception as e:
                        raise e
                    if (_options['Video file'] is None):
                        del im
                else:
                    try:
                        im = plt.clip(d.data[time_index],vmin,vmax)
                        img = ax.contourf(xdata,ydata,im,contour_levels,norm=norm,
                                          origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,**_plot_opt)
                        if (_options['Video file'] is None):
                            del im
                    except Exception as e:
                        raise e
    
            if (_options['Colorbar']):
                cbar = plt.colorbar(img,cax=_plot_id.base_subplot)
                cbar.set_label(d.data_unit.name)

            if (xrange is not None):
                ax.set_xlim(xrange[0],xrange[1])
            if (yrange is not None):
                ax.set_ylim(yrange[0],yrange[1])        
            ax.set_xlabel(ax_list[0].title(language=language))
            ax.set_ylabel(ax_list[1].title(language=language))
            if (_options['Log x']):
                ax.set_xscale('log')
            if (_options['Log y']):
                ax.set_yscale('log')
            title = coord_t.unit.name+'='+"{:10.5f}".format(tdata[it])+' ['+coord_t.unit.unit+']'
            plt.title(title)
            plt.show()
            time.sleep(_options['Waittime'])
            plt.pause(0.001)
            if (_options['Video file'] is not None):
                anim_ims.append([img])
        if (_options['Video file'] is not None):
            if (_options['animation.ffmpeg_path'] is not None):
                plt.rcParams['animation.ffmpeg_path'] = _options['animation.ffmpeg_path']
            im_ani = animation.ArtistAnimation(plt.gcf(), anim_ims, interval=1000./_options['Video framerate'],
                                               blit=False)
            im_ani.save(_options['Video file'])    
#            plt.rcParams['animation.ffmpeg_path'] = 'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
#            im_ani = animation.ArtistAnimation(plt.gcf(), anim_ims, interval=50, repeat_delay=3000,
#                                               blit=False)
#            im_ani.save(_options['Video file'])
#            im_ani.save(_options['Video file'], writer='ffmpeg')
    

    plt.show()       
 
#    if (_options['Clear']):
#        _plot_id.number_of_plots = 0
#        _plot_id.plot_data = []
#        _plot_id.plt_axis_list = None            
#    _plot_id.number_of_plots += 1
    _plot_id.axes = ax_list
    _plot_id.plot_data.append(pdd_list)
    #Setting this one the default plot ID
    _plot_id.options.append(_options)
    _plot_id.plot_type = _plot_type
    set_plot_id(_plot_id)

    return _plot_id

