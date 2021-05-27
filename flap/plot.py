# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:37:06 2019

@author: Zoletnik
@coauthor: Lampert
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
import os
import copy
from enum import Enum
import math
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.animation as animation
import matplotlib.lines as mlines
from matplotlib.widgets import Button, Slider
from matplotlib import ticker

import numpy as np

try:
    import cv2
    cv2_presence = True
except:
    print('OpenCV is not present on the computer. Video saving is not available.')
    cv2_presence = False

#from .coordinate import *
#from .tools import *
import flap.config
import flap.tools
import flap.coordinate

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
            axx = flap.coordinate.Unit()
        pdd_list.append(pdd)
        ax_list.append(axx)
    return pdd_list,ax_list

class PlotAnimation:
    
    def __init__(self, ax_list, axes, axes_unit_conversion, d, xdata, ydata, tdata, xdata_range, ydata_range,
                 cmap_obj, contour_levels, coord_t, coord_x,
                 coord_y, cmap, options, overplot_options, xrange, yrange,
                 zrange, image_like, plot_options, language, plot_id, gs):
   
        self.ax_list = ax_list
        self.axes = axes
        self.axes_unit_conversion=axes_unit_conversion
        self.contour_levels = contour_levels        
        self.cmap = cmap
        self.cmap_obj = cmap_obj
        self.coord_t = coord_t  
        self.coord_x = coord_x
        self.coord_y = coord_y        
        self.current_frame = 0.
        self.d = d
        self.fig = plt.figure(plot_id.figure) 
        self.gs = gs
        self.image_like = image_like
        self.language = language
        self.options = options
        self.overplot_options = overplot_options
        self.pause = False
        self.plot_id = plot_id
        self.plot_options = plot_options        
        self.speed = 40.
        
        self.tdata = tdata
        self.xdata = xdata
        self.ydata = ydata
        if xdata_range is not None:
            self.xdata_range = [self.axes_unit_conversion[0] * xdata_range[0],
                                self.axes_unit_conversion[0] * xdata_range[1]]
        else:
            self.xdata_range=None
        if ydata_range is not None:
            self.ydata_range = [self.axes_unit_conversion[1] * ydata_range[0],
                                self.axes_unit_conversion[1] * ydata_range[1]]
        else:
            self.ydata_range=None
        
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange

        
        if (self.contour_levels is None):
            self.contour_levels = 255
            
    def animate(self):
        
        pause_ax = plt.figure(self.plot_id.figure).add_axes((0.78, 0.94, 0.1, 0.04))
        self.pause_button = Button(pause_ax, 'Pause', hovercolor='0.975')
        self.pause_button.on_clicked(self._pause_animation)
        pause_ax._button=self.pause_button
        
        reset_ax = plt.figure(self.plot_id.figure).add_axes((0.78, 0.89, 0.1, 0.04))
        reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
        reset_button.on_clicked(self._reset_animation)
        reset_ax._button=reset_button
        
        slow_ax = plt.figure(self.plot_id.figure).add_axes((0.88, 0.94, 0.1, 0.04))
        self.slow_button = Button(slow_ax, str(int(1000./(self.speed/0.8)))+'fps', hovercolor='0.975')
        self.slow_button.on_clicked(self._slow_animation)
        slow_ax._button=self.slow_button
        
        speed_ax = plt.figure(self.plot_id.figure).add_axes((0.88, 0.89, 0.1, 0.04))
        self.speed_button = Button(speed_ax, str(int(1000./(self.speed*0.8)))+'fps', hovercolor='0.975')
        self.speed_button.on_clicked(self._speed_animation)
        speed_ax._button=self.speed_button
        
        
        slider_ax = plt.figure(self.plot_id.figure).add_axes((0.1, 0.94, 0.5, 0.04))
        self.time_slider = Slider(slider_ax, label=self.axes[2],
                                  valmin=self.tdata[0]*self.axes_unit_conversion[2], 
                                  valmax=self.tdata[-1]*self.axes_unit_conversion[2],
                                  valinit=self.tdata[0]*self.axes_unit_conversion[2])
        self.time_slider.on_changed(self._set_animation)
        
        #The following line needed to be removed for matplotlib 3.4.1
        #plt.subplot(self.plot_id.base_subplot)
        
        self.ax_act = plt.subplot(self.gs[0,0])
        if (len(self.coord_x.dimension_list) == 3 or
            len(self.coord_y.dimension_list) == 3):
            self.ax_act.set_autoscale_on(False)
            
            
#The following lines set the axes to be equal if the units of the axes-to-be-plotted are the same
        if self.options['Equal axes']:
            axes_coordinate_decrypt=[0] * len(self.axes)
            for i_axes in range(len(self.axes)):
                for j_coordinate in range(len(self.d.coordinates)):
                    if (self.d.coordinates[j_coordinate].unit.name == self.axes[i_axes]):
                        axes_coordinate_decrypt[i_axes]=j_coordinate
            for i_check in range(len(self.axes))        :
                for j_check in range(i_check+1,len(self.axes)):
                    if (self.d.coordinates[axes_coordinate_decrypt[i_check]].unit.unit ==
                        self.d.coordinates[axes_coordinate_decrypt[j_check]].unit.unit):
                        self.ax_act.set_aspect(1.0)
                    
        time_index = [slice(0,dim) for dim in self.d.data.shape]
        time_index[self.coord_t.dimension_list[0]] = 0
        time_index = tuple(time_index)
        
        act_ax_pos=self.ax_act.get_position()
        slider_ax.set_position([act_ax_pos.x0,0.94,0.5,0.04])

        if (self.zrange is None):
            self.zrange=[np.nanmin(self.d.data),
                         np.nanmax(self.d.data)]
        self.vmin = self.zrange[0]
        self.vmax = self.zrange[1]


        if (self.vmax <= self.vmin):
            raise ValueError("Invalid z range.")
            
        if (self.options['Log z']):
            if (self.vmin <= 0):
                raise ValueError("z range[0] cannot be negative or zero for logarithmic scale.")
            self.norm = colors.LogNorm(vmin=self.vmin, vmax=self.vmax)
            self.locator = ticker.LogLocator(subs='all')
            ticker.LogLocator(subs='all')
        else:
            self.norm = None
            self.locator = None
                
            
        _plot_opt = self.plot_options[0]

        if (self.image_like):
            try: 
                if (self.coord_x.dimension_list[0] < self.coord_y.dimension_list[0]):
                    im = np.clip(np.transpose(self.d.data[time_index]),self.vmin,self.vmax)
                else:
                    im = np.clip(self.d.data[time_index],self.vmin,self.vmax)
                img = plt.imshow(im,extent=self.xdata_range + self.ydata_range,norm=self.norm,
                                cmap=self.cmap_obj,vmin=self.vmin,aspect=self.options['Aspect ratio'],interpolation=self.options['Interpolation'],
                                vmax=self.vmax,origin='lower',**_plot_opt)            
                del im
            except Exception as e:
                raise e
        else:
            if (len(self.xdata.shape) == 3 and len(self.ydata.shape) == 3):
                xgrid, ygrid = flap.tools.grid_to_box(self.xdata[time_index]*self.axes_unit_conversion[0],
                                                      self.ydata[time_index]*self.axes_unit_conversion[1])
            else:
                xgrid, ygrid = flap.tools.grid_to_box(self.xdata*self.axes_unit_conversion[0],
                                                      self.ydata*self.axes_unit_conversion[1])
                
            im = np.clip(np.transpose(self.d.data[time_index]),self.vmin,self.vmax)
            try:
                img = plt.pcolormesh(xgrid,ygrid,im,norm=self.norm,cmap=self.cmap,vmin=self.vmin,
                                  vmax=self.vmax,**_plot_opt)
            except Exception as e:
                raise e
            del im
            
            self.xdata_range=None
            self.ydata_range=None
            if (self.xrange is None):
                self.xrange=[np.min(self.xdata),np.max(self.xdata)]
            
            plt.xlim(self.xrange[0]*self.axes_unit_conversion[0],
                     self.xrange[1]*self.axes_unit_conversion[0])
                
            if (self.yrange is None):
                self.yrange=[np.min(self.ydata),np.max(self.ydata)]
                
            plt.ylim(self.yrange[0]*self.axes_unit_conversion[1],
                     self.yrange[1]*self.axes_unit_conversion[1]) 

        if (self.options['Colorbar']):
            cbar = plt.colorbar(img,ax=self.ax_act)
            cbar.set_label(self.d.data_unit.name)
#EFIT overplot feature implementation:
            #It needs to be more generalied in the future as the coordinates are not necessarily in this order: [time_index,spat_index]
            #This needs to be cross-checked with the time array's dimensions wherever there is a call for a certain index.            

        
        if self.axes_unit_conversion[0] == 1.:
            plt.xlabel(self.ax_list[0].title(language=self.language))
        else:
            plt.xlabel(self.ax_list[0].title(language=self.language, 
                                             new_unit=self.options['Plot units'][self.axes[0]]))
            
        if self.axes_unit_conversion[1] == 1.:
            plt.ylabel(self.ax_list[1].title(language=self.language))
        else:
            plt.ylabel(self.ax_list[1].title(language=self.language, 
                                             new_unit=self.options['Plot units'][self.axes[1]]))
            
        if (self.options['Log x']):
            plt.xscale('log')
        if (self.options['Log y']):
            plt.yscale('log')
        
        if self.options['Plot units'] is not None:
            if self.axes[2] in self.options['Plot units']:
                time_unit=self.options['Plot units'][self.axes[2]]
                time_coeff=self.axes_unit_conversion[2]
            else:
                time_unit=self.coord_t.unit.unit
            time_coeff=1.
        else:
            time_unit=self.coord_t.unit.unit
            time_coeff=1.
        title = str(self.d.exp_id)+' @ '+self.coord_t.unit.name+'='+"{:10.7f}".format(self.tdata[0]*time_coeff)+\
                ' ['+time_unit+']'
                              
        plt.title(title)

        plt.show(block=False)        
        self.anim = animation.FuncAnimation(self.fig, self.animate_plot, 
                                            len(self.tdata),
                                            interval=self.speed,blit=False)
        
    def animate_plot(self, it):
        
        time_index = [slice(0,dim) for dim in self.d.data.shape]
        time_index[self.coord_t.dimension_list[0]] = it
        time_index = tuple(time_index)
        
        self.time_slider.eventson = False
        self.time_slider.set_val(self.tdata[it]*self.axes_unit_conversion[2])
        self.time_slider.eventson = True       
        
        self.current_frame = it

        plot_opt = copy.deepcopy(self.plot_options[0])
        self.ax_act.clear()
        
        if (self.image_like):
            try: 
                if (self.coord_x.dimension_list[0] < self.coord_y.dimension_list[0]):
                    im = np.clip(np.transpose(self.d.data[time_index]),self.vmin,self.vmax)
                else:
                    im = np.clip(self.d.data[time_index],self.vmin,self.vmax)
                plt.imshow(im,extent=self.xdata_range + self.ydata_range,norm=self.norm,
                           cmap=self.cmap_obj,vmin=self.vmin,
                           aspect=self.options['Aspect ratio'],
                           interpolation=self.options['Interpolation'],
                           vmax=self.vmax,origin='lower',**plot_opt)
                del im
            except Exception as e:
                raise e
        else:
            self.ax_act.set_autoscale_on(False)
            self.ax_act.set_xlim(self.xrange[0]*self.axes_unit_conversion[0],
                                 self.xrange[1]*self.axes_unit_conversion[0])  
            self.ax_act.set_ylim(self.yrange[0]*self.axes_unit_conversion[1],
                                 self.yrange[1]*self.axes_unit_conversion[1])
            if (len(self.xdata.shape) == 3 and len(self.ydata.shape) == 3):
                xgrid, ygrid = flap.tools.grid_to_box(self.xdata[time_index]*self.axes_unit_conversion[0],
                                                      self.ydata[time_index]*self.axes_unit_conversion[1]) #Same issue, time is not necessarily the first flap.coordinate.
            else:
                xgrid, ygrid = flap.tools.grid_to_box(self.xdata*self.axes_unit_conversion[0],
                                                      self.ydata*self.axes_unit_conversion[1])
            im = np.clip(np.transpose(self.d.data[time_index]),self.vmin,self.vmax)
            try:
                plt.pcolormesh(xgrid,ygrid,im,norm=self.norm,cmap=self.cmap,vmin=self.vmin,
                               vmax=self.vmax,**plot_opt)
            except Exception as e:
                raise e
            del im
        
        if (self.overplot_options is not None):
            for path_obj_keys in self.overplot_options['path']:
                if self.overplot_options['path'][path_obj_keys]['Plot']:
                    im = plt.plot(self.overplot_options['path'][path_obj_keys]['data']['Data resampled'][0,it,:]*self.axes_unit_conversion[0],
                                  self.overplot_options['path'][path_obj_keys]['data']['Data resampled'][1,it,:]*self.axes_unit_conversion[1],
                                  color=self.overplot_options['path'][path_obj_keys]['Color'])

            for contour_obj_keys in self.overplot_options['contour']:
                if self.overplot_options['contour'][contour_obj_keys]['Plot']:
                    im = plt.contour(self.overplot_options['contour'][contour_obj_keys]['data']['X coord resampled'][it,:,:].transpose()*self.axes_unit_conversion[0],
                                     self.overplot_options['contour'][contour_obj_keys]['data']['Y coord resampled'][it,:,:].transpose()*self.axes_unit_conversion[1],
                                     self.overplot_options['contour'][contour_obj_keys]['data']['Data resampled'][it,:,:],
                                     levels=self.overplot_options['contour'][contour_obj_keys]['nlevel'],
                                     cmap=self.overplot_options['contour'][contour_obj_keys]['Colormap'])
            for line_obj_keys in self.overplot_options['line']:
                xmin, xmax = self.ax_act.get_xbound()
                ymin, ymax = self.ax_act.get_ybound()
                if self.overplot_options['line'][line_obj_keys]['Plot']:
                    
                    if 'Horizontal' in self.overplot_options['line'][line_obj_keys]:
                        h_coords=self.overplot_options['line'][line_obj_keys]['Horizontal']
                        for segments in h_coords:
                            if segments[0] > ymin and segments[0] < ymax:
                                l = mlines.Line2D([xmin,xmax], [segments[0],segments[0]], color=segments[1])
                                self.ax_act.add_line(l)
                                
                    if 'Vertical' in self.overplot_options['line'][line_obj_keys]:
                        v_coords=self.overplot_options['line'][line_obj_keys]['Vertical']
                        for segments in v_coords:
                            if segments[0] > xmin and segments[0] < xmax:
                                l = mlines.Line2D([segments[0],segments[0]], [ymin,ymax], color=segments[1])
                                self.ax_act.add_line(l)

        if self.axes_unit_conversion[0] == 1.:
            plt.xlabel(self.ax_list[0].title(language=self.language))
        else:
            plt.xlabel(self.ax_list[0].title(language=self.language, new_unit=self.options['Plot units'][self.axes[0]]))
            
        if self.axes_unit_conversion[1] == 1.:
            plt.ylabel(self.ax_list[1].title(language=self.language))
        else:
            plt.ylabel(self.ax_list[1].title(language=self.language, new_unit=self.options['Plot units'][self.axes[1]]))
        
        if self.options['Plot units'] is not None:
            if self.axes[2] in self.options['Plot units']:
                time_unit=self.options['Plot units'][self.axes[2]]
                time_coeff=self.axes_unit_conversion[2]
            else:
                time_unit=self.coord_t.unit.unit
                time_coeff=1.
        else:
            time_unit=self.coord_t.unit.unit
            time_coeff=1.
            
        title = str(self.d.exp_id)+' @ '+self.coord_t.unit.name+'='+"{:10.7f}".format(self.tdata[it]*time_coeff)+\
                ' ['+time_unit+']'
        
        self.ax_act.set_title(title)
    
    def _reset_animation(self, event):
        self.anim.event_source.stop()
        self.speed = 40.
        self.anim = animation.FuncAnimation(plt.figure(self.plot_id.figure), self.animate_plot, 
                                            len(self.tdata),interval=self.speed,blit=False)
        self.anim.event_source.start()
        self.pause = False
        
    def _pause_animation(self, event):
        if self.pause:
            self.anim.event_source.start()
            self.pause = False
            self.pause_button.label.set_text("Pause")
        else:
            self.anim.event_source.stop()
            self.pause = True
            self.pause_button.label.set_text("Start")
        
    def _set_animation(self, time):
        self.anim.event_source.stop()
        frame=(np.abs(self.tdata*self.axes_unit_conversion[2]-time)).argmin()
        self.anim = animation.FuncAnimation(plt.figure(self.plot_id.figure), self.animate_plot, 
                                            frames=np.arange(frame,len(self.tdata)-1),
                                            interval=self.speed,blit=False)
        self.anim.event_source.start()
        self.pause = False
        
    def _slow_animation(self, event):
        self.anim.event_source.stop()
        self.speed=self.speed/0.8
        self.anim = animation.FuncAnimation(plt.figure(self.plot_id.figure), self.animate_plot, 
                                            frames=np.arange(self.current_frame,len(self.tdata)-1),
                                            interval=self.speed,blit=False)
        self.speed_button.label.set_text(str(int(1000./(self.speed*0.8)))+'fps')
        self.slow_button.label.set_text(str(int(1000./(self.speed/0.8)))+'fps')
        self.anim.event_source.start()
        self.pause = False
        
    def _speed_animation(self, event):  
        self.anim.event_source.stop()
        self.speed=self.speed*0.8
        self.anim = animation.FuncAnimation(plt.figure(self.plot_id.figure), self.animate_plot, 
                                            frames=np.arange(self.current_frame,len(self.tdata)-1),
                                            interval=self.speed,blit=False)
        self.speed_button.label.set_text(str(int(1000./(self.speed*0.8)))+'fps')
        self.slow_button.label.set_text(str(int(1000./(self.speed/0.8)))+'fps')
        self.anim.event_source.start()
        self.pause = False
            
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
                if ((ax_in.name != ax_plot.name) or (ax_in.unit != ax_plot.unit)):
                    if (force):
                        u = flap.coordinate.Unit()
                        if (ax_in.name == ax_plot.name):
                            u.name = ax_in.name
                        elif ((ax_in.name is None) or (ax_in.name == '')):
                            u.name = ax_plot.name 
                        elif ((ax_plot.name is None) or (ax_plot.name == '')):
                            u.name = ax_in.name    
                        if (ax_in.unit == ax_plot.unit):
                            u.unit = ax_in.unit
                        elif ((ax_in.unit is None) or (ax_in.unit == '')):
                            u.unit = ax_plot.unit 
                        elif ((ax_plot.unit is None) or (ax_plot.unit == '')):
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
    plot_options: Dictionary or list of dictionaries. Will be passed over to the plot call. For plots with multiple subplots this can be
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
        'Video format': Format of the video. Valid: 'avi' (windows,macOS,linux), 
                                                    'mkv', 'mp4' (macOS,linux)
                        Note for macOS: mp4 is the only embeddable in Powerpoint.
        'Clear': Boolean. If True don't use the existing plots, generate new. (No overplotting.)
        'Force axes': Force overplotting even if axes are incpomatible
        'Colorbar': Boolelan. Switch on/off colorbar
        'Nan color': The color to use in image data plotting for np.nan (not-a-number) values
        'Interpolation': Interpolation method for image plot.
        'Language': Language of certain standard elements in the plot. ('EN', 'HU')
        'Overplot options': Dictionary of overplotting options:
            Path, contour or line overplotting for the different plot-types:
            Dictionary with keys of 'contour', 'path' or 'line':
            
            Example options['Overplot options']['contour']:
                
                options['Overplot options']['contour']=\
                {'CNAME':{'Data object':None, #2D spatial FLAP object name
                          'Plot':False,       #Boolean for plotting
                          'Colormap':None,    #Colormap for the contour
                          'nlevel':51,        #Level for the contour
                          'Slicing':None,     #Slicing for the FLAP object
                          }}    
                Can contain multiple CNAME (contour name) keywords, each is 
                plotted. Works in 2D plots. Should be in the same time unit as
                the data is in the animations.
                
            Example options['Overplot options']['path']:  
                
                options['Overplot options']['path']=\
                {'PNAME':{'Data object X':None, #1D spatial FLAP object name
                          'Data object Y':None, #1D spatial FLAP object name
                          'Plot':False,         #BOOlean for plotting
                          'Color':'black',      #Color of the path
                          'Slicing':None,       #Slicing for the FLAP object
                          }}
                Can contain multiple PNAME (path name) keyword, each is 
                plotted. Works in 2D plots. Should be in the same time unit as
                the data is in the animations.
                
            Example options['Overplot options']['line']:                  
                
                options['Overplot options']['path']=\
                {'LNAME':{'Vertical':[X_coord,'red'], #[X coordinate in the unit of the plot, plot color]
                          'Horizontal':[Y_coord,'blue'], #[Y coordinate in the unit of the plot, plot color]
                          'Plot':False,
                          }}
                Can contain multiple LNAME (line name) keywords. If 'Vertical'
                keyword is present, a line is vertical plotted at X_coord.
                If 'Horizontal' keyword is present, a line is plottad at Y_coord.
                Works in all plot types.
                
        'Prevent saturation': Prevents saturation of the video signal when it exceeds zrange[1]
            It uses data modulo zrange[1] to overcome the saturation. (works for animation)
            (default: False)
            Caveat: doubles the memory usage of the plotted dataset during the plotting time.
        'Plot units': The plotting units for each axis. It can be a dictionary or a list. Its
            use is different for the two cases as such:
            Dictionary input: keywords: axis name (e.g. 'Device R'), value: unit (e.g. 'mm')
                The number of keys is not limited. The ones from the axes will be used.
                e.g.: options['Plot units']={'Device R':'mm', 'Device z':'mm', 'Time':'ms'}
            List input: the number of values should correspond to the axes input as such:
                e.g.: axes=['Device R','Device z','Time'] --> options['Plot units']=['mm','mm','ms']
        'Equal axes': IF the units of the x and y axes coordinates match, it makes the plot's
            axes to have equal spacing. Doesn't care about the plot units, just data units.
            (default: False)
        'Axes visibility': Hides the title and labels of the axes if set to [False,False]. First
            value is for the X axis and the second is for the Y axis. (default: [True,True])
            
    """

    default_options = {'All points': False, 'Error':True, 'Y separation': None,
                       'Log x': False, 'Log y': False, 'Log z': False, 'maxpoints':4000, 'Complex mode':'Amp-phase',
                       'X range':None, 'Y range': None, 'Z range': None,'Aspect ratio':'auto',
                       'Clear':False,'Force axes':False,'Language':'EN','Maxpoints': 4000,
                       'Levels': 10, 'Colormap':None, 'Waittime':1,'Colorbar':True,'Nan color':None,
                       'Interpolation':'bilinear','Video file':None, 'Video framerate': 20,'Video format':'avi',
                       'Overplot options':None, 'Prevent saturation':False, 'Plot units':None,
                       'Equal axes':False, 'Axes visibility':[True,True],
                       }
    
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
        
    if (_options['Z range'] is not None) and (_options['Prevent saturation']):
        if (_options['Z range'] is not None):
            if ((type(_options['Z range']) is not list) or (len(_options['Z range']) != 2)):
                raise ValueError("Invalid Z range setting.")
        if ((slicing is not None) or (summing is not None)):
            d = copy.deepcopy(data_object.slice_data(slicing=slicing, summing=summing, options=slicing_options))
        else:
            d = copy.deepcopy(data_object)  
        d.data=np.mod(d.data,_options['Z range'][1])
    else:
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
    known_plot_types = ['xy','scatter','multi xy', 'image', 'anim-image','contour','anim-contour','animation']
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
            _plot_type = flap.tools.find_str_match(plot_type,known_plot_types)
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
        compt = flap.tools.find_str_match(_options['Complex mode'], ['Amp-phase','Real-imag'])
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
    
    if (_options['Video file'] is not None):
        if ((os.sys.platform == 'darwin' or 'linux' in os.sys.platform) and 
            (options['Video format'] not in ['avi','mkv', 'mp4'])):
            raise ValueError("The chosen video format is not cupported on macOS.")
        if os.sys.platform == 'win32' and _options['Video format'] != 'avi':
            raise ValueError("The chosen video format is not cupported on Windowd.")
        video_codec_decrypt={'avi':'XVID',
                             'mkv':'X264',
                             'mp4':'mp4v'}
        video_codec_code=video_codec_decrypt[_options['Video format']]
        print('Forcing waittime to be 0s for video saving.')
        _options['Waittime']=0.
            
    #These lines do the coordinate unit conversion
    axes_unit_conversion=[1.,1.,1.]

    if _options['Plot units'] is not None:
        unit_length=len(_options['Plot units'])
        if unit_length > 3:
            raise ValueError('Only three units are allowed for the three coordinates.')
        #For some reason the input list or dictionary can morph into a list or a dict class    
        possible_types=[list,dict,'<class \'list\'>','<class \'dict\'>'] 
        if not (type(_options['Plot units']) in possible_types):
            raise TypeError('The \'Plot units\' option needs to be either a dictionary or a list.')
        #Converting the input list into dictionary:
        if (type(_options['Plot units']) is list or 
            type(_options['Plot units']) == '<class \'list\'>'):
            temp_units= _options['Plot units'][:]
            _options['Plot units']={}
            for index_axes in range(len(axes)):
                _options['Plot units'][axes[index_axes]]=temp_units[index_axes]
        #Finding the corresponding data coordinate to the input unit conversion and converting
        unit_conversion_coeff={}
        for plot_unit_name in _options['Plot units']:
            for index_data_unit in range(len(d.coordinates)):
                if (plot_unit_name == d.coordinates[index_data_unit].unit.name):
                    data_coordinate_unit=d.coordinates[index_data_unit].unit.unit
                    plot_coordinate_unit=_options['Plot units'][plot_unit_name]
                    unit_conversion_coeff[plot_unit_name]=flap.tools.unit_conversion(original_unit=data_coordinate_unit,
                                                                                     new_unit=plot_coordinate_unit)
        #Saving the coefficients in the same order as the axes are
        for index_axes in range(len(axes)):
            if axes[index_axes] in _options['Plot units']:
                axes_unit_conversion[index_axes]=unit_conversion_coeff[axes[index_axes]]

    #overplot_available_plot_types=['image', 'anim-image','contour','anim-contour','animation']
    overplot_options=None
    if _options['Overplot options'] is not None:
        
        default_overplot_options={'contour':{'NAME':{'Data object':None,
                                                     'Plot':False,
                                                     'Colormap':None,
                                                     'nlevel':51,
                                                     'Slicing':None,
                                                     }},
    
                                  'path':{'NAME':{'Data object X':None,
                                                  'Data object Y':None,
                                                  'Plot':False,
                                                  'Color':'black',
                                                  'Slicing':None,
                                                  }},
                                                  
                                  'line':{'NAME':{'Vertical':[0,'red'],
                                                  'Horizontal':[1,'blue'],
                                                  'Plot':False,
                                                  }}
                                  }
        
        overplot_options=flap.config.merge_options(default_overplot_options,_options['Overplot options'],data_source=data_object.data_source)
             
        #TIME (AXES(2)) INDEPENDENT OVERPLOTTING OBJECT CREATION
        if plot_type in ['image', 'contour']:
            for path_obj_keys in overplot_options['path']:
                try:
                    x_object_name=overplot_options['path'][path_obj_keys]['Data object X']
                    x_object=flap.get_data_object_ref(x_object_name,exp_id=d.exp_id)
                except:
                    raise ValueError("The object "+x_object_name+" cannot be read.")
                try:
                    y_object_name=overplot_options['path'][path_obj_keys]['Data object Y']
                    y_object=flap.get_data_object_ref(y_object_name,exp_id=d.exp_id)
                except:
                    raise ValueError("The object "+y_object_name+" cannot be read.")
                #Error handling
                
                if 'Slicing' in overplot_options['path'][path_obj_keys]:
                    try:
                        x_object=x_object.slice_data(slicing=overplot_options['path'][path_obj_keys]['Slicing'])
                        y_object=y_object.slice_data(slicing=overplot_options['path'][path_obj_keys]['Slicing'])
                    except:
                        raise ValueError('Slicing did not succeed. Try it outside the plotting!')
                        
                if (len(x_object.data.shape) != 1 or len(y_object.data.shape) != 1):
                    raise ValueError("The "+overplot_options['path'][path_obj_keys]['Data object X']+' or '
                                     "the "+overplot_options['path'][path_obj_keys]['Data object Y']+" data is not 1D. Use slicing.")
                unit_conversion_coeff=[1]*2
                original_units=[x_object.data_unit.unit,y_object.data_unit.unit]
                for index_coordinate in range(len(d.coordinates)):
                    for index_axes in range(2):
                        if (d.coordinates[index_coordinate].unit.name == axes[index_axes]):                
                            unit_conversion_coeff[index_axes] = flap.tools.unit_conversion(original_unit=original_units[index_axes],
                                                                                           new_unit=d.coordinates[index_coordinate].unit.unit)
                
                overplot_options['path'][path_obj_keys]['data']={}
                overplot_options['path'][path_obj_keys]['data']['Data']=np.asarray([x_object.data*unit_conversion_coeff[0],
                                                                            y_object.data*unit_conversion_coeff[1]])
            
            for contour_obj_keys in overplot_options['contour']:
                if (overplot_options['contour'][contour_obj_keys]['Plot']):   
                    try:
                        xy_object_name=overplot_options['contour'][contour_obj_keys]['Data object']
                        xy_object=flap.get_data_object(xy_object_name,exp_id=d.exp_id)
                    except:
                        raise ValueError(xy_object_name+'data is not available. The data object needs to be read first.')  
                    if 'Slicing' in overplot_options['path'][path_obj_keys]:
                        try:
                            xy_object.slice_data(slicing=overplot_options['path'][path_obj_keys]['Slicing'])
                        except:
                            raise ValueError('Slicing did not succeed. Try it outside the plotting!')
                        
                    if len(xy_object.data.shape) != 2:
                        raise ValueError('Contour object\'s, ('+xy_object_name+') data needs to be a 2D matrix.')

                    unit_conversion_coeff=[1.] * 2
                    for index_data_coordinate in range(len(d.coordinates)):
                        for index_oplot_coordinate in range(len(xy_object.coordinates)):
                            for index_axes in range(2):
                                if ((d.coordinates[index_data_coordinate].unit.name == axes[index_axes]) and
                                   (xy_object.coordinates[index_oplot_coordinate].unit.name) == axes[index_axes]):                
                                    unit_conversion_coeff[index_axes] = flap.tools.unit_conversion(original_unit=xy_object.coordinates[index_oplot_coordinate].unit.unit,
                                                                                                   new_unit=d.coordinates[index_data_coordinate].unit.unit)

                    overplot_options['contour'][contour_obj_keys]['data']={}
                    overplot_options['contour'][contour_obj_keys]['data']['Data']=xy_object.data
                    overplot_options['contour'][contour_obj_keys]['data']['X coord']=xy_object.coordinate(axes[0])[0]*unit_conversion_coeff[0]
                    overplot_options['contour'][contour_obj_keys]['data']['Y coord']=xy_object.coordinate(axes[1])[0]*unit_conversion_coeff[1]
        #TIME (AXES(2)) DEPENDENT OVERPLOTTING OBJECT CREATION
        if plot_type in ['anim-image','anim-contour','animation'] :
            for index_time in range(len(d.coordinates)):
                if (d.coordinates[index_time].unit.name == axes[2]):
                    if len(d.coordinates[index_time].dimension_list) != 1:
                        raise ValueError('The time coordinate is changing along multiple dimensions.')
                    time_dimension_data=d.coordinates[index_time].dimension_list[0]
                    data_time_index_coordinate=index_time
                    
            time_index=[0]*len(d.coordinate(axes[2])[0].shape)
            time_index[time_dimension_data]=Ellipsis
            tdata=d.coordinate(axes[2])[0][tuple(time_index)]
                    
            for path_obj_keys in overplot_options['path']:
                if (overplot_options['path'][path_obj_keys]['Plot']):              
                    try:
                        x_object_name=overplot_options['path'][path_obj_keys]['Data object X']
                        x_object=flap.get_data_object_ref(x_object_name,exp_id=d.exp_id)
                    except:
                        raise ValueError("The object "+x_object_name+" cannot be read.")
                    try:
                        y_object_name=overplot_options['path'][path_obj_keys]['Data object Y']
                        y_object=flap.get_data_object_ref(y_object_name,exp_id=d.exp_id)
                    except:
                        raise ValueError("The object "+y_object_name+" cannot be read.")
                    #Error handling
                    if (len(x_object.data.shape) != 2 or len(y_object.data.shape) != 2):
                        raise ValueError("The "+overplot_options['path'][path_obj_keys]['Data object X']+' or '
                                         "the "+overplot_options['path'][path_obj_keys]['Data object Y']+" data is not 1D.")    
                        
                    for index_coordinate in range(len(x_object.coordinates)):
                        if (x_object.coordinates[index_coordinate].unit.name == axes[2]):
                            oplot_x_time_index_coordinate=index_coordinate
                    for index_coordinate in range(len(x_object.coordinates)):
                        if (x_object.coordinates[index_coordinate].unit.name == axes[2]):
                            oplot_y_time_index_coordinate=index_coordinate
                            
                    if (d.coordinates[data_time_index_coordinate].unit.unit != x_object.coordinates[oplot_x_time_index_coordinate].unit.unit or
                        d.coordinates[data_time_index_coordinate].unit.unit != x_object.coordinates[oplot_y_time_index_coordinate].unit.unit):
                        raise TypeError('The '+axes[2]+' unit of the overplotted contour object differs from the data\'s. Interpolation cannot be done.')
                        
                    #Interpolate the path data to the time vector of the original data
                    try:
                        x_object_interp=flap.slice_data(x_object_name,
                                                        exp_id=d.exp_id,
                                                        slicing={axes[2]:tdata},
                                                        options={'Interpolation':'Linear'},
                                                        output_name='X OBJ INTERP')                    
                    except:
                        raise ValueError('Interpolation cannot be done for the \'Data object X\' along axis '+axes[2]+'.')    
                    try:
                        y_object_interp=flap.slice_data(y_object_name,
                                                        exp_id=d.exp_id,
                                                        slicing={axes[2]:tdata},
                                                        options={'Interpolation':'Linear'},
                                                        output_name='Y OBJ INTERP')
                    except:
                        raise ValueError('Interpolation cannot be done for the \'Data object Y\' along axis '+axes[2]+'.')
                    
                    #Finding the time coordinate
                    for index_time in range(len(x_object.coordinates)):
                        if (x_object.coordinates[index_time].unit.name == axes[2]):
                            time_dimension_oplot=x_object.coordinates[index_time].dimension_list[0]
                            
                    #Convert the units of the path if its coordinates are not the same as the data object's.
                    unit_conversion_coeff=[1]*2
                    original_units=[x_object.data_unit.unit,y_object.data_unit.unit]
                    for index_coordinate in range(len(d.coordinates)):
                        for index_axes in range(2):
                            if (d.coordinates[index_coordinate].unit.name == axes[index_axes]):                
                                unit_conversion_coeff[index_axes] = flap.tools.unit_conversion(original_unit=original_units[index_axes],
                                                                                               new_unit=d.coordinates[index_coordinate].unit.unit)
                    overplot_options['path'][path_obj_keys]['data']={}
                    overplot_options['path'][path_obj_keys]['data']['Data resampled']=np.asarray([x_object_interp.data*unit_conversion_coeff[0],
                                                                                                  y_object_interp.data*unit_conversion_coeff[1]])
                    overplot_options['path'][path_obj_keys]['data']['Time dimension']=time_dimension_oplot
                                
            for contour_obj_keys in overplot_options['contour']:
                if (overplot_options['contour'][contour_obj_keys]['Plot']):   
                    try:
                        xy_object_name=overplot_options['contour'][contour_obj_keys]['Data object']
                        xy_object=flap.get_data_object(xy_object_name,exp_id=d.exp_id)
                    except:
                        raise ValueError(xy_object_name+'data is not available. The data object needs to be read first.')  
                    if len(xy_object.data.shape) != 3:
                        raise ValueError('Contour object\'s, ('+xy_object_name+') data needs to be a 3D matrix (x,y,temporal), not necessarily in this order.')
                        
                    for index_coordinate in range(len(xy_object.coordinates)):
                        if (xy_object.coordinates[index_coordinate].unit.name == axes[2]):
                            time_dimension_index=xy_object.coordinates[index_coordinate].dimension_list[0]
                            oplot_time_index_coordinate=index_coordinate
                            
                    if d.coordinates[data_time_index_coordinate].unit.unit != xy_object.coordinates[oplot_time_index_coordinate].unit.unit:
                        raise TypeError('The '+axes[2]+' unit of the overplotted contour object differs from the data\'s. Interpolation cannot be done.')
                    
                    unit_conversion_coeff=[1.] * 3
                    for index_data_coordinate in range(len(d.coordinates)):
                        for index_oplot_coordinate in range(len(xy_object.coordinates)):
                            for index_axes in range(3):
                                if ((d.coordinates[index_data_coordinate].unit.name == axes[index_axes]) and
                                   (xy_object.coordinates[index_oplot_coordinate].unit.name) == axes[index_axes]):                
                                    unit_conversion_coeff[index_axes] = flap.tools.unit_conversion(original_unit=xy_object.coordinates[index_oplot_coordinate].unit.unit,
                                                                                                   new_unit=d.coordinates[index_data_coordinate].unit.unit)
                                    
                    xy_object_interp=flap.slice_data(xy_object_name,
                                                     exp_id=d.exp_id,
                                                     slicing={axes[2]:tdata},
                                                     options={'Interpolation':'Linear'},
                                                     output_name='XY OBJ INTERP')
                    overplot_options['contour'][contour_obj_keys]['data']={}
                    overplot_options['contour'][contour_obj_keys]['data']['Data resampled']=xy_object_interp.data
                    overplot_options['contour'][contour_obj_keys]['data']['X coord resampled']=xy_object_interp.coordinate(axes[0])[0]*unit_conversion_coeff[0]
                    overplot_options['contour'][contour_obj_keys]['data']['Y coord resampled']=xy_object_interp.coordinate(axes[1])[0]*unit_conversion_coeff[1]
                    overplot_options['contour'][contour_obj_keys]['data'][axes[2]]=tdata
                    overplot_options['contour'][contour_obj_keys]['data']['Time dimension']=time_dimension_index

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

    """ ---------------------------------------------
        |       XY AND SCATTER PLOT DEFINITION      |
        ---------------------------------------------"""

    if ((_plot_type == 'xy') or (_plot_type == 'scatter')):
        # 1D plots: xy, scatter and complex versions
        # Checking whether oveplotting into the same plot type
        if ((d.data is not None) and (d.data.dtype.kind == 'c')): 
            subtype = 1
        else:
            subtype = 0
        if (not _options['Clear']):
            if ((_plot_id.plot_type is not None) and ((_plot_id.plot_type != 'xy') and (_plot_id.plot_type != 'scatter')) 
                or (_plot_id.plot_subtype is not None) and (_plot_id.plot_subtype != subtype)):
                raise ValueError("Overplotting into different plot type. Use option={'Clear':True} to erase first.")
        #Processing axes
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
#        plotdata_low = [0]*2                                                    #UNUSED
#        plotdata_high = [0]*2                                                   #UNUSED
        ploterror = [0]*2
        for ax_ind in range(2):
            if (pdd_list[ax_ind].data_type == PddType.Data):
                if (len(pdd_list[ax_ind].data_object.shape) > 1):
                    raise ValueError("xy plot is applicable only to 1D data. Use slicing.")
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
                
            if xerr is not None:
                xerr_plot=xerr*axes_unit_conversion[0]
            else:
                xerr_plot=xerr
                
            if (_plot_type == 'xy'):                
                if (plot_error):
                    ax.errorbar(x*axes_unit_conversion[0],
                                y,
                                xerr=xerr_plot,
                                yerr=yerr,errorevery=errorevery,**_plot_opt)
                else:
                    ax.plot(x*axes_unit_conversion[0],
                            y,**_plot_opt)
            else:
                if (plot_error):
                    ax.errorbar(x*axes_unit_conversion[0],
                                y,
                                xerr=xerr_plot,
                                yerr=yerr,
                                errorevery=errorevery,fmt='o',**_plot_opt)
                else:
                    ax.scatter(x*axes_unit_conversion[0],
                               y,**_plot_opt)
                    
            if (_options['Log x']):
                ax.set_xscale('log')
            if (_options['Log y']):
                ax.set_yscale('log')
                
            if (xrange is not None):
                ax.set_xlim(xrange[0]*axes_unit_conversion[0],
                            xrange[1]*axes_unit_conversion[0])
                
            if (yrange is not None):
                ax.set_ylim(yrange[0],
                            yrange[1])
                
            if (overplot_options is not None):
                if overplot_options['line'] is not None:
                    for line_obj_keys in overplot_options['line']:
                        xmin, xmax = ax.get_xbound()
                        ymin, ymax = ax.get_ybound()
                        if overplot_options['line'][line_obj_keys]['Plot']:
                            if 'Horizontal' in overplot_options['line'][line_obj_keys]:
                                h_coords=overplot_options['line'][line_obj_keys]['Horizontal']
                                for segments in h_coords:
                                    if segments[0] > ymin and segments[0] < ymax:
                                        l = mlines.Line2D([xmin,xmax], [segments[0],segments[0]], color=segments[1])
                                        ax.add_line(l) 
                            if 'Vertical' in overplot_options['line'][line_obj_keys]:
                                v_coords=overplot_options['line'][line_obj_keys]['Vertical']
                                for segments in v_coords:
                                    if segments[0] > xmin and segments[0] < xmax:
                                        l = mlines.Line2D([segments[0],segments[0]], [ymin,ymax], color=segments[1])
                                        ax.add_line(l)
            # Setting axis labels    
            if axes_unit_conversion[0] == 1.:
                ax.set_xlabel(ax_list[0].title(language=language))
            else:
                ax.set_xlabel(ax_list[0].title(language=language, 
                                               new_unit=_options['Plot units'][axes[0]]))
                
            ax.set_ylabel(ax_list[1].title(language=language))

            ax.get_xaxis().set_visible(_options['Axes visibility'][0])
            ax.get_yaxis().set_visible(_options['Axes visibility'][1])
       
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
                if (np.isscalar(xdata)):
                    if (np.isscalar(ydata)):
                        xdata = np.full(1, xdata)
                    else:
                        xdata = np.full(ydata.size, xdata)
                    if (plot_error):
                        c=pdd_list[0].value #THESE THREE VARIABLES WERE UNDIFINED
                        xdata_low=np.min(xdata)
                        xdata_high=np.max(xdata)
                        xerror = d._plot_coord_ranges(c, xdata, xdata_low, xdata_high)
                    else:
                        xerror = None                
                if (np.isscalar(ydata)):
                    if (np.isscalar(xdata)):
                        ydata = np.full(1, ydata)
                    else:
                        ydata = np.full(xdata.size, ydata)
                    if (plot_error):
                        c=pdd_list[1].value #THESE THREE VARIABLES WERE UNDIFINED
                        ydata_low=np.min(ydata)
                        ydata_high=np.max(ydata)
                        yerror = d._plot_coord_ranges(c, ydata, ydata_low, ydata_high)
                    else:
                        yerror = None
        
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
                    
                if xerr is not None:
                    xerr_plot=xerr*axes_unit_conversion[0]
                else:
                    xerr_plot=xerr
                    
                if (_plot_type == 'xy'):                
                    if (plot_error):
                        ax.errorbar(x*axes_unit_conversion[0],
                                    y,
                                    xerr=xerr_plot,
                                    yerr=yerr,
                                    errorevery=errorevery,**_plot_opt)
                    else:
                        ax.plot(x*axes_unit_conversion[0],
                                y,**_plot_opt)
                else:
                    if (plot_error):
                        ax.errorbar(x*axes_unit_conversion[0],
                                    y,
                                    xerr=xerr_plot,
                                    yerr=yerr,
                                    errorevery=errorevery,fmt='o',**_plot_opt)
                    else:
                        ax.scatter(x*axes_unit_conversion[0],
                                   y,**_plot_opt)
    
                if (_options['Log x']):
                    ax.set_xscale('log')
                if (_options['Log y']):
                    ax.set_yscale('log')
 
                if (xrange is not None):
                    ax.set_xlim(xrange[0]*axes_unit_conversion[0],
                                xrange[1]*axes_unit_conversion[0])
                    
                if (yrange is not None):
                    ax.set_ylim(yrange[i_comp][0],yrange[i_comp][1])        
                if (overplot_options is not None):
                    if overplot_options['line'] is not None:
                        for line_obj_keys in overplot_options['line']:
                            xmin, xmax = ax.get_xbound()
                            ymin, ymax = ax.get_ybound()
                            
                            if overplot_options['line'][line_obj_keys]['Plot']:
                                if 'Horizontal' in overplot_options['line'][line_obj_keys]:
                                    h_coords=overplot_options['line'][line_obj_keys]['Horizontal']
                                    for segments in h_coords:
                                        if segments[0] > ymin and segments[0] < ymax:
                                            l = mlines.Line2D([xmin,xmax], [segments[0],segments[0]], color=segments[1])
                                            ax.add_line(l) 
                                            
                                if 'Vertical' in overplot_options['line'][line_obj_keys]:
                                    v_coords=overplot_options['line'][line_obj_keys]['Vertical']
                                    for segments in v_coords:
                                        if segments[0] > xmin and segments[0] < xmax:
                                            l = mlines.Line2D([segments[0],segments[0]], [ymin,ymax], color=segments[1])
                                            ax.add_line(l)
                    
                # Setting axis labels    
                if axes_unit_conversion[0] == 1.:
                    ax.set_xlabel(ax_list[0].title(language=language))
                else:
                    ax.set_xlabel(ax_list[0].title(language=language, 
                                                   new_unit=_options['Plot units'][axes[0]]))
                    

                ax.set_ylabel(ax_list[1].title(language=language), 
                                               complex_txt=[comptype,i_comp])
                ax.get_xaxis().set_visible(_options['Axes visibility'][0])
                ax.get_yaxis().set_visible(_options['Axes visibility'][1])
                
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

        """ ---------------------------------------------
            |          MULTIXY PLOT DEFINITION          |
            ---------------------------------------------"""
        
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
        if (ysep is None):
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
            if xerr is not None:
                xerr_plot=xerr*axes_unit_conversion[0]
            else:
                xerr_plot=xerr
            if (plot_error):
                ax.errorbar(x*axes_unit_conversion[0],
                            y,
                            xerr=xerr_plot,
                            yerr=yerr,
                            errorevery=errorevery,**_plot_opt)
            else:
                ax.plot(x*axes_unit_conversion[0],
                        y,
                        **_plot_opt)
        
        if (xrange is not None):
            ax.set_xlim(xrange[0]*axes_unit_conversion[0],
                        xrange[1]*axes_unit_conversion[0])
            
        if (yrange is not None):
            ax.set_ylim(yrange[0],yrange[1])
        if (overplot_options is not None):
            if overplot_options['line'] is not None:
                for line_obj_keys in overplot_options['line']:
                    xmin, xmax = ax.get_xbound()
                    ymin, ymax = ax.get_ybound()
                    if overplot_options['line'][line_obj_keys]['Plot']:
                        if 'Horizontal' in overplot_options['line'][line_obj_keys]:
                            h_coords=overplot_options['line'][line_obj_keys]['Horizontal']
                            for segments in h_coords:
                                if segments[0] > ymin and segments[0] < ymax:
                                    l = mlines.Line2D([xmin,xmax], [segments[0],segments[0]], color=segments[1])
                                    ax.add_line(l) 
                        if 'Vertical' in overplot_options['line'][line_obj_keys]:
                            v_coords=overplot_options['line'][line_obj_keys]['Vertical']
                            for segments in v_coords:
                                if segments[0] > xmin and segments[0] < xmax:
                                    l = mlines.Line2D([segments[0],segments[0]], [ymin,ymax], color=segments[1])
                                    ax.add_line(l)
            
        if axes_unit_conversion[0] == 1.:
            ax.set_xlabel(ax_list[0].title(language=language))
        else:
            ax.set_xlabel(ax_list[0].title(language=language, 
                                           new_unit=_options['Plot units'][axes[0]]))
            
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

        """ ---------------------------------------------
            |    IMAGE AND CONTOUR PLOT DEFINITION      |
            ---------------------------------------------"""

    elif ((_plot_type == 'image') or (_plot_type == 'contour')):
        if (d.data is None):
            raise ValueError("Cannot plot DataObject without data.")
        if (len(d.shape) != 2):
            raise TypeError("Image/contour plot is applicable to 2D data only. Use slicing.")
        if (d.data.dtype.kind == 'c'):
            raise TypeError("Image/contour plot is applicable only to real data.")
        # Checking for numeric type
        try:
            d.data[0,0]+1
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
        if _options['Equal axes']:
            if (coord_x.unit.unit == coord_y.unit.unit):
                #ax.axis('equal')
                ax.set_aspect(1.0)
            else:
                print('Equal axis is not possible. The axes units are not equal.')
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
        else:
            image_like = False
        if (image_like):        
            xdata_range = coord_x.data_range(data_shape=d.shape)[0]
            ydata_range = coord_y.data_range(data_shape=d.shape)[0]
            xdata_range = [axes_unit_conversion[0] * xdata_range[0],
                           axes_unit_conversion[0] * xdata_range[1]]
            ydata_range = [axes_unit_conversion[1] * ydata_range[0],
                           axes_unit_conversion[1] * ydata_range[1]]
        else:
            xdata,xdata_low,xdata_high = coord_x.data(data_shape=d.shape)
            ydata,ydata_low,ydata_high = coord_y.data(data_shape=d.shape)
            
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
            ticker.LogLocator(subs='all')
            #locator = ticker.LogLocator(subs='all')                            #VARIABLE WAS UNUSED --> UNASSIGNED NOW
        else:
            norm = None
            #locator = None
                
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
#                if (coord_x.dimension_list[0] == 0):
                if (coord_x.dimension_list[0] < coord_y.dimension_list[0]):
                    im=np.clip(np.transpose(d.data),vmin,vmax)
                else:
                    im=np.clip(d.data,vmin,vmax)
                img = ax.imshow(im,extent=xdata_range + ydata_range,norm=norm,
                                cmap=cmap_obj,vmin=vmin,aspect=_options['Aspect ratio'],interpolation=_options['Interpolation'],
                                vmax=vmax,origin='lower',**_plot_opt)
                del im                    
            except Exception as e:
                raise e
        else:
            if (_plot_type == 'image'):
                xgrid, ygrid = flap.tools.grid_to_box(xdata*axes_unit_conversion[0],
                                                      ydata*axes_unit_conversion[1])
                try:
                    img = ax.pcolormesh(xgrid,ygrid,
                                        np.clip(np.transpose(d.data),vmin,vmax),
                                        norm=norm,cmap=cmap,vmin=vmin,
                                        vmax=vmax,**_plot_opt)
                except Exception as e:
                    raise e
            else:
                try:
                    img = ax.contourf(xdata*axes_unit_conversion[0],
                                      ydata*axes_unit_conversion[1],
                                      np.clip(d.data,vmin,vmax),
                                      contour_levels,norm=norm,
                                      origin='lower',cmap=cmap,
                                      vmin=vmin,vmax=vmax,**_plot_opt)
                except Exception as e:
                    raise e                    
                    
        if (overplot_options is not None):
            ax.set_autoscale_on(False)
            for path_obj_keys in overplot_options['path']:
                if overplot_options['path'][path_obj_keys]['Plot']:
                    ax.plot(overplot_options['path'][path_obj_keys]['data']['Data'][0,:]*axes_unit_conversion[0],
                            overplot_options['path'][path_obj_keys]['data']['Data'][1,:]*axes_unit_conversion[1],
                            color=overplot_options['path'][path_obj_keys]['Color'])
            
            for contour_obj_keys in overplot_options['contour']:
                if overplot_options['contour'][contour_obj_keys]['Plot']:
                    ax.contour(overplot_options['contour'][contour_obj_keys]['data']['X coord'].transpose()*axes_unit_conversion[0],
                               overplot_options['contour'][contour_obj_keys]['data']['Y coord'].transpose()*axes_unit_conversion[1],
                               overplot_options['contour'][contour_obj_keys]['data']['Data'],
                               levels=overplot_options['contour'][contour_obj_keys]['nlevel'],
                               cmap=overplot_options['contour'][contour_obj_keys]['Colormap'])
            for line_obj_keys in overplot_options['line']:
                xmin, xmax = ax.get_xbound()
                ymin, ymax = ax.get_ybound()
                if overplot_options['line'][line_obj_keys]['Plot']:
                    if 'Horizontal' in overplot_options['line'][line_obj_keys]:
                        h_coords=overplot_options['line'][line_obj_keys]['Horizontal']
                        for segments in h_coords:
                            if segments[0] > ymin and segments[0] < ymax:
                                l = mlines.Line2D([xmin,xmax], [segments[0],segments[0]], color=segments[1])
                                ax.add_line(l) 
                    if 'Vertical' in overplot_options['line'][line_obj_keys]:
                        v_coords=overplot_options['line'][line_obj_keys]['Vertical']
                        for segments in v_coords:
                            if segments[0] > xmin and segments[0] < xmax:
                                l = mlines.Line2D([segments[0],segments[0]], [ymin,ymax], color=segments[1])
                                ax.add_line(l)
            
        if (_options['Colorbar']):
            cbar = plt.colorbar(img,ax=ax)
            if (d.data_unit.unit is not None) and (d.data_unit.unit != ''):
                unit_name = '['+d.data_unit.unit+']'
            else:
                unit_name = ''
                cbar.set_label(d.data_unit.name+' '+unit_name)
        
        if (xrange is not None):
            ax.set_xlim(xrange[0]*axes_unit_conversion[0],
                        xrange[1]*axes_unit_conversion[0])
            
        if (yrange is not None):
            ax.set_ylim(yrange[0]*axes_unit_conversion[1],
                        yrange[1]*axes_unit_conversion[1])
            
        if axes_unit_conversion[0] == 1.:
            ax.set_xlabel(ax_list[0].title(language=language))
        else:
            ax.set_xlabel(ax_list[0].title(language=language, 
                          new_unit=_options['Plot units'][axes[0]]))
            
        if axes_unit_conversion[1] == 1.:
            ax.set_ylabel(ax_list[1].title(language=language))
        else:
            ax.set_ylabel(ax_list[1].title(language=language, 
                          new_unit=_options['Plot units'][axes[1]]))
        ax.get_xaxis().set_visible(_options['Axes visibility'][0])
        ax.get_yaxis().set_visible(_options['Axes visibility'][1])    
        
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

        """ -----------------------------------------------
            | ANIM-IMAGE AND ANIM-CONTOUR PLOT DEFINITION |
            -----------------------------------------------"""
                
    elif ((_plot_type == 'anim-image') or (_plot_type == 'anim-contour')):
        if (d.data is None):
            raise ValueError("Cannot plot DataObject without data.")
        if (len(d.shape) != 3):
            raise TypeError("Animated image plot is applicable to 3D data only. Use slicing.")
        if (d.data.dtype.kind == 'c'):
            raise TypeError("Animated image plot is applicable only to real data.")
        # Checking for numeric type
        try:
            d.data[0,0] + 1
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
            raise ValueError("Time coordinate of anim-image/anim-contour plot should be flap.coordinate.")

        coord_x = pdd_list[0].value
        coord_y = pdd_list[1].value
        coord_t = pdd_list[2].value
        
        if (len(coord_t.dimension_list) != 1):
            raise ValueError("Time coordinate for anim-image/anim-contour plot should be changing only along one dimension.")
            
        index = [0] * 3
        index[coord_t.dimension_list[0]] = ...
        tdata = coord_t.data(data_shape=d.shape,index=index)[0].flatten()
        if (not coord_y.isnumeric()):
            raise ValueError('Coordinate '+coord_y.unit.name+' is not numeric.')

        if ((coord_x.mode.equidistant) and (len(coord_x.dimension_list) == 1) and
            (coord_y.mode.equidistant) and (len(coord_y.dimension_list) == 1)):
            # This data is image-like with data points on a rectangular array
            image_like = True
            xdata=coord_x.data(data_shape=d.data.shape)[0]
            ydata=coord_y.data(data_shape=d.data.shape)[0]
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
        if (image_like and (_plot_type == 'anim-image')):        
            xdata_range = coord_x.data_range(data_shape=d.shape)[0]
            ydata_range = coord_y.data_range(data_shape=d.shape)[0]
            xdata_range = [axes_unit_conversion[0] * xdata_range[0],
                           axes_unit_conversion[0] * xdata_range[1]]
            ydata_range = [axes_unit_conversion[1] * ydata_range[0],
                           axes_unit_conversion[1] * ydata_range[1]]
        else:
            index = [...]*3
            if (len(coord_x.dimension_list) < 3 and 
                len(coord_y.dimension_list) < 3):
                index[coord_t.dimension_list[0]] = 0
            ydata = np.squeeze(coord_y.data(data_shape=d.shape,index=index)[0])
            xdata = np.squeeze(coord_x.data(data_shape=d.shape,index=index)[0])
        try:
            cmap_obj = plt.cm.get_cmap(cmap)
            if (_options['Nan color'] is not None):
                cmap_obj.set_bad(_options['Nan color'])
        except ValueError:
            raise ValueError("Invalid color map.")

        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=_plot_id.base_subplot)
        _plot_id.plt_axis_list = []
#        _plot_id.plt_axis_list.append(plt.subplot(gs[0,0]))
        _plot_id.plt_axis_list.append(_plot_id.base_subplot)
#        plt.subplot(_plot_id.base_subplot)
#        plt.plot()
#        plt.cla()
#        ax=plt.gca()

        if (xrange is None):
            xrange=[np.min(xdata),np.max(xdata)]
        if (yrange is None):
            yrange=[np.min(ydata),np.max(ydata)]
            
        if _options['Equal axes'] and not coord_x.unit.unit == coord_y.unit.unit:
            print('Equal axis is not possible. The axes units are not equal.')            

        for it in range(len(tdata)):
            # This is a hack here. The problem is, that the colorbar() call reduces the axes size
            del gs
            gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=_plot_id.base_subplot)
            plt.subplot(gs[0,0])
#            plt.subplot(_plot_id.base_subplot)
#            ax_act = _plot_id.base_subplot
            ax_act = plt.subplot(gs[0,0])
            if _options['Equal axes'] and coord_x.unit.unit == coord_y.unit.unit:
                #ax_act.axis('equal')
                ax_act.set_aspect(1.0)
            time_index = [slice(0,dim) for dim in d.data.shape]
            time_index[coord_t.dimension_list[0]] = it
            time_index = tuple(time_index)
    
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
                #locator = ticker.LogLocator(subs='all')
                ticker.LogLocator(subs='all')
            else:
                norm = None
                #locator = None
                    
            if (contour_levels is None):
                contour_levels = 255
            ax_act.clear()
            plt.xlim(xrange[0]*axes_unit_conversion[0],
                     xrange[1]*axes_unit_conversion[0])
            plt.ylim(yrange[0]*axes_unit_conversion[1],
                     yrange[1]*axes_unit_conversion[1])
                
            _plot_opt = _plot_options[0]
            if (image_like and (_plot_type == 'anim-image')):
                try:
                    if (coord_x.dimension_list[0] < coord_y.dimension_list[0]):
                        im = np.clip(np.transpose(d.data[time_index]),vmin,vmax)
                    else:
                        im = np.clip(d.data[time_index],vmin,vmax)
                        
                    img = plt.imshow(im,extent=xdata_range + ydata_range,norm=norm,
                                     cmap=cmap_obj,vmin=vmin,vmax=vmax,
                                     aspect=_options['Aspect ratio'],
                                     interpolation=_options['Interpolation'],
                                     origin='lower',**_plot_opt)            
                    del im
                except Exception as e:
                    raise e
            else:
                ax_act.set_autoscale_on(False)
                if (_plot_type == 'anim-image'):
                    #xgrid, ygrid = flap.tools.grid_to_box(xdata*axes_unit_conversion[0],
                    #                                      ydata*axes_unit_conversion[1])
                    if (len(xdata.shape) == 3 and len(ydata.shape) == 3):
                        xgrid, ygrid = flap.tools.grid_to_box(xdata[time_index]*axes_unit_conversion[0],
                                                              ydata[time_index]*axes_unit_conversion[1])
                    else:
                        xgrid, ygrid = flap.tools.grid_to_box(xdata*axes_unit_conversion[0],
                                                              ydata*axes_unit_conversion[1])
                    im = np.clip(np.transpose(d.data[time_index]),vmin,vmax)
                    try:
                        img = plt.pcolormesh(xgrid,ygrid,im,norm=norm,cmap=cmap,vmin=vmin,
                                             vmax=vmax,**_plot_opt)
                    except Exception as e:
                        raise e
                    del im
                else:
                    try:
                        if (len(xdata.shape) == 3 and len(ydata.shape) == 3):
                            xgrid=xdata[time_index]*axes_unit_conversion[1]
                            ygrid=ydata[time_index]*axes_unit_conversion[1]
                        else:
                            xgrid=xdata*axes_unit_conversion[0]
                            ygrid=ydata*axes_unit_conversion[1]
                        im = np.clip(d.data[time_index],vmin,vmax)
                        img = plt.contourf(xgrid,
                                           ygrid,
                                           im,
                                           contour_levels,norm=norm,origin='lower',
                                           cmap=cmap,vmin=vmin,vmax=vmax,**_plot_opt)
                        del im
                    except Exception as e:
                        raise e
                        
            if (overplot_options is not None):
                ax_act.set_autoscale_on(False)
                for path_obj_keys in overplot_options['path']:
                    if overplot_options['path'][path_obj_keys]['Plot']:
                        time_index = [slice(0,dim) for dim in overplot_options['path'][path_obj_keys]['data']['Data resampled'][0].shape]
                        time_index[overplot_options['path'][path_obj_keys]['data']['Time dimension']] = it
                        time_index = tuple(time_index)
                        im = plt.plot(overplot_options['path'][path_obj_keys]['data']['Data resampled'][0][time_index]*axes_unit_conversion[0],
                                      overplot_options['path'][path_obj_keys]['data']['Data resampled'][1][time_index]*axes_unit_conversion[1],
                                      color=overplot_options['path'][path_obj_keys]['Color'])
                
                for contour_obj_keys in overplot_options['contour']:
                    if overplot_options['contour'][contour_obj_keys]['Plot']:
                        time_index = [slice(0,dim) for dim in overplot_options['contour'][contour_obj_keys]['data']['Data resampled'].shape]
                        time_index[overplot_options['contour'][contour_obj_keys]['data']['Time dimension']] = it
                        time_index = tuple(time_index)
                        im = plt.contour(overplot_options['contour'][contour_obj_keys]['data']['X coord resampled'][time_index].transpose()*axes_unit_conversion[0],
                                         overplot_options['contour'][contour_obj_keys]['data']['Y coord resampled'][time_index].transpose()*axes_unit_conversion[1],
                                         overplot_options['contour'][contour_obj_keys]['data']['Data resampled'][time_index],
                                         levels=overplot_options['contour'][contour_obj_keys]['nlevel'],
                                         cmap=overplot_options['contour'][contour_obj_keys]['Colormap'])
            
                for line_obj_keys in overplot_options['line']:
                    xmin, xmax = ax_act.get_xbound()
                    ymin, ymax = ax_act.get_ybound()
                    if overplot_options['line'][line_obj_keys]['Plot']:
                        if 'Horizontal' in overplot_options['line'][line_obj_keys]:
                            h_coords=overplot_options['line'][line_obj_keys]['Horizontal']
                            for segments in h_coords:
                                if segments[0] > ymin and segments[0] < ymax:
                                    l = mlines.Line2D([xmin,xmax], [segments[0],segments[0]], color=segments[1])
                                    ax_act.add_line(l) 
                        if 'Vertical' in overplot_options['line'][line_obj_keys]:
                            v_coords=overplot_options['line'][line_obj_keys]['Vertical']
                            for segments in v_coords:
                                if segments[0] > xmin and segments[0] < xmax:
                                    l = mlines.Line2D([segments[0],segments[0]], [ymin,ymax], color=segments[1])
                                    ax_act.add_line(l)
                    
            if (_options['Colorbar']):
                cbar = plt.colorbar(img,ax=ax_act)
                cbar.set_label(d.data_unit.name)
                
            if axes_unit_conversion[0] == 1.:
                plt.xlabel(ax_list[0].title(language=language))
            else:
                plt.xlabel(ax_list[0].title(language=language, 
                                            new_unit=_options['Plot units'][axes[0]]))
                
            if axes_unit_conversion[1] == 1.:
                plt.ylabel(ax_list[1].title(language=language))
            else:
                plt.ylabel(ax_list[1].title(language=language, 
                                            new_unit=_options['Plot units'][axes[1]]))
                
            if (_options['Log x']):
                plt.xscale('log')
            if (_options['Log y']):
                plt.yscale('log')
            
            if _options['Plot units'] is not None:
                if axes[2] in _options['Plot units']:
                    time_unit=_options['Plot units'][axes[2]]
                    time_coeff=axes_unit_conversion[2]
                else:
                    time_unit=coord_t.unit.unit
                time_coeff=1.
            else:
                time_unit=coord_t.unit.unit
                time_coeff=1.
            title = str(d.exp_id)+' @ '+coord_t.unit.name+'='+"{:10.5f}".format(tdata[it]*time_coeff)+\
                    ' ['+time_unit+']'
                
            plt.title(title)
            plt.show(block=False)
            time.sleep(_options['Waittime'])
            plt.pause(0.001)
            if ((_options['Video file'] is not None) and (cv2_presence is not False)):
                fig = plt.gcf()
                fig.canvas.draw()
                # Get the RGBA buffer from the figure
                w,h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                if buf.shape[0] == h*2 * w*2 * 3:
                    buf.shape = ( h*2, w*2, 3 ) #ON THE MAC'S INTERNAL SCREEN, THIS COULD HAPPEN NEEDS A MORE ELEGANT FIX
                else:
                    buf.shape = ( h, w, 3 )
                buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
                try:
                    video
                except NameError:
                    height = buf.shape[0]
                    width = buf.shape[1]
                    video = cv2.VideoWriter(_options['Video file'],  
                                            cv2.VideoWriter_fourcc(*video_codec_code), 
                                            float(_options['Video framerate']), 
                                            (width,height),
                                            isColor=True)
                video.write(buf)
        if ((_options['Video file'] is not None) and (cv2_presence is not False)):
            cv2.destroyAllWindows()
            video.release()  
            del video

        """ ---------------------------------------------
            |         ANIMATION PLOT DEFINITION         |
            ---------------------------------------------"""
            
    elif (_plot_type == 'animation'):
        if (d.data is None):
            raise ValueError("Cannot plot DataObject without data.")
        if (len(d.shape) != 3):
            raise TypeError("Animated image plot is applicable to 3D data only. Use slicing.")
        if (d.data.dtype.kind == 'c'):
            raise TypeError("Animated image plot is applicable only to real data.")
        # Checking for numeric type
        try:
            d.data[0,0]+1
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
            raise ValueError("For the animation plot only data can be plotted on the z axis.")
        if ((pdd_list[0].data_type != PddType.Coordinate) or (pdd_list[1].data_type != PddType.Coordinate)) :
            raise ValueError("X and y coordinates of the animation plot type should be coordinates.")
        if (pdd_list[2].data_type != PddType.Coordinate) :
            raise ValueError("Time coordinate of the animation plot should be flap.coordinate.")

        coord_x = pdd_list[0].value
        coord_y = pdd_list[1].value
        coord_t = pdd_list[2].value
        
        if (len(coord_t.dimension_list) != 1):
            raise ValueError("Time coordinate for anim-image/anim-contour plot should be changing only along one dimension.")
            
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
            ydata = np.squeeze(coord_y.data(data_shape=d.shape,index=index)[0])
            xdata = np.squeeze(coord_x.data(data_shape=d.shape,index=index)[0])
        else:
            index = [...]*3
            if (len(coord_x.dimension_list) < 3 and 
                len(coord_y.dimension_list) < 3):
                index[coord_t.dimension_list[0]] = 0
            xdata_range = None
            ydata_range = None
            ydata = np.squeeze(coord_y.data(data_shape=d.shape,index=index)[0])
            xdata = np.squeeze(coord_x.data(data_shape=d.shape,index=index)[0])
        try:
            cmap_obj = plt.cm.get_cmap(cmap)
            if (_options['Nan color'] is not None):
                cmap_obj.set_bad(_options['Nan color'])
        except ValueError:
            raise ValueError("Invalid color map.")
        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=_plot_id.base_subplot)

        _plot_id.plt_axis_list = []
        _plot_id.plt_axis_list.append(plt.subplot(gs[0,0]))

        oargs=(ax_list, axes, axes_unit_conversion, d, xdata, ydata, tdata, xdata_range, ydata_range,
               cmap_obj, contour_levels, 
               coord_t, coord_x, coord_y, cmap, _options, overplot_options,
               xrange, yrange, zrange, image_like, 
               _plot_options, language, _plot_id, gs)
        
        anim = PlotAnimation(*oargs)
        anim.animate()
        
    plt.show(block=False)
    _plot_id.axes = ax_list
    _plot_id.plot_data.append(pdd_list)
    #Setting this one the default plot ID
    _plot_id.options.append(_options)
    _plot_id.plot_type = _plot_type
    set_plot_id(_plot_id)
    if _options['Prevent saturation']:
        del d
    return _plot_id
