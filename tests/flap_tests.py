# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 22:11:28 2018

@author: Sandor Zoletnik  (zoletnik.sandor@ek-cer.hu)
Centre for Energy Research

Example/test programs for FLAP.
Will run various tests and print/plot the result. 
As default all testst will be run. 
If you don't want all tests look for "test_all = True" and change True to False
Below that you can set each test step to True/False to run them separately.

"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
from scipy import signal
import copy
import time
import math
from tkinter import Tk  # Needed for getting the screen size
import random as ran

# Importing the FLAP
import flap
# Importing the testdata module which is part of the FLAP distribution
import flap.testdata
# Registering the TESTDATA data source
flap.testdata.register()

def test_config():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Testing configuration file <<<<<<<<<<<<<<<<<<<<<<<< ")
    #Reading a single element
    txt = flap.config.get('General','Test_txt',default='')
    print("[General]Test_txt: {:s} (type:{:s})".format(txt,str(type(txt))))
    txt = flap.config.interpret_config_value(txt)
    print("After flap.config.interpret_config_value: {:s} (type:{:s})".format(txt,str(type(txt))))
    txt = flap.config.get('General','Test_bool',default='')
    print("[General]Test_bool: {:s} (type:{:s})".format(txt,str(type(txt))))
    txt = flap.config.interpret_config_value(txt)
    print("After flap.config.interpret_config_value: {:} (type:{:s})".format(txt,str(type(txt))))
    
    print()
    default_options = {"Test_para1":"p1","Test_para2":"p2","Opt3":"opt3"}
    print("Default options: %", default_options)
    _options = flap.config.merge_options(default_options,
                                         {"Op":"Input opt3"},
                                         section="TESTSECTION", 
                                         data_source='TESTDATA')
    print("Merged options: %",_options)
    
def test_storage(signals='TEST-*',timerange=[0,0.001]):
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Test storage operations on test data <<<<<<<<<<<<<<<<<<<<<<<<")
    flap.delete_data_object('*',exp_id='*')
    if (type(signals) is list):
        s = "["
        for sig in signals:
            s += sig+" "
        s += "]"
    else:
        s = signals
    print("**** Reading signal "+s+" for time range ["+str(timerange[0])+'-'+str(timerange[1])+'] with no_data=True')
    d=flap.get_data('TESTDATA',name=signals,
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':timerange},
                    no_data = True)
    print("**** Storage contents")
    flap.list_data_objects()
    print()
    print("**** Reading the same with data")
    d=flap.get_data('TESTDATA',name=signals,
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':timerange})
    print("**** Storage contents")
    flap.list_data_objects()

def test_testdata():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Test TESTDATA data source <<<<<<<<<<<<<<<<<<<<<<<<")
    flap.delete_data_object('*',exp_id='*')
    print("**** Generating 0.01 s long test data signals on [4,5] matrix with fixed frequency changing from channel to channel")
    d=flap.get_data('TESTDATA',
                    name='TEST-*-*',
                    options={'Scaling':'Volt',
                              'Frequency':[1e3,1e4],
                              'Length':1e-2,
                              'Row number':4,
                              'Column number':5
                              },
                    object_name='TESTDATA'
                    )
    plt.figure()
    print("**** Plotting row 2")
    d.plot(slicing={'Row':2},axes=['Time'])
    

    print("**** Generating 0.01 s long test data signal with linearly changing frequency: 10-100kHz")   
    f = np.linspace(1e4,1e5,num=11)
    coord = flap.Coordinate(name='Time',
                            start=0.0,
                            step=0.001,
                            mode=flap.CoordinateMode(equidistant=True),
                            dimension_list=[0]
                            )
    f_obj = flap.DataObject(data_array=f,
                            coordinates=[coord],
                            data_unit=flap.Unit(name='Frequency',unit='Hz')
                            )
    flap.list_data_objects(f_obj)
    d=flap.get_data('TESTDATA',
                    name='TEST-1-1',
                    options={'Scaling':'Volt',
                             'Frequency':f_obj,
                             'Length':0.01,
                             'Row number':1,
                             'Column number':1
                             },
                    object_name='TESTDATA'
                    )
    plt.figure()
    d.plot(axes='Time',options={'All':True})  
    
def test_saveload():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Test save/load <<<<<<<<<<<<<<<<<<<<<<<<")
    flap.delete_data_object('*')
    print("**** Storage contents before save.")
    flap.list_data_objects()
    print("**** Saving all storage and deleting storage contents.")
    flap.save('*','flap_save_test.dat')
    flap.delete_data_object('*')
    print("**** Storage contents after erasing.")
    flap.list_data_objects()
    flap.load('flap_save_test.dat')
    print("**** Storage contents after loading.")
    flap.list_data_objects()
    flap.delete_data_object('*')
    print("**** Storage contents after erasing.")
    d = flap.load('flap_save_test.dat',options={'No':True})
    print(d)
    print("**** Storage contents after loading with 'No storage' option.")
    flap.list_data_objects()
    flap.delete_data_object('*')
    flap.save([d,'test'],'flap_save_test.dat')
    d1 = flap.load('flap_save_test.dat',options={'No':True})
    print(d1)

def test_coordinates():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Testing adding coordinates <<<<<<<<<<<<<<<<<<<<<<<<")
    flap.delete_data_object('*',exp_id='*')

    print("**** Reading signal TEST-2-5 for time range [0,0.1]")
    d=flap.get_data('TESTDATA',name='TEST-2-5',
                    options={'Scaling':'Volt'},
                    object_name='TEST-1-1',
                    coordinates={'Time':[0,0.1]})
    print("**** Storage contents")
    flap.list_data_objects()
    print()
    print("**** Adding Device x coordinate")
    flap.add_coordinate('TEST-1-1',exp_id='*',coordinates=['Device x','Device z', 'Device y'])
    print("**** Storage contents")
    flap.list_data_objects()

    print()
    print("**** Reading all test signals for time range [0,0.001]")
    d=flap.get_data('TESTDATA',name='TEST-*',
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':[0,0.001]})
    print("**** Storage contents")
    flap.list_data_objects()
    print()
    print("**** Adding Device x coordinate")
    flap.add_coordinate('TESTDATA',exp_id='*',coordinates=['Device x','Device z', 'Device y'])
    print("**** Storage contents")
    flap.list_data_objects()  
    print("**** Getting the time coordinate. The result will be a 3D object with 1 element in all dimensions except along the time.")
    t = flap.get_data_object_ref('TESTDATA').coordinate('Time',options={'Chang':True})[0].flatten()
    plt.figure()
    plt.plot(t)
    plt.xlabel('Index')
    plt.ylabel('Time')

    
def test_arithmetic():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Testing DataObject arithmetic <<<<<<<<<<<<<<<<<<<<<<<<")
    d = flap.get_data('TESTDATA',name='TEST-*',
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':[0,0.001]})
    
    print("\n***** Adding two DataObjects. One coordinate name is different.")
    d1 = copy.deepcopy(d)
    d1.coordinates[1].unit.name='ssd'
    d2 = d + d1
    flap.list_data_objects([d,d1,d2])
    print(d.data[0,0,0],d1.data[0,0,0],d2.data[0,0,0])
    print(d.data[1,1,1],d1.data[1,1,1],d2.data[1,1,1])   

    print("\n***** Adding scalar to a DataObject.")
    d2 = d + 3
    flap.list_data_objects([d,d2])
    print(d.data[0,0,0],d2.data[0,0,0])
    print(d.data[1,1,1],d2.data[1,1,1])   
    d2 = 3 + d
    print(d.data[0,0,0],d2.data[0,0,0])
    print(d.data[1,1,1],d2.data[1,1,1])   

    print("\n***** Subtracting two DataObjects.")
    d1 = d+3
    d2 = d - d1
    flap.list_data_objects([d,d1,d2])
    print(d.data[0,0,0],d1.data[0,0,0],d2.data[0,0,0])
    print(d.data[1,1,1],d1.data[1,1,1],d2.data[1,1,1])   
    d2 = d1 - d
    flap.list_data_objects([d,d1,d2])
    print(d.data[0,0,0],d1.data[0,0,0],d2.data[0,0,0])
    print(d.data[1,1,1],d1.data[1,1,1],d2.data[1,1,1])   

    print("\n***** Subtracting scalar from a DataObject.")
    d2 = d - 3
    flap.list_data_objects([d,d2])
    print(d.data[0,0,0],d2.data[0,0,0])
    print(d.data[1,1,1],d2.data[1,1,1])   
    d2 = 3 - d
    print(d.data[0,0,0],d2.data[0,0,0])
    print(d.data[1,1,1],d2.data[1,1,1])   

    
    print("\n***** Multiplying two DataObjects.")
    d1 = copy.deepcopy(d)
    d1 = d1 + 3
    d2 = d * d1
    flap.list_data_objects([d,d1,d2])
    print(d.data[0,0,0],d1.data[0,0,0],d2.data[0,0,0])
    print(d.data[1,1,1],d1.data[1,1,1],d2.data[1,1,1])   
    
    print("\n***** Multiplying DataObject with constant.")
    d1 = d * 3
    flap.list_data_objects([d,d1,d2])
    print(d.data[0,0,0],d1.data[0,0,0])
    print(d.data[1,1,1],d1.data[1,1,1])   
    

def test_plot():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Testing various plot modes <<<<<<<<<<<<<<<<<<<<<<<<")
    flap.delete_data_object('*')
    plt.close('all')

    print("**** Generating some test data.")
    length = 0.01
    flap.get_data('TESTDATA', name='TEST-1-1', options={'Signal':'Sin','Length':length}, object_name='TEST-1-1')
    flap.get_data('TESTDATA', name='TEST-1-2', options={'Signal':'Const.','Length':length}, object_name='TEST-1-2')
    flap.get_data('TESTDATA', name='TEST-1-3', options={'Signal':'Const.','Length':length}, object_name='TEST-1-3')
    flap.get_data('TESTDATA', name='TEST-1-[1-5]', options={'Signal':'Sin','Length':length}, object_name='TESTDATA')
    flap.get_data('TESTDATA', name='TEST-1-1', options={'Signal':'Complex-Sin','Length':length}, object_name='TEST-1-1_comp')
    flap.get_data('TESTDATA', name='TEST-1-2', options={'Signal':'Complex-Sin','Length':length}, object_name='TEST-1-2_comp')
    flap.get_data('TESTDATA', name='TEST-1-[1-5]', options={'Signal':'Complex-Sin','Length':length}, object_name='TESTDATA_comp')
    
    print("**** Creating a single plot in the upper left corner.")
    plt.figure()
    gs = GridSpec(2, 2)
    plt.subplot(gs[0,0])
    plot_1 = flap.plot('TEST-1-1')
    print("**** Creating a multi xy on the right side.")
    plt.subplot(gs[:,1])
    plot_2 = flap.plot('TESTDATA')
    plot_2 = flap.plot('TESTDATA',slicing={'Signal name':['TEST-1-2','TEST-1-3']})
    print("**** Overplotting into the first plot.")
    flap.plot('TEST-1-3',plot_id=plot_1)
    
    
    print("**** Plotting two complex signals into the same plot.")
    plt.figure()
    plot_3 = flap.plot('TEST-1-1_comp')
    plot_3 = flap.plot('TEST-1-2_comp')
    
    print("**** Plotting absolute value and phase of multiple complex signals.")
    plt.figure()
    gs = GridSpec(1, 2)
    plt.subplot(gs[0,0])
    plot_4 = flap.abs_value('TESTDATA_comp').plot()
    plt.subplot(gs[0,1])
    plot_5 = flap.phase('TESTDATA_comp').plot(options={'Y sep':10})
#    plot_4 = flap.plot('TESTDATA_comp')
    
    print("**** Image plot of testdata and some single plots.")
    plt.figure()
    gs = GridSpec(2,2)
    plt.subplot(gs[0,0])
    plot_5 = flap.plot('TESTDATA',
                       axes=['Time','Row'],
                       plot_type='image',
                       options={'Colormap':'bwr','Z range':[-2,2]})
    plt.subplot(gs[1,0])
    plot_6 = flap.plot('TESTDATA',
                       slicing={'Signal name':'TEST-1-1'},
                       axes=['Time'],
                       options={'Y range':[-2,2]},
                       plot_options={'linestyle':'-'})
    flap.plot('TESTDATA',
                       slicing={'Signal name':'TEST-1-2'},
                       axes=['Time'],
                       plot_options={'linestyle':'--'})
    legend=['Row 1','Row 2']
    plot_6.plt_axis_list[0].legend(legend)
    plt.subplot(gs[:,1])
    plot_7 = flap.plot('TESTDATA',plot_type='multi xy',axes='Time')    
    
def test_plot_xy():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Test plot x-y <<<<<<<<<<<<<<<<<<<<<<<<")
    plt.close('all')
    flap.delete_data_object('*')
    print("**** Reading signal TEST-2-5 for time range [0,0.1]")
    d=flap.get_data('TESTDATA',name='TEST-2-5',
                    options={'Scaling':'Volt'},
                    object_name='TEST-1-1',
                    coordinates={'Time':[0,0.1]})
    print("**** Default plot")
    plt.figure()
    d.plot()
    print("**** Plotting time vs data")
    plt.figure()
    d.plot(axes=['__Data__','Time'])
    print("**** Reading all test signals for time range [0,0.001]")
    d=flap.get_data('TESTDATA',name='TEST-*',
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':[0,0.001]})
    print("**** Adding Device coordinates")
    flap.add_coordinate('TESTDATA',exp_id='*',coordinates=['Device x','Device z', 'Device y'])
    flap.list_data_objects()
    print("**** Plotting measurement points in device corodinates.")
    plt.figure()
    flap.plot('TESTDATA',axes=['Device x','Device z'],plot_type ='scatter')
    print("**** Plotting Device x as a function of Row.")
    plt.figure()
    flap.plot('TESTDATA',axes=['Row','Device x'],plot_type ='scatter',)

def test_plot_multi_xy():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Test plot multi x-y --------")
    plt.close('all')
    plt.figure()
    flap.delete_data_object('*')
    d=flap.get_data('TESTDATA',name='TEST-1-*',
                    options={'Scaling':'Volt'},
                    object_name='TEST')
    print("**** Storage contents")
    flap.list_data_objects()
    flap.plot('TEST',axes='Time',options={'All points':False,'Y sep':4})

def test_simple_slice():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Test simple slice <<<<<<<<<<<<<<<<<<<<<<<<")
    flap.delete_data_object('*')
    print("**** Reading all test signals for time range [0,0.001]")
    d=flap.get_data('TESTDATA',name='TEST-*',
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':[0,0.001]})
    print("**** Adding Device coordinates")
    flap.add_coordinate('TESTDATA',coordinates=['Device x','Device z', 'Device y'])
    print("**** Storage contents before slice")
    flap.list_data_objects()
    print("**** Slicing with {'Signal name': 'TEST-1-*'}")
    flap.slice_data('TESTDATA',slicing={'Signal name': 'TEST-1-*'},output_name='TESTDATA_slice')
    print("**** Sliced object")
    flap.list_data_objects(name='TESTDATA_slice')
    
def test_resample():
    plt.close('all')
    print()
    print(">>>>>>>>>>>>> Test signal resampling (interpolation) <<<<<<<<<<<")
    flap.delete_data_object('*')
    print("**** Generating two test signals with different sampling frequency.")
    flap.get_data('TESTDATA',
                 name='TEST-1-1',
                 options={'Scaling':'Volt','Frequency':1e3, 'Samplerate':1e6},
                 object_name='TEST-1MHz',
                 coordinates={'Time':[0,0.001]}
                 )
    flap.get_data('TESTDATA',
                 name='TEST-1-1',
                 options={'Scaling':'Volt','Frequency':1.5e3, 'Samplerate':3e6},
                 object_name='TEST-3MHz',
                 coordinates={'Time':[0,0.001]}
                 )
    print("\n***** Resampling from lower to higher frequency.")
    plt.figure()
    flap.plot('TEST-1MHz',axes='Time',plot_options={'marker':'o'})
    flap.plot('TEST-3MHz',plot_options={'marker':'o'})
    flap.slice_data('TEST-1MHz',
                    slicing={'Time':flap.get_data_object('TEST-3MHz')},
                    options={'Interpol':'Linear'},
                    output_name='TEST-1MHz_resample')
    flap.plot('TEST-1MHz_resample',plot_options={'marker':'x'})
    
    print("\n***** Resampling from higher to lower frequency.")
    plt.figure()
    flap.plot('TEST-1MHz',axes='Time',plot_options={'marker':'o'})
    flap.plot('TEST-3MHz',plot_options={'marker':'o'})
    flap.slice_data('TEST-3MHz',
                    slicing={'Time':flap.get_data_object('TEST-1MHz')},
                    options={'Interpol':'Linear'},
                    output_name='TEST-3MHz_resample')
    flap.plot('TEST-3MHz_resample',plot_options={'marker':'x'})
    
    print("\n***** Cutting parts.")
    plt.figure()
    flap.slice_data('TEST-1MHz',
                    slicing={'Time':flap.Intervals([1e-4,5e-4],[2e-4,7e-4])},
                    options={'Slice':'Simple'},                     
                    output_name='TEST-1MHz_parts')
    flap.plot('TEST-1MHz_parts',axes='Time',plot_options={'marker':'o'})
    flap.list_data_objects()
    
def test_select_multislice():
    plt.close('all')
    print()
    print('>>>>>>>>>>>>>>>>>>> Test select on maxima and multi slice <<<<<<<<<<<<<<<<<<<<<<<<')
    flap.delete_data_object('*')
    d = flap.get_data('TESTDATA',name='TEST-1-1',object_name='TEST-1-1',options={'Length':0.050})
    print("**** Selecting 100 microsec long intervals around the maxima of the signal.")
    d_int = flap.select_intervals('TEST-1-1',
                                  coordinate='Time',
                                  options={'Select':None,
                                           'Length':0.0001,
                                           'Event':{'Type':'Max-weight',
                                                    'Threshold':1,
                                                    'Thr-type':'Sigma'}},
                                  plot_options={'All points':True},
                                  output_name='SELECT')
    flap.list_data_objects()
    d_int.plot(axes=['__Data__',0],plot_type='scatter',options={'Force':True})
    if (d_int is not None):
        print("**** Overplotting the signals in the selected intervals.")
        flap.slice_data('TEST-1-1',slicing={'Time':d_int},output_name='TEST-1-1_sliced')
        flap.list_data_objects()
        plt.figure()
        n_int = d_int.shape[0]
        for i in range(n_int):
            flap.plot('TEST-1-1_sliced',
                      slicing={'Interval(Time)':i},
                      axes='Rel. Time in int(Time)')
            
def test_select_corr():
    plt.close('all')
    print('>>>>>>>> Test select with correlation <<<<<<<<<')
    flap.delete_data_object('*')
    
    #getting a data object with a sin signal
    d = flap.get_data('TESTDATA',name='TEST-1-1',object_name='TEST-1-1',options={'Length':0.150})
    d.data = d.data/2 #its intensity was too high, this way the test is more realistic
    t = d.coordinate("Time")[0][:]
    
    print("**** Generating random perturbations.")
    #parameters of the perturbations:
    number = 100
    ts1 = 30
    ts2 = 60
    Imax = 0.35
    Imin = 0.20
    
    #adding events
    timeres = t[1]-t[0]
    ti = np.zeros((number))
    I = np.zeros((number))
    t_indexes = range(1, t.shape[0], 1)
    ts = np.zeros((number))
    for i in range(number):
        ti[i] = ran.choice(t_indexes)
        I[i] = ran.random()*(Imax-Imin) + Imin
        ts[i] = ran.random()*(ts2-ts1) + ts1
    add_light = np.zeros((d.data.shape))
    for k in range(number):
        add_light = add_light + I[k]*np.e**(-(1/2)*((t-t[int(ti[k])])/(ts[k]*timeres))**2)
        
    #adding events and a mean signal
    mean_signal = 1.4997363040171798 #W7-X 20181018.026 ABES-11
    d.data = d.data + add_light + mean_signal
    
    #plotting test signal without noise
    plt.figure()
    plt.plot(t,d.data)
    plt.xlabel("Time [s]")
    plt.ylabel("Test signal without noise [V]")
    plt.title("Gaussian events")
    
    #adding noise
    a = 0.04378451
    b = 0.00524973
    mean_error = a * np.sqrt(mean_signal) + b #valid error for W7-X 20181018.026 ABES-11
    d.data += np.random.normal(0,mean_error,d.data.shape[0])
    
    #plotting test signal with noise
    plt.figure()
    plt.plot(t,d.data)
    plt.xlabel("Time [s]")
    plt.ylabel("Test signal [V]")
    plt.title("Gaussian events")
    
    #filtering the signal to, then showing its APSD before and after
    intfilter_tau = 16e-6
    difffilter_tau = 50e-6
    d_f = d.filter_data(options={'Type':'Int','Tau':intfilter_tau})
    d_f = d_f.filter_data(options={'Type':'Diff','Tau':difffilter_tau})
    
    plt.figure()
    p1 = d.apsd(options={'Interval':1, 'Range':[1,1E6],'Res':100})
    p2 = d_f.apsd(options={'Interval':1, 'Range':[1,1E6],'Res':100})
    p1.plot(axes="Frequency",options={'Log x': True,
            'Log y':True, 'Error':False, 'X range':[100,2e5]})
    p2.plot(axes="Frequency",options={'Log x': True,
            'Log y':True, 'Error':False, 'X range':[100,2e5]})
    plt.ylabel("APSD",fontsize=15)
    plt.xlabel("Frequency [Hz]",fontsize = 15)
    legend=["before filter","after filter"]
    plt.legend(legend,fontsize = 12)
    plt.title("Gaussian events")
    
    print("**** Selecting intervals.")
    #parameters of the selection:
    Lp = ts1*20*timeres
    trh = 3
    N = 5
    sigmastep = (ts1 + ts2)/N
    minsigma = ts1
    sigm = np.arange(minsigma,N*sigmastep+minsigma,sigmastep) * timeres
    
    #interval selection, and average event:
    d_int = flap.select_intervals(d,coordinate='Time',
                                options={'Select':None,
                                        'Length':Lp,
                                        'Event':{'Type':'Correlation',
                                                  'Threshold':trh,
                                                  'Thr-type':'sigma',
                                                  'Gaussian sigma':sigm,
                                                  'Int tau':intfilter_tau,
                                                  'Diff tau':difffilter_tau}})
    mean_event = d.slice_data(slicing={'Time':d_int},
                              summing ={'Start Time in int(Time)':"Mean"},
                              options={"Regenerate coordinates":False})
    
    #showing the time coordinates of the events:
    inde = np.zeros((t.shape))
    for i in range(d_int.data.shape[0]):
        inde += 1e-10 > abs(d_int.data[i] + Lp/2 - t) 
    ind=np.nonzero(inde)[0]
    plt.figure()
    plt.plot(t,d.data,"+")
    plt.plot(d_int.data + Lp/2,d.data[ind],"ro")
    plt.xlabel("Time [s]",fontsize = 15)
    plt.ylabel("Test signal [V]")
    legend = ["signal","selected events"]
    plt.legend(legend,loc = "upper right")
    plt.title("Gaussian events")
    
    #showing the mean event:
    plt.figure()
    mean_event.plot(axes = "Rel. Time in int(Time)")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean event [V]")
    plt.title("Gaussian events")
    
    
    #creating a new, noizy signal
    d2 = flap.get_data('TESTDATA',name='TEST-1-1',object_name='TEST-1-1',options={'Length':0.150})
    d2.data = d2.data/2 #its intensity was too high, this way the test is more realistic
    t = d2.coordinate("Time")[0][:]
    a = 0.04378451
    b = 0.00524973
    mean_error = a * np.sqrt(mean_signal) + b #valid error for W7-X 20181018.026 ABES-11
    d2.data = d2.data + np.random.normal(mean_signal,mean_error,d2.data.shape[0])
    
    tau=0.0001
    E = 0.00005
    y= (2*E/(3*np.sqrt(np.pi)))*(1+(tau/(t))**2)*(tau/(t)**2)*np.exp(-(tau/(t))**2)
    plt.figure()
    plt.plot(t*1e3,y)
    plt.xlim(-0.0001*1e3,0.002*1e3)
    plt.xlabel("Time [ms]")
    plt.ylabel("Signal [V]")
    plt.title("Theoretical function for type I ELMs in the Free Streaming Model")
    
    print("**** Generating random ELMs.")
    #parameters of the perturbations:
    taumin = tau/2
    taumax = tau*2
    Emax = E*2
    Emin = E/2
    number = 50
    
    #adding perturbations:
    ti = np.zeros((number))
    Es = np.zeros((number))
    t_indexes = range(1, t.shape[0], 1)
    taus = np.zeros((number))
    for i in range(number):
        ti[i] = ran.choice(t_indexes)
        Es[i] = ran.random()*(Emax-Emin) + Emin
        taus[i] = ran.random()*(taumax-taumin) + taumin
    add_light = np.zeros((d2.data.shape))
    for k in range(number):
        y = (2*Es[k]/(3*np.sqrt(np.pi)))*(1+(taus[k]/(t-t[int(ti[k])]))**2)*(taus[k]/(t-t[int(ti[k])])**2)*np.exp(-(taus[k]/(t-t[int(ti[k])]))**2)
        for i in range(y.shape[0]):
            if(y.shape[0] > i + int(ti[k])):
                if(i > ti[k]):
                    y[i] = y[i]
                else:
                    y[i] = 0
            elif(i + int(ti[k]) >= y.shape[0]):
                if(i > ti[k]):
                    y[i] = y[i]
                else:
                    y[i] = 0
        add_light = add_light + y
    d2.data = d2.data + add_light
    
    #howing the raw test signal
    plt.figure()
    plt.plot(t,d2.data)
    plt.xlabel("Time [ms]")
    plt.ylabel("Signal [V]")
    plt.title("ELM-like events")
    
    #filtering the signal to, then showing its APSD before and after
    intfilter_tau = 16e-6
    difffilter_tau = 50e-6
    d_f = d2.filter_data(options={'Type':'Int','Tau':intfilter_tau})
    d_f = d_f.filter_data(options={'Type':'Diff','Tau':difffilter_tau})
    plt.figure()
    p1 = d2.apsd(options={'Interval':1, 'Range':[1,1E6],'Res':100})
    p2 = d_f.apsd(options={'Interval':1, 'Range':[1,1E6],'Res':100})
    p1.plot(axes="Frequency",options={'Log x': True,
            'Log y':True, 'Error':False, 'X range':[100,2e5]})
    p2.plot(axes="Frequency",options={'Log x': True,
            'Log y':True, 'Error':False, 'X range':[100,2e5]})
    plt.ylabel("APSD",fontsize=15)
    plt.xlabel("Frequency [Hz]",fontsize = 15)
    legend=["before filter","after filter"]
    plt.legend(legend,fontsize = 12)
    plt.title("ELM-like events")
    
    #showing the effect of filter on the events:
    plt.figure()
    plt.plot(t,d_f.data)
    plt.xlabel("Time [ms]")
    plt.ylabel("Signal after filter [V]")
    plt.title("ELM-like events")
    
    print("**** Selecting intervals.")
    #parameters of the selection:
    Lp = ts1*20*timeres
    trh = 3
    N = 5
    sigmastep = (ts1 + ts2)/N
    minsigma = ts1
    sigm = np.arange(minsigma,N*sigmastep+minsigma,sigmastep) * timeres
    
    #interval selection, and average event:
    d2_int = flap.select_intervals(d2,coordinate='Time',
                                options={'Select':None,
                                        'Length':Lp,
                                        'Event':{'Type':'Correlation',
                                                  'Threshold':trh,
                                                  'Thr-type':'sigma',
                                                  'Gaussian sigma':sigm,
                                                  'Int tau':intfilter_tau,
                                                  'Diff tau':difffilter_tau}})
    mean_event = d2.slice_data(slicing={'Time':d2_int},
                              summing ={'Start Time in int(Time)':"Mean"},
                              options={"Regenerate coordinates":False})
    
    #showing the mean event:
    plt.figure()
    mean_event.plot(axes = "Rel. Time in int(Time)")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean event [V]")
    plt.title("ELM-like events")

def test_binning():
    print()
    print()
    print('>>>>>>> Test image binning through multi-slice<<<<<<<<<<<')
    print("**** Generating a sequence of test images")
    flap.get_data('TESTDATA',
                  name='VIDEO',
                  object_name='TEST_VIDEO',
                  options={'Length':0.05,'Samplerate':1e3,'Width':500,'Height':800,'Image':'Gauss','Spotsize':10})
    print("***** Showing one image")
    plt.figure()
    flap.plot('TEST_VIDEO',
              slicing={'Time':30e-3/3},
              plot_type='image',
              axes=['Image x','Image y'],
              options={'Clear':True,'Interpolation':None,'Aspect':'equal'})
    flap.list_data_objects('TEST_VIDEO')
    flap.slice_data('TEST_VIDEO',
                    slicing={'Image x':flap.Intervals(0,4,step=5), 'Image y':flap.Intervals(0,9,step=10)},
                    summing={'Interval(Image x) sample index':'Mean','Interval(Image y) sample index':'Mean'},
                    output_name='TEST_VIDEO_binned'
                    )
    flap.list_data_objects('TEST_VIDEO_binned')
    print("***** Showing one image of the (5,10) binned video ")
    plt.figure()
    flap.plot('TEST_VIDEO_binned',
              slicing={'Time':30e-3/3},
              plot_type='image',
              axes=['Image x','Image y'],
              options={'Clear':True,'Interpolation':None,'Aspect':'equal'})
    flap.list_data_objects()

def test_detrend():
    plt.close('all')            
    print()    
    print('>>>>>>>>>>>>>>>>>>> Test detrend <<<<<<<<<<<<<<<<<<<<<<<<')
    flap.delete_data_object('*')
    print("**** Generating 8 sine signals with variable frequency.")
    d = flap.get_data('TESTDATA',name='TEST-1-*',object_name='TEST-1',options={'Signal':'Sin', 'Freq':[1e3,5E3], 'Length':0.005})
    print("**** Detrending in 2 intervals with second order poly fit.")
    plt.figure()
    flap.plot('TEST-1',axes='Time')
    flap.detrend('TEST-1',intervals={'Time':flap.Intervals(0.001,0.0015,step=0.003,number=2)},
                                     options={'Trend':['Poly',2]},output_name='TEST-1_detrend')
    flap.plot('TEST-1_detrend',axes='Time')

def test_apsd():
    plt.close('all')
    print()
    print('>>>>>>>>>>>>>>>>>>> Test apsd (Auto Power Spectral Density) <<<<<<<<<<<<<<<<<<<<<<<<')
    flap.delete_data_object('*')
    #plt.close('all')
    print('**** Generating test signals with frequency changing from channel to channel.')
    d = flap.get_data('TESTDATA',name='TEST*',object_name='TEST-1_S',
                      options={'Signal':'Sin','F':[1e3,1e4],'Length':1.})
    print('**** Calculating 150 APSDs, each 1 million sample.')
    print('**** APSD START')
    start = time.time()
    flap.apsd('TEST-1_S',output_name='TEST-1_APSD_Sin1',options={'Res':12, 'Int':10})
    stop = time.time()
    print('**** APSD STOP')
    print("**** Calculation time: {:5.2f} second/signal".format((stop-start)/150.))
    plt.figure()
    flap.plot('TEST-1_APSD_Sin1',slicing={'Row':1}, axes='Frequency',options={'All':True,'X range':[0,5e3]})
    plt.title('TEST-1-1_APSD_Sin1')
    
    print("**** Testing with a complex signal.")
    flap.delete_data_object('*')
    d = flap.get_data('TESTDATA',name='TEST-1-1',object_name='TEST-1-1_CS',options={'Signal':'Complex-Sin'})
    flap.apsd('TEST-1-1_CS',coordinate='Time', output_name='TEST-1-1_APSD_Complex-Sin',
              options={'Res':10,'Range':[-1e5,1e5]})
    flap.slice_data('TEST-1-1_APSD_Complex-Sin',slicing={'Frequency':flap.Intervals(-5e3,5e3)},
                                                         output_name='TEST-1-1_APSD_Complex-Sin_sliced')
    flap.list_data_objects()
    plt.figure()
    flap.plot('TEST-1-1_APSD_Complex-Sin_sliced',axes='Frequency',options={'All':True})
    plt.title('TEST-1-1_APSD_Complex-Sin_sliced')
    
    print("**** Testing interval selection in apsd. APSD from 8 intervals, each 80 ms long.")
    d = flap.get_data('TESTDATA',name='TEST-1-1',object_name='TEST-1-1',options={'Signal':'Sin','Length':1})
    intervals = flap.Intervals(0,0.08,step=0.1,number=8)
    flap.apsd('TEST-1-1',output_name='TEST-1-1_APSD',intervals=intervals,options={'Res':12, 'Int':10})
    plt.figure()
    flap.plot('TEST-1-1_APSD',options={'X range':[0,5e3]})

def test_filter():
    plt.close('all')
    print()
    print('>>>>>>>>>>>>>>>>>>> Test filter <<<<<<<<<<<<<<<<<<<<<<<<')
    flap.delete_data_object('*')

    print("**** Generating 10 square wave signals and filtering with integrating filter, 10 microsec")
    t = np.arange(1000)*1e-6
    d = np.ndarray((len(t),10),dtype=float)
    for i in range(10):
        d[:,i] = np.sign(np.sin(math.pi*2*(1e4+i*1e3)*t)) + 1
    c = flap.Coordinate(name='Time',
                        unit='Second',
                        mode=flap.CoordinateMode(equidistant=True),
                        start=0.0,
                        step=1e-6,
                        dimension_list=[0]
                        )
    d = flap.DataObject(data_array=d,coordinates=[c])
    flap.add_data_object(d,"Signal")

    
    plt.figure()
    d.plot(options={'Y sep':3})
    di = d.filter_data(coordinate='Time',
                       intervals=flap.Intervals(np.array([1e-4,6e-4]),np.array([2e-4,8e-4])),
                       options={'Type':'Int','Tau':10e-6}).plot(options={'Y sep':3})

    print("**** Filtering with differential filter, 10 microsec")
    plt.figure()
    d.plot(options={'Y sep':3})
    flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                       intervals=flap.Intervals(np.array([1e-4,6e-4]),np.array([2e-4,8e-4])),
                       options={'Type':'Diff','Tau':10e-6})
    flap.plot('Signal_filt',options={'Y sep':3})

    print("**** Generating random data, 1 million points and overplotting spectra with various filters.")
    d = flap.get_data('TESTDATA',
                      name='TEST-1-1',
                      options={'Signal':'Random','Scaling':'Digit','Length':1},
                      object_name='Signal')    

    plt.figure()
    flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                     options={'Type':'Int','Tau':16e-6})
    flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid = flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid.plt_axis_list[-1].set_title("{'Type':'Int','Tau':16e-6}")   
  
    plt.figure()
    flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                     options={'Type':'Diff','Tau':16e-6})
    flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid = flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid.plt_axis_list[-1].set_title("{'Type':'Diff','Tau':16e-6}")   

    plt.figure()
    flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                     options={'Type':'Lowpass','f_high':5e4})
    flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid = flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid.plt_axis_list[-1].set_title("{'Type':'Lowpass','f_high':5e4}") 

    plt.figure()
    flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                     options={'Type':'Highpass','f_low':1e4,'f_high':5e4})
    flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid = flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid.plt_axis_list[-1].set_title("{'Type':'Highpass','f_low':1e4,'f_high':5e4}") 

    plt.figure()
    flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                     options={'Type':'Bandpass','f_low':5e3,'f_high':5e4})
    flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid = flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
       .plot(options={'Log x':True, 'Log y': True})
    plotid.plt_axis_list[-1].set_title("{'Type':'Bandpass','f_low':5e3,'f_high':5e4}") 
    
    plt.figure()
    print("**** Bandpower signal [5e4-2e5] Hz, inttime 20 microsec")
    flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                     options={'Type':'Bandpass','f_low':5e4,'f_high':2e5, 'Power':True, 'Inttime':20e-6})
    plotid = flap.plot('Signal_filt')
    plotid.plt_axis_list[-1].set_title("'Type':'Bandpass','f_low':5e4,'f_high':2e5, 'Power':True, 'Inttime':20e-6}") 
    
def test_cpsd():
    plt.close('all')
    print()
    print('>>>>>>>>>>>>>>>>>>> Test cpsd (Cross Spectral Power Density) <<<<<<<<<<<<<<<<<<<<<<<<')
    flap.delete_data_object('*')

    print("**** Generating 8 random data, 1 million points each.")

    d =  flap.get_data('TESTDATA', name='TEST-1-[1-8]', options={'Signal':'Random','Length':1}, object_name='TESTDATA')
    print("**** Calculating all cpsd")
    flap.cpsd('TESTDATA',
              options={'Norm':True,'Interval':50, 'Log':True,'Res':10,'Range':[100,1e5]},
              output_name='TESTDATA_cpsd')
    flap.abs_value('TESTDATA_cpsd',output_name='TESTDATA_cpsd_abs')

    print("**** Plotting coherency between channels 1-2 and its significance level.")    
    plt.figure()
    flap.plot('TESTDATA_cpsd_abs',
              axes='Frequency',
              slicing={'Row (Ref)':1,'Row':2},
              options={'Log y':True,'Log x':True, 'Error':False})
    flap.error_value('TESTDATA_cpsd_abs').plot(slicing={'Row (Ref)':1,'Row':2})
    
    plt.figure()
    print("**** Plotting mean coherence in 1e4-1e5 frequency range as a function of row index.")
    flap.slice_data('TESTDATA_cpsd_abs',
                    slicing={'Frequency':flap.Intervals(1e4,1e5)},
                    summing={'Frequency':'Mean'}).plot(axes='Row (Ref)',options={'Y sep': 1.5})

def test_ccf():
    plt.close('all')
    print()
    print('>>>>>>>>>>>>>>>>>>> Test ccf (Cross Correlation Function) <<<<<<<<<<<<<<<<<<<<<<<<')
    flap.delete_data_object('*')
    print("**** Generating 10x15 random test signals, 5000 points each, 1 MHz sampling.")
    flap.get_data('TESTDATA',
                  name='TEST-*-*',
                  options={'Length':0.005, 'Signal':'Random'},
                  object_name='TESTDATA')
    print("**** Filtering with 10 microsec integrating filter.")
    flap.filter_data('TESTDATA',coordinate='Time',options={'Type':'Int','Tau':1e-5},output_name='TESTDATA_filt')
    flap.list_data_objects()
    plt.figure()
    print("**** Plotting an original and a filtered signal.")
    flap.plot('TESTDATA',slicing={'Row':1,'Column':1},axes='Time')
    flap.plot('TESTDATA_filt',slicing={'Row':1,'Column':1})
    print('**** Calculating the 10x15x10x15 CCFs, each 5000 samples.')
    print('**** CCF START')
    start = time.time()    
    flap.ccf('TESTDATA_filt',coordinate='Time',
             options={'Trend':'Mean','Range':[-1e-4,1e-4],'Res':1e-5,'Norm':True},output_name='CCF')
    stop = time.time()
    print('**** CCF STOP')
    print("**** Calculation time: {:6.3f} ms/signal".format(1000*(stop-start)/(10*15*10*15)))
    flap.list_data_objects()
    print("**** Plotting spatiotemporal correlation function at ref row, column 3,3, column 3")
    plt.figure()
    flap.plot('CCF',slicing={'Row (Ref)':3,'Column (Ref)':3,'Column':3},axes=['Time lag'],plot_type='multi xy')

    print("**** Slicing TESTDATA_filt for row: 1-3, column:1-4")
    flap.slice_data('TESTDATA_filt',slicing={'Row':[1,2,3],'Column':[1,2,3,4]},output_name='TESTDATA_filt_3x4')
    print('**** Calculating CCFs, between original and sliced TESTDATAfilt')
    print('**** CCF START')
    flap.ccf('TESTDATA_filt',ref='TESTDATA_filt_3x4',coordinate='Time',
             options={'Trend':'Mean','Range':[-1e-4,1e-4],'Res':1e-5,'Norm':True},output_name='CCF_ref')
    print('**** CCF STOP')
    flap.list_data_objects()
    print("**** Plotting spatiotemporal correlation function at ref row, column 3,3, column 3")
    plt.figure()
    flap.plot('CCF_ref',slicing={'Row (Ref)':3,'Column (Ref)':3,'Column':3},axes=['Time lag'],plot_type='multi xy')

def test_image():
    plt.close('all')
    print()
    print('>>>>>>>>>>>>>>>>>>> Test image <<<<<<<<<<<<<<<<<<<<<<<<')
    flap.delete_data_object('*')
    print("**** Generating a sequence of test images")
    flap.get_data('TESTDATA',name='VIDEO',object_name='TEST_VIDEO',options={'Length':0.1,'Samplerate':1e3, 'Frequency':10,'Spotsize':100})
    flap.list_data_objects()
    print("***** Showing one image")
    plt.figure()
    flap.plot('TEST_VIDEO',slicing={'Time':30e-3/4},plot_type='image',axes=['Image x','Image y'],options={'Clear':True})
    plt.figure()
    print("**** Showing a sequence of images and saving to test_video.avi")
    flap.plot('TEST_VIDEO',plot_type='anim-image',axes=['Image x','Image y','Time'],
              options={'Z range':[0,4095],'Wait':0.01,'Clear':True,'Video file':'test_video.avi','Colorbar':True,'Aspect ratio':'equal'})
    plt.figure()
    print("*** Showing the same images as contour plots and saving to test_video_contour.avi")
    flap.plot('TEST_VIDEO',plot_type='anim-contour',axes=['Image x','Image y','Time'],
              options={'Z range':[0,4095],'Wait':0.01,'Clear':True,'Video file':'test_video_contour.avi','Colorbar':False})
    print("*** Converting data object x, y coordinates to non-equidistant.")
    d=flap.get_data_object('TEST_VIDEO')
    coord_x = d.get_coordinate_object('Image x')
    index = [0]*3
    index[coord_x.dimension_list[0]] = ...
    x = np.squeeze(d.coordinate('Image x',index=index)[0])
    coord_x.mode.equidistant = False
    coord_x.values = x
    coord_x.shape = x.shape
    coord_y = d.get_coordinate_object('Image y')
    index = [0]*3
    index[coord_y.dimension_list[0]] = ...
    y = np.squeeze(d.coordinate('Image y',index=index)[0])
    coord_y.mode.equidistant = False
    coord_y.values = y
    coord_y.shape = y.shape
    flap.add_data_object(d,"TEST_VIDEO_noneq")
    flap.list_data_objects()
    plt.figure()
    print("**** Showing this video and saving to  test_video_noneq.avi")
    flap.plot('TEST_VIDEO_noneq',plot_type='anim-image',axes=['Image x','Image y','Time'],
              options={'Z range':[0,4095],'Wait':0.01,'Clear':True,'Video file':'test_video_noneq.avi','Colorbar':True,'Aspect ratio':'equal'})
def test_pdf():
    print('>>>>>>>>>>>>>>>>>>> Test Probability Distribution Function (PDF) <<<<<<<<<<<<<<<<<<<<<<<<')
    print("**** Generating 10x15 random test signals, 5000 points each, 1 MHz sampling.")
    flap.get_data('TESTDATA',
                  name='TEST-*-*',
                  options={'Length':0.005, 'Signal':'Sin'},
                  object_name='TESTDATA')
    flap.pdf('TESTDATA',coordinate='Time',options={'Number':30},output_name='PDF')
    flap.list_data_objects()
    plt.figure()
    flap.plot('PDF',slicing={'Column':3},axes=['Signal'])
    plt.title('PDF of sine waves')
    
def test_stft():
    print('>>>>>>>>>>>>>>>>>>> Test Short Time Fourier Transform (STFT) <<<<<<<<<<<<<<<<<<<<<<<<')
    print("**** Generating 0.1 s long test data signal with linearly changing frequency: 10-100kHz")   
    f = np.linspace(1e4,1e5,num=11)
    coord = flap.Coordinate(name='Time',
                            start=0.0,
                            step=0.01,
                            mode=flap.CoordinateMode(equidistant=True),
                            dimension_list=[0]
                            )
    f_obj = flap.DataObject(data_array=f,
                            coordinates=[coord],
                            data_unit=flap.Unit(name='Frequency',unit='Hz')
                            )
    d=flap.get_data('TESTDATA',
                    name='TEST-1-1',
                    options={'Scaling':'Volt',
                             'Frequency':f_obj,
                             'Length':0.1,
                             'Row number':1,
                             'Column number':1
                             },
                    object_name='TESTDATA'
                    )
    flap.stft('TESTDATA',output_name='TEST_STFT')
    flap.abs_value('TEST_STFT',output_name='TEST_STFT')
    flap.list_data_objects()
    plt.figure()
    flap.plot('TEST_STFT',axes=['Time','Frequency'],plot_type='image')
    
 
def show_plot():
    plt.pause(0.05)
    plt.show(block=False)
    plt.pause(0.05)
  
def keypress_event(event):
    global keypressed
    keypressed = event.key
    
def keypress_start():
    global keypressed
    keypressed = None
    
def key_pressed():
    global keypressed
    if (keypressed is not None):
        return True
    
def wait_press(flag=False):
    if (flag):
        input("Press Enter to continue...")

def wait_for_key(flag=False):
    if (flag):
        print("Press any key ON A PLOT to continue...")
        fig = plt.gcf()
        kbd_press = fig.canvas.mpl_connect('key_press_event', keypress_event)
        keypress_start()
        while not key_pressed():
            time.sleep(0.01)
            plt.pause(0.01)
        fig.canvas.mpl_disconnect(kbd_press)
    
    
# Reading configuration file in the test directory
thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_tests.cfg")
flap.config.read(file_name=fn)

test_all = False

# Running tests
plt.close('all')

root = Tk()
dpi = plt.rcParams["figure.dpi"]
plt.rcParams["figure.figsize"] = [root.winfo_screenwidth()*0.3/dpi, root.winfo_screenheight()*0.3/dpi]

if (False or test_all):
    test_storage()
    wait_press()
if (False or test_all):
    test_testdata()
    wait_press()
if (False or test_all):
    test_saveload()
    wait_press()
if (False or test_all):
    test_coordinates()
    show_plot()
    wait_for_key()
if (False or test_all):    
    test_arithmetic()
    wait_press()
if (False or test_all):
    test_plot()
    wait_for_key()
if (False or test_all):
    test_plot_xy()
    show_plot()
    wait_for_key()
if (False or test_all):
    test_plot_multi_xy()
    show_plot()
    wait_for_key()
if (False or test_all):
    test_simple_slice()
    show_plot()
    wait_for_key()
if (False or test_all):
    test_resample()
    plt.pause(0.05)
    wait_for_key()   
if (False or test_all):
    test_select_multislice()
    show_plot()
    wait_for_key()
if (True or test_all):
    test_select_corr()
    show_plot()
    wait_for_key()    
if (False or test_all):
    test_binning()
    show_plot()
    wait_for_key()    
if (False or test_all): 
    test_detrend()
    plt.pause(0.05)
    wait_for_key()
if (False or test_all):
    test_apsd()
    show_plot()
    wait_for_key()
if (False or test_all):
    test_filter()
    plt.pause(0.05)
    wait_for_key()
if (False or test_all):
    test_cpsd()
    show_plot()
    wait_for_key()
if (False or test_all):
    test_ccf()
    show_plot()
    wait_for_key()
if (False or test_all):
    test_stft()
    show_plot()
    wait_for_key()
if (False or test_all):
    test_image()
    show_plot()
    wait_for_key()
if (False or test_all):
    test_pdf()
    wait_press()

print(">>>>>>>>>>>>>>>> All tests finished <<<<<<<<<<<<<<<<<<<<")



