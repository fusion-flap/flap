# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 22:11:28 2018

@author: Sandor Zoletnik  (zoletnik.sandor@wigner.mta.hu)

Example/test programs for FLAP.

"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
from scipy import signal
import copy
import time
import math

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

def test_saveload():
    print()
    print("\n>>>>>>>>>>>>>>>>>>> Test save/load <<<<<<<<<<<<<<<<<<<<<<<<")
    flap.delete_data_object('*')
    d=flap.get_data('TESTDATA',name='TEST-*',
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':[0,0.001]})
    flap.slice_data('TESTDATA',slicing={'Signal name': 'TEST-1-*'},output_name='TESTDATA_slice')
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
    print("**** Plotting Device x as a funciton of Row.")
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
    print('>>>>>>>>>>>>>>>>>>> Test apds (Auto Power Spectral Density) <<<<<<<<<<<<<<<<<<<<<<<<')
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
    flap.get_data('TESTDATA',name='VIDEO',object_name='TEST_VIDEO',options={'Length':0.1})
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
    print("**** Showing this video and test_video_noneq.avi")
    flap.plot('TEST_VIDEO_noneq',plot_type='anim-image',axes=['Image x','Image y','Time'],
              options={'Z range':[0,4095],'Wait':0.01,'Clear':True,'Video file':'test_video_noneq.avi','Colorbar':True,'Aspect ratio':'equal'})
 
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

def wait_for_key():
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

if (False or test_all):
    test_storage()
    input("Press Enter to continue...")
if (False or test_all):
    test_saveload()
    input("Press Enter to continue...")
if (False or test_all):
    test_coordinates()
    input("Press Enter to continue...")
if (False or test_all):    
    test_arithmetic()
    input("Press Enter to continue...")
if (False or test_all):
    test_plot()
    wait_for_key()
#    input("Press Enter to continue...")
if (False or test_all):
    test_plot_xy()
    show_plot()
    wait_for_key()
#    input("Press Enter to continue...")
if (False or test_all):
    test_plot_multi_xy()
    show_plot()
    wait_for_key()
#    input("Press Enter to continue...")
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
if (True or test_all):
    test_image()
    show_plot()
    wait_for_key()

print(">>>>>>>>>>>>>>>> All tests finished <<<<<<<<<<<<<<<<<<<<")



