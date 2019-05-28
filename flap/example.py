# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 22:11:28 2018

@author: Zoletnik

This is an example program for the flap package
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
from scipy import signal
import copy

import flap
import flap.modules.apdcam
import flap.modules.w7x_abes
import flap.testdata
import flap.config
import flap.modules.w7x_webapi

flap.modules.w7x_abes.register()
flap.testdata.register()
flap.modules.apdcam.register()
flap.modules.w7x_webapi.register()



def test_coordinate_data():
    print("\n------- test coordinate data  --------")
#    dp = 'c:/Data/W7-X_ABES/'
    d=flap.get_data('W7X_ABES',exp_id='20181018.003',name=['ABES-1[5,6]'],\
                    options={'Scaling':'Volt'},object_name='ABES',\
                    coordinates=flap.Coordinate(name='Time',c_range=[1,1.00001]))
    print("**** Storage contents")
    flap.list_data_objects()
    print("Getting Time for row 0")
    c,cl,ch = d.coordinate('Time',[...,0])
    print("  shape:"+str(c.shape)+" range: "+str(np.min(c))+" - "+str(np.max(c)))
    if (max(c.shape) < 30):
        print(" Coordinates: "+str(c))
    if (cl is not None):
        print("  L range shape:"+str(cl.shape)+" range: "+str(np.min(cl))+" - "+str(np.max(cl)))
    if (ch is not None):
        print("  H range shape:"+str(ch.shape)+" range: "+str(np.min(ch))+" - "+str(np.max(ch)))

    print("Getting Signal name for column 0")
    c,cl,ch = d.coordinate('Signal name',[0,...])
    if (max(c.shape) < 30):
        print(" Coordinates: "+str(c))

    print("Getting Channel for all data")
    c,cl,ch = d.coordinate('Channel',...)
    print("  shape:"+str(c.shape)+" range: "+str(np.min(c))+" - "+str(np.max(c)))
    if (max(c.shape) < 30):
        print(" Coordinates: "+str(c))

    print("Getting Time for all data")
    c,cl,ch = d.coordinate('Time',...)
    print("  shape:"+str(c.shape)+" range: "+str(np.min(c))+" - "+str(np.max(c)))
    if (max(c.shape) < 30):
        print(" Coordinates: "+str(c))

    d=flap.get_data('W7X_ABES',exp_id='20181018.003',name=['ABES-12'],\
                    options={'Scaling':'Volt'},object_name='ABES12',\
                    coordinates=flap.Coordinate(name='Time',c_range=[5,6]))
    print("Getting Time for all data of ABES12")
    c,cl,ch = d.coordinate('Time',...)
    print("  shape:"+str(c.shape)+" range: "+str(np.min(c))+" - "+str(np.max(c)))
    if (max(c.shape) < 30):
        print(" Coordinates: "+str(c))

def test_plot_object():
    print("\n------- test plot data object --------")
    plt.close()
#    dp = os.path.join('c:/Data/W7-X_ABES/','20181018.003')
    d=flap.get_data('APDCAM',name='ADC10',coordinates={'Time':[5,6]})
    plt.close()
    d.plot()
    plt.figure()
    d.plot(axes=['Time'])


def test_plot_xy():
    print("\n------- test plot x-y --------")
    plt.close('all')
    print("---- Reading signal TEST-2-5 for time range [0,0.1]")
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
    print("---- Reading all test signals for time range [0,0.001]")
    d=flap.get_data('TESTDATA',name='*',
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':[0,0.001]})
    print("**** Adding Device coordinates")
    flap.add_coordinate('TESTDATA',exp_id='*',coordinates=['Device x','Device z', 'Device y'])
    print("**** Plotting measurement points in device corodinates.")
    plt.figure()
    d.plot(axes=['Device x','Device z'],plot_type ='scatter')
    print("**** Plotting Device x as a funciton of Row.")
    plt.figure()
    d.plot(axes=['Row','Device x'],plot_type ='scatter',)

def test_errorplot():
    x = np.arange(20)
    y = x**2
    err = y*0.2
    c = flap.Coordinate(mode=flap.CoordinateMode(equidistant=True),
                        start=4.,
                        step=0.2,
                        name='X coord',
                        unit='[mm]',
                        dimension_list=[0],
                        value_ranges=0.1)
    d=flap.DataObject(data_array=y,error=err,coordinates=[c])
    plt.close('all')
    d.plot()
    plt.figure()
    d.plot(axes=['__Data__','X coord'])
    plt.figure()
    d.plot(plot_type='scatter')

def test_plot_multi_xy():
    print("\n------- test plot multi x-y --------")
    plt.close()
    d=flap.get_data('W7X_ABES',name='ABES-[8-20]',
                    exp_id='20181018.008',
                    options={'Scaling':'Volt'},
                    object_name='ABES')
    print("**** Storage contents")
    flap.list_data_objects()
    plt.close()
    flap.plot('ABES',axes='Time',options={'All points':False,'Y sep':4})

def test_slice():
    print("\n------- test slice --------")
    flap.delete_data_object('*',exp_id='*')
    print("---- Reading all test signals for time range [0,0.001]")
    d=flap.get_data('TESTDATA',name='*',
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
    #flap.plot('TESTDATA_slice')

def test_storage(signals='*',timerange=[0,0.001]):
    print("\n------- Operations on test data --------")
    flap.delete_data_object('*',exp_id='*')
    if (type(signals) is list):
        s = "["
        for sig in signals:
            s += sig+" "
        s += "]"
    else:
        s = signals
    print("Reading signal "+s+" for time range ["+str(timerange[0])+'-'+str(timerange[1])+'] with no_data=True')
    d=flap.get_data('TESTDATA',name=signals,
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':timerange},
                    no_data = True)
    print("**** Storage contents")
    flap.list_data_objects()
    print("Reading the same with data")
    d=flap.get_data('TESTDATA',name=signals,
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':timerange})
    print("**** Storage contents")
    flap.list_data_objects()

def test_coordinates():
    print("\n------- Testing adding coordinates --------")
    flap.delete_data_object('*',exp_id='*')

    print("---- Reading signal TEST-2-5 for time range [0,0.1]")
    d=flap.get_data('TESTDATA',name='TEST-2-5',
                    options={'Scaling':'Volt'},
                    object_name='TEST-1-1',
                    coordinates={'Time':[0,0.1]})
    print("**** Storage contents")
    flap.list_data_objects()
    print("**** Adding Device x coordinate")
    flap.add_coordinate('TEST-1-1',exp_id='*',coordinates=['Device x','Device z', 'Device y'])
    print("**** Storage contents")
    flap.list_data_objects()

    print("---- Reading all test signals for time range [0,0.001]")
    d=flap.get_data('TESTDATA',name='*',
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':[0,0.001]})
    print("**** Storage contents")
    flap.list_data_objects()
    print("**** Adding Device x coordinate")
    flap.add_coordinate('TESTDATA',exp_id='*',coordinates=['Device x','Device z', 'Device y'])
    print("**** Storage contents")
    flap.list_data_objects()

def test_config():
    print("\n------- Testing configuration file --------- ")
    dp = flap.config.get('Module APDCAM','Datapath',default='')
    print("Datapath: "+dp)
    d = flap.config.get_all_section('Module APDCAM')

def test_range_limits():
    s = flap.RangeSequence(1.1, 1.15, 0.2,number=13)
    start, stop = s.range_limits(limits=[3.12,3.52])
    print(start)
    print(stop)



def test_multi_slice():
    plt.close('all')
    flap.delete_data_object('*',exp_id='*')
    exp_id = '20181018.008'
    d=flap.get_data('W7X_ABES',name='ABES-[8-20]',
                    exp_id=exp_id,
                    options={'Scaling':'Volt'},
                    object_name='ABES')
    d_beam_on=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                         options={'State':{'Chop': 0, 'Defl': 0}}, object_name='Beam_on')
    flap.plot('ABES',axes='Start Time in int(Sample)',
              slicing={'Sample':d_beam_on},
              summing={'Interval(Sample) sample index':'Mean'},
              options={'Y sep':3})
    d_beam_off=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                         options={'State':{'Chop': 1, 'Defl': 0}}, object_name='Beam_off')
    flap.plot('ABES',axes='Start Time in int(Sample)',
              slicing={'Sample':d_beam_off},
              summing={'Interval(Sample) sample index':'Mean'},
              options={'Y sep':3})
    flap.list_data_objects()

def test_apsd():
    plt.close('all')
    if (False):
        # Calculating all 150 APSDs
        d = flap.get_data('TESTDATA',name='TEST*',object_name='TEST-1_S',options={'Signal':'Sin','F':[1e3,1e4]})
        print('APSD START')
        flap.apsd('TEST-1_S',output_name='TEST-1_APSD_Sin1',options={'Res':12, 'Int':10})
        print('APSD STOP')
        # Plotting one raw
        flap.plot('TEST-1_APSD_Sin1',slicing={'Row':1}, axes='Frequency',options={'All':True})
        plt.title('TEST-1-1_APSD_Sin1')
    if (False):
        # Testing with a complex signal
        plt.figure()
        d = flap.get_data('TESTDATA',name='TEST-1-1',object_name='TEST-1-1_CS',options={'Signal':'Complex-Sin'})
        flap.apsd('TEST-1-1_CS',coordinate='Time', output_name='TEST-1-1_APSD_Complex-Sin')
        flap.slice_data('TEST-1-1_APSD_Complex-Sin',slicing={'Frequency':flap.Intervals(-2e3,2e3)},
                                                             output_name='TEST-1-1_APSD_Complex-Sin_sliced')
        flap.list_data_objects()
        flap.plot('TEST-1-1_APSD_Complex-Sin_slice',axes='Frequency',options={'All':True})
        plt.title('TEST-1-1_APSD_Complex-Sin_sliced')
        flap.list_data_objects()
    if (False):
        # Testing interval selection
        d = flap.get_data('TESTDATA',name='TEST-1-1',object_name='TEST-1-1',options={'Signal':'Sin'})
        intervals = flap.Intervals(0,0.08,step=0.1,number=8)
        flap.apsd('TEST-1-1',output_name='TEST-1-1_APSD',intervals=intervals,options={'Res':12, 'Int':10})
        flap.plot('TEST-1-1_APSD')
    if (True):
        plt.close('all')
        flap.get_data('W7X_ABES',name='ABES-10',exp_id='20181018.008',object_name='ABES')
        d_beam_on=flap.get_data('W7X_ABES',exp_id='20181018.008',name='Chopper_time',
                         options={'State':{'Chop': 0, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_on', coordinates={'Time':[4,6]})
        d_beam_off=flap.get_data('W7X_ABES',exp_id='20181018.008',name='Chopper_time',
                         options={'State':{'Chop': 1, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_off', coordinates={'Time':[4,6]})
        legend = []
        flap.plot('ABES',axes='Time',options={'Y sep': 1})
        legend.append('ABES')
        flap.plot('Beam_on',axes=['Time',1],plot_type='scatter')
        legend.append('Beam on')
        flap.plot('Beam_off',axes=['Time',1.1],plot_type='scatter')
        legend.append('Beam off')
        plt.legend(legend)

        plt.figure()
        flap.apsd('ABES',output_name='ABES_APSD',intervals=d_beam_on,options={'Log':True,'Res':100,'Rang':[1e3,1e6]})
        flap.plot('ABES_APSD',options={'Log x':True, 'Log y': True, 'Y sep': 100})
        flap.apsd('ABES',output_name='ABES_APSD_off_Han',intervals=d_beam_off,options={'Han':True,'Log':True,'Res':100,'Rang':[1e3,1e6]})
        flap.apsd('ABES',output_name='ABES_APSD_off',intervals=d_beam_off,options={'Han':False,'Log':True,'Res':100,'Rang':[1e3,1e6]})
        flap.plot('ABES_APSD_off',options={'Log x':True, 'Log y': True, 'Y sep': 100})
        flap.plot('ABES_APSD_off_Han',options={'Log x':True, 'Log y': True, 'Y sep': 100})
        flap.list_data_objects()
    if (False):
        # testing calcualtion time with shorter signals
        d = flap.get_data('TESTDATA',name='TEST*',object_name='TEST_ALL',
                          options={'Signal':'Sin'}, coordinates={'Time': [0,0.05]})
        print('START_APSD')
        flap.apsd('TEST_ALL',output_name='TEST_ALL_APSD',options={'Rang':[0,3e5],'Res':1E3})
        print('STOP_APSD')
        flap.list_data_objects()

def test_detrend():
    d = flap.get_data('TESTDATA',name='TEST-1-*',object_name='TEST-1',options={'Signal':'Sin', 'Freq':[1e3,2E3]})
    flap.slice_data('TEST-1',intervals={'Time':flap.Intervals(0,0.5e-3)})
    plt.close('all')
    flap.plot('TEST-1',axes='Time')
    flap.detrend('TEST-1',options={'Trend':['poly',2]},output_name='TEST-1_detrend')
    plt.figure()
    flap.plot('TEST-1_detrend',axes='Time')

def test_mouse():

    def testmouse(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

    plt.close('all')
    fig = plt.figure()
    plt.ion()
    plt.plot(np.arange(10))
    plt.show()
    plt.draw()
    cid = fig.canvas.mpl_connect('button_press_event', testmouse)

    import time
    while(1):
        time.sleep(0.01)
        plt.pause(0.05)

def test_select():
    plt.close('all')
    d = flap.get_data('TESTDATA',name='TEST-1-1',object_name='TEST-1-1')
    d_int = flap.select_intervals('TEST-1-1',
                                  coordinate='Time',
                                  intervals={'Time':flap.Intervals(0,0.005, step=0.01, number=3)},
                                  options={'Select':None,
                                           'Length':0.0001,
                                           'Event':{'Type':'Max-weight',
                                                    'Threshold':1,
                                                    'Thr-type':'Sigma'}},
                                  plot_options={'All points':True},
                                  output_name='SELECT')
    if (d_int is not None):
        print(d_int.data)
        print(d_int.error)
        flap.slice_data('TEST-1-1',slicing={'Time':d_int},output_name='TEST-1-1_sliced')
        flap.list_data_objects()
        plt.figure()
        n_int = d_int.shape[0]
        for i in range(n_int):
            flap.plot('TEST-1-1_sliced',
                      slicing={'Interval(Time)':i},
                      axes='Rel. Time in int(Time)')

def test_w7x_detrend():
    plt.close('all')
    d = flap.get_data('W7X_ABES',
                      name=['ABES-15','ABES-20'],
                      exp_id='20181018.008',
                      object_name='ABES-20',
                      coordinates={'Time':[3,7]})
    d.plot(options={'All':True},axes='Time')
    d_beam_on=flap.get_data('W7X_ABES',exp_id='20181018.008',name='Chopper_time',
                         options={'State':{'Chop': 0, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_on', coordinates={'Time':[4,6]})
    d_tr = flap.detrend('ABES-20',coordinate='Time',
                 intervals={'Time':d_beam_on},
                 output_name='ABES-20_detrend')
    d_tr.plot(options={'All':True},axes='Time')
    flap.list_data_objects()

def test_w7x_conditional_average():
    plt.close('all')
    d = flap.get_data('W7X_ABES',
                  name='ABES-[7-28]',
                  exp_id='20180912.014',
                  object_name='ABES',
                  coordinates={'Time':[15,16]})
    d_beam_on=flap.get_data('W7X_ABES',exp_id='20181018.008',name='Chopper_time',
                         options={'State':{'Chop': 0, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_on', coordinates={'Time':[4,6]})
    d_beam_off=flap.get_data('W7X_ABES',exp_id='20181018.008',name='Chopper_time',
                         options={'State':{'Chop': 1, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_off', coordinates={'Time':[4,6]})

    if (True):
        plt.figure()
        flap.apsd('ABES',
                  intervals=d_beam_on,
                  output_name='ABES_apsd').plot(options={'Log x':True, 'Log y': True,'Y sep':10})
    if (True):
        plt.figure()
        flap.slice_data('ABES',output_name='ABES-9',slicing={'Signal name':'ABES-9'})\
          .apsd(intervals=d_beam_on).plot(options={'Log x':True, 'Log y': True})
#        flap.get_data_object('ABES_apsd').slice_data(slicing={'Signal name':'ABES-9'})\
#          .plot(options={'Log x':True, 'Log y': True})
        flap.slice_data('ABES',output_name='ABES-9',slicing={'Signal name':'ABES-9'})\
          .apsd(intervals=d_beam_off).plot(options={'Log x':True, 'Log y': True})
    if (True):  
        plt.figure()
        flap.filter_data('ABES',
                         output_name='ABES_filt',
                         coordinate='Time',
                         options={'Type':'Int','Tau':1e-5})
        flap.detrend('ABES_filt',
                     intervals=d_beam_on,
                     output_name='ABES_filt_detrend',
                     coordinate='Time')
        flap.plot('ABES_filt_detrend',
                  slicing={'Time':flap.Intervals(4.614,4.683)},
                  axes='Time',
                  options={'All':True, 'Y sep': 0.3})

def test_filter():
    plt.close('all')

    if (False):
        # This is just to test the scipy filter
        t = np.arange(1000)*1e-6
        d = np.sign(np.sin(math.pi*2*1e4*t)) + 1
        step = 1e-6
    
        plt.figure()
        plt.plot(t,d)
        tau = 100e-6
        a = np.array([1,-math.exp(-1/(tau/step))])/(1-math.exp(-1/(tau/step)))
        b = np.array([1])
        zi = [np.mean(d[0:2*int(tau/step)])]
        di,zo = signal.lfilter(b, 
                               a, 
                               d,
                               zi=zi)
        plt.plot(t,di)   
        
        plt.figure()
        plt.plot(t,d)
        tau = 10e-6
        #a = np.array([1,-math.exp(-1/(tau/step))])
        #b = np.array([1])
        a = np.array([1,-math.exp(-1/(tau/step))])
        dd = d - signal.lfilter(b, a, d)*(1-math.exp(-1/(tau/step)))
        plt.plot(t,dd)   
    if (False):
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
        plt.figure()
        d.plot(options={'Y sep':3})
        flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                           intervals=flap.Intervals(np.array([1e-4,6e-4]),np.array([2e-4,8e-4])),
                           options={'Type':'Diff','Tau':10e-6})
        flap.plot('Signal_filt',options={'Y sep':3})

        plt.figure()
        t = np.arange(2000000)*1e-6
        d = np.random.randn(len(t))
        c = flap.Coordinate(name='Time',
                            unit='Second',
                            mode=flap.CoordinateMode(equidistant=True),
                            start=0.0,
                            step=1e-6,
                            dimension_list=[0]
                            )
        d = flap.DataObject(data_array=d,coordinates=[c])
        flap.add_data_object(d,'Signal')
        flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                         options={'Type':'Int','Tau':16e-6})
        flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})
        flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})
      
        plt.figure()
        flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                         options={'Type':'Diff','Tau':16e-6})
        flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})
        flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})

        plt.figure()
        flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                         options={'Type':'Lowpass','f_high':5e4})
        flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})
        flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})

        plt.figure()
        flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                         options={'Type':'Highpass','f_low':1e4,'f_high':5e4})
        flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})
        flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})

        plt.figure()
        flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                         options={'Type':'Bandpass','f_low':5e3,'f_high':5e4})
        flap.apsd('Signal',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})
        flap.apsd('Signal_filt',options={'Log':True,'Res':20,'Range':[100,5e5]},output_name='Signal_APSD')\
           .plot(options={'Log x':True, 'Log y': True})
        
        plt.figure()
        flap.filter_data('Signal',output_name='Signal_filt',coordinate='Time',
                         options={'Type':'Bandpass','f_low':5e4,'f_high':2e5, 'Power':True, 'Inttime':20e-6})
        flap.plot('Signal_filt')
    if (True):
        d = flap.get_data('TESTDATA',name='TEST-1-1',options={'Signal':'Random','Scaling':'Digit'},object_name='test')    
        flap.list_data_objects()
        flap.filter_data('test',options={'Type':'Highpass', 'f_low':1e2,'f_high':5e4},output_name='test_filt')
        flap.plot('test_filt',axes='Time')
    
def test_saveload():
    flap.delete_data_object('*')
    d=flap.get_data('TESTDATA',name='*',
                    options={'Scaling':'Volt'},
                    object_name='TESTDATA',
                    coordinates={'Time':[0,0.001]})
    flap.slice_data('TESTDATA',slicing={'Signal name': 'TEST-1-*'},output_name='TESTDATA_slice')
    print("Storage contents before save.")
    flap.list_data_objects()
    print("Saving all storage and deleting storage contents.")
    flap.save('*','flap_save_test.dat')
    flap.delete_data_object('*')
    print("Storage contents after erasing.")
    flap.list_data_objects()
    flap.load('flap_save_test.dat')
    print("Storage contents after loading.")
    flap.list_data_objects()
    flap.delete_data_object('*')
    print("Storage contents after erasing.")
    d = flap.load('flap_save_test.dat',options={'No':True})
    print(d)
    print("Storage contents after loading with 'No storage' option.")
    flap.list_data_objects()
    flap.delete_data_object('*')
    flap.save([d,'test'],'flap_save_test.dat')
    d1 = flap.load('flap_save_test.dat',options={'No':True})
    print(d1)


def test_hanning():
    plt.close('all')
    d = flap.get_data('TESTDATA',
                      name='TEST-1-1',
                      options={'Signal': 'Random', 'Length': 2}, 
                      object_name='TEST-1-1')
    d = flap.filter_data('TEST-1-1', options={'Type':'Int', 'Tau':1e-6})
    d.apsd(options={'Res':100,'Range':[1e3,5e5],'Log':True, 'Hanning':False})\
       .plot(options={'Log x':True, 'Log y': True})
    d.apsd(options={'Res':100,'Range':[1e3,5e5],'Log':True, 'Hanning':True})\
       .plot(options={'Log x':True, 'Log y': True})
    
def test_mult():
    a1_orig = np.arange(12)
    a1_orig = a1_orig.reshape((3,4))
    a2_orig = np.arange(60)
    a2_orig = a2_orig.reshape((3,4,5))
    com_dim = [1,1]
    r, axis_source, axis_number = flap.multiply_along_axis(a1_orig, a2_orig,com_dim)
    print(axis_source)
    print(axis_number)
    for ii in range(100):
        ind = np.random.random(4)
        for i in range(r.ndim):
            ind[i] = ind[i]*r.shape[i]
        ind = ind.astype(np.int32)
        ind_1 = ind[0:2]
        ind_2 = np.concatenate((np.array([ind[2]]),np.array([ind[1]]),np.array([ind[3]])))
        if (r[tuple(ind)] != a1_orig[tuple(ind_1)]*a2_orig[tuple(ind_2)]):
            print(ind)
            break
    print("Test passed.")
    pass
       

def test_webapi():
    flap.delete_data_object('*')
    d = flap.get_data('W7X_WEBAPI', name='AUG-2',
                      exp_id='20181016.037',
                      options={'Scale Time': True},
                      object_name='AUG2_data',
                      coordinates={'Time [s]': [2, 3]})
    flap.save('AUG2_data','20181016.037_AUG2.dat')

    plt.contourf(np.linspace(1,45,num=45),d.coordinates[0].values,d.data[:,:,1])
    
def test_cpsd():
    plt.close('all')
    if (False):
        d =  flap.get_data('TESTDATA', name='TEST-1-[1-8]', options={'Signal':'Random'}, object_name='TESTDATA')
        flap.cpsd('TESTDATA',
                  options={'Norm':True,'Interval':50},
                  output_name='TESTDATA_cpsd')
        flap.abs('TESTDATA_cpsd',output_name='TESTDATA_cpsd_abs')
        flap.plot('TESTDATA_cpsd_abs',
                  axes='Frequency',
                  slicing={'Row (Ref)':1,'Row':2},
                  options={'Log y':True,'Log x':True, 'Error':False})
        flap.error_value('TESTDATA_cpsd_abs').plot(slicing={'Row (Ref)':1,'Row':2})
        return
        flap.slice_data('TESTDATA_cpsd',
                        slicing={'Frequency':flap.Intervals(1e4,1e5)},
                        summing={'Frequency':'Mean'}).plot(axes='Row (Ref)',options={'Y sep': 1.5})
        return
        flap.plot
        flap.list_data_objects()
        d_cps = flap.get_data_object('TESTDATA_cpsd')\
                  .abs() \
                  .plot(axes='Frequency',
                        slicing={'Signal name (Ref)':'TEST-1-1'},
                        options={'Log x':True, 'Log y':True, 'Y sep':1})
    if (True):
        exp_id = '20181018.003'
        timerange=[4.1,4.7]
        flap.get_data('W7X_ABES',exp_id=exp_id,name='ABES-[8-20]',object_name='ABES',coordinates={'Time':timerange})
        d_beam_on=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                         options={'State':{'Chop': 0, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_on')
        d_beam_off=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                         options={'State':{'Chop': 1, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_off')
        flap.cpsd('ABES',
                  intervals={'Time':d_beam_on},
                  coordinate='Time',
                  options={'Norm':True},
                  output_name='ABES_cpsd'
                  )
        gs = GridSpec(2, 2)
        plt.subplot(gs[0,0])
        plot_1 = flap.plot('ABES_cpsd',
                           slicing={'Signal name':'ABES-19','Signal name (Ref)':'ABES-20'},
                           axes='Frequency',
                           options={'Log x':True})
        plot_1.plt_axis_list[0].set_title('Coherency ABES-20 - ABES-19')
        plt.subplot(gs[1,0])
        plot_2 = flap.plot('ABES_cpsd',
                           slicing={'Signal name':'ABES-12','Signal name (Ref)':'ABES-20'},
                           axes='Frequency',
                           options={'Log x':True})
        plot_2.plt_axis_list[0].set_title('Coherency ABES-20 - ABES-12')
        plt.subplot(gs[:,1])
        plot_3 = flap.abs('ABES_cpsd').plot(
                               slicing={'Signal name (Ref)':'ABES-20'},
                               axes='Frequency',
                               options={'Log x':True})
        plot_3.plt_axis_list[0].set_title('Coherencies relative to ABES-20')
        
        plt.figure()
        gs = GridSpec(2,1)
        plt.subplot(gs[0,:])
        flap.abs('ABES_cpsd').slice_data(slicing={'Signal name (Ref)':'ABES-20',
                                                  'Frequency':flap.Intervals(1e3,1e4)},
                                         summing={'Frequency':'Mean'}
                                         ).plot(axes='Channel')
        plt.subplot(gs[1,:])
        flap.phase('ABES_cpsd').slice_data(slicing={'Signal name (Ref)':'ABES-20',
                                                  'Frequency':flap.Intervals(1e3,1e4)},
                                         summing={'Frequency':'Mean'}
                                         ).plot(axes='Channel')
        plt.figure()
        flap.abs('ABES_cpsd').slice_data(slicing={'Signal name (Ref)':'ABES-20'},
                                         ).plot(axes=['Frequency','Channel'],
                                                plot_type='image',
                                                options={'Log x':True,'Log z':True,'Z range':[0.01,1]})
       
        flap.apsd('ABES',intervals={'Time':d_beam_on},output_name='ABES_apsd')
        plt.figure()
        gs = GridSpec(2,2)
        plt.subplot(gs[:,0])
        plt_APSD = flap.plot('ABES_apsd',axes=['Frequency','Channel'],
                                                plot_type='image',
                                                options={'Log x':True,'Log z':True})
        plt.subplot(gs[0,1])
        plt_1 = flap.plot('ABES_apsd',slicing={'Signal name':'ABES-18'},axes=['Frequency'],
                                                plot_type='xy',
                                                options={'Log x':True,'Log y':True,'Error':False})
        flap.plot('ABES_apsd',slicing={'Signal name':'ABES-19'},options={'Error':False})
        flap.plot('ABES_apsd',slicing={'Signal name':'ABES-20'},options={'Error':False})
        plt_1.plt_axis_list[0].set_title('ABES-18...20')
        plt.subplot(gs[1,1])
        plt_2 = flap.plot('ABES_apsd',slicing={'Signal name':'ABES-13'},axes=['Frequency'],
                                                plot_type='xy',
                                                options={'Log x':True,'Log y':True,'Error':False})
        flap.plot('ABES_apsd',slicing={'Signal name':'ABES-14'},options={'Error':False})
        flap.plot('ABES_apsd',slicing={'Signal name':'ABES-15'},options={'Error':False})
        plt_2.plt_axis_list[0].set_title('ABES-13...15')
        

def test_plot():
    plt.close('all')
    flap.set_plot_id(None)

    length = 0.01
    flap.get_data('TESTDATA', name='TEST-1-1', options={'Signal':'Sin','Length':length}, object_name='TEST-1-1')
    flap.get_data('TESTDATA', name='TEST-1-2', options={'Signal':'Const.','Length':length}, object_name='TEST-1-2')
    flap.get_data('TESTDATA', name='TEST-1-3', options={'Signal':'Const.','Length':length}, object_name='TEST-1-3')
    flap.get_data('TESTDATA', name='TEST-1-[1-5]', options={'Signal':'Sin','Length':length}, object_name='TESTDATA')
    flap.get_data('TESTDATA', name='TEST-1-1', options={'Signal':'Complex-Sin','Length':length}, object_name='TEST-1-1_comp')
    flap.get_data('TESTDATA', name='TEST-1-2', options={'Signal':'Complex-Sin','Length':length}, object_name='TEST-1-2_comp')
    flap.get_data('TESTDATA', name='TEST-1-[1-5]', options={'Signal':'Complex-Sin','Length':length}, object_name='TESTDATA_comp')
    
    # Creating a single plot in the upper left corner    
    gs = GridSpec(2, 2)
    plt.subplot(gs[0,0])
    plot_1 = flap.plot('TEST-1-1')
    # Creating a multi xy on the righ side
    plt.subplot(gs[:,1])
    plot_2 = flap.plot('TESTDATA')
    plot_2 = flap.plot('TESTDATA',slicing={'Signal name':['TEST-1-2','TEST-1-3']})
    # Overplotting into the first plot
    flap.plot('TEST-1-3',plot_id=plot_1)
    
    
    plt.figure()
    plot_3 = flap.plot('TEST-1-1_comp')
    plot_3 = flap.plot('TEST-1-2_comp')
    
    plt.figure()
    gs = GridSpec(1, 2)
    plt.subplot(gs[0,0])
    plot_4 = flap.abs('TESTDATA_comp').plot()
    plt.subplot(gs[0,1])
    plot_5 = flap.phase('TESTDATA_comp').plot(options={'Y sep':10})
#    plot_4 = flap.plot('TESTDATA_comp')
    
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


def problems():
    d=flap.get_data('W7X_ABES',exp_id='20181018.003',name=['ABES-1[5,6]'],\
                    options={'Scaling':'Volt'},object_name='ABES',\
                    coordinates=flap.Coordinate(name='Time',c_range=[1,1.00001]))
    
    #test_config()
#test_storage()
#test_coordinates()
#test_plot_xy()
#test_plot_multi_xy()
#test_coordinate_data()
#test_slice()
#test_plot_object()
#test_plot_multi_xy()
#test_range_limits()
#test_errorplot()
#test_multi_slice()
#test_apsd()
#test_detrend()
#test_mouse()
#test_select()
#test_w7x_detrend()
#test_w7x_conditional_average()
#test_filter()
#test_webapi()
#test_saveload()
#test_hanning()
#test_mult()
#test_cpsd()
#test_plot()
problems()