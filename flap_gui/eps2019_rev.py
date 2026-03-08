# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:33:29 2019

@author: Zoletnik
"""
import scipy.io 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import time
import numpy as np
import math
import copy
import h5py

import flap
import flap_w7x_abes
# import flap_mdsplus
# import flap_w7x_mdsplus
# import flap_w7x_camera

flap_w7x_abes.register()
# flap_mdsplus.register()
# flap_w7x_mdsplus.register()
# flap_w7x_camera.register()


plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['axes.labelsize'] = 35  # 28 for paper
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.width'] = 2
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['legend.fontsize'] = 28

def change_coord_name(data_name,coord_from, coord_to):
    d = flap.get_data_object(data_name)
    d.get_coordinate_object(coord_from).unit.name = coord_to
    flap.add_data_object(d,data_name)

def three_spectra(exp_id,timerange,signals=['ABES-12','ABES-14','ABES-21']):
    flap.get_data('W7X_ABES',exp_id=exp_id,coordinates={'Time':timerange},name='ABES-[10-30]',object_name='ABES')        
    d_beam_on=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                        options={'State':{'Chop': 0, 'Defl': 0},'Start':1000,'End':-1000},\
                        object_name='Beam_on',
                        )
    legend = []
    for i,sig in enumerate(signals):
        flap.get_data_object('ABES', exp_id=exp_id)\
            .slice_data({'Signal name':sig})\
            .apsd(coordinate='Time',
                  intervals={'Time':d_beam_on},
                  options={'Interval':1, 'Range':[100,1E6],'Res':20})\
            .plot(options={'Log x': True, 'Log y':True, 'Error':False, 'Y range':[3e-4,1e2]})   
        if (i != 0):
            plt.ylabel('')
    d=flap.add_coordinate('ABES',coordinates='Device R').slice_data(slicing={'Signal name':signals})         
    r = d.get_coordinate_object('Device R').values
    legend.append('R={:5.3f}cm (Island out)'.format(r[0]))
    legend.append('R={:5.3f}cm (O-point)'.format(r[1]))
    legend.append('R={:5.3f}cm (Plasma edge)'.format(r[2]))
    d_on = flap.get_data_object('Beam_on', exp_id=exp_id)
    timerange, xx = flap.get_data_object('ABES').coordinate_range('Time')
    plt.title(exp_id+' [{:5.3f},{:5.3f}]s'.format(timerange[0],timerange[1]))
    plt.ylabel('Power [a.u.]')
    plt.legend(legend)
    flap.delete_data_object('*')

def three_spectra_highiota(exp_id,timerange,signals=['ABES-16','ABES-19','ABES-23'],label=None):
    flap.get_data('W7X_ABES',exp_id=exp_id,coordinates={'Time':timerange},name='ABES-[10-30]',object_name='ABES')        
    d_beam_on=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                        options={'State':{'Chop': 0, 'Defl': 0},'Start':1000,'End':-1000},\
                        object_name='Beam_on',
                        )
    legend = []
    for i,sig in enumerate(signals):
        flap.get_data_object('ABES', exp_id=exp_id)\
            .slice_data({'Signal name':sig})\
            .apsd(coordinate='Time',
                  intervals={'Time':d_beam_on},
                  options={'Interval':1, 'Range':[100,1E6],'Res':20})\
            .plot(options={'Log x': True, 'Log y':True, 'Error':False, 'Y range':[5e-5,1e0]})   
        if (i != 0):
            plt.ylabel('')
    d=flap.add_coordinate('ABES',coordinates='Device R').slice_data(slicing={'Signal name':signals})         
    r = d.get_coordinate_object('Device R').values
    legend.append('R={:5.3f}cm (SOL)'.format(r[0]))
    legend.append('R={:5.3f}cm (O-point)'.format(r[1]))
    legend.append('R={:5.3f}cm (Plasma edge)'.format(r[2]))
    d_on = flap.get_data_object('Beam_on', exp_id=exp_id)
    timerange, xx = flap.get_data_object('ABES').coordinate_range('Time')
    plt.title(exp_id+' [{:5.3f},{:5.3f}]s'.format(timerange[0],timerange[1]))
    plt.ylabel('Power [a.u.]')
    if (label is not None):
        plt.text(2e5,1e-2,label,color='k',fontsize=40)
    plt.legend(legend)
    flap.delete_data_object('*')
    
def three_spectra_fastchop(exp_id,timerange,signals=None,label=None):
    legend = []
    for i,sig in enumerate(signals):
        flap.get_data_object('ABES', exp_id=exp_id)\
            .slice_data({'Signal name':sig})\
            .apsd(coordinate='Time',
                  options={'Interval':1, 'Range':[100,1E6],'Res':10})\
            .plot(options={'Log x': True, 'Log y':True, 'Error':False, 'X range':[50,1.5e6]})   
        if (i != 0):
            plt.ylabel('')
    r = flap.get_data_object('ABES').slice_data(slicing={'Signal name':signals}).get_coordinate_object('R').values
    legend.append('R={:5.3f}cm'.format(r[0]))
    legend.append('R={:5.3f}cm'.format(r[1]))
    legend.append('R={:5.3f}cm'.format(r[2]))
    timerange, xx = flap.get_data_object('ABES').coordinate_range('Time')
    plt.title(exp_id+' [{:5.3f},{:5.3f}]s'.format(timerange[0],timerange[1]))
    if (label is not None):
        plt.text(2e5,2e-4,label,color='k',fontsize=40)
    plt.ylabel('Power [a.u.]')
    plt.legend(legend)
    
def plot_cr_abes_correlation(exp_id='20181018.003',refl_channel='CR-B',
                             abes_channels='ABES-[6-21]',interval=215,
                             time_lag_res = 1e-6,
                             time_lag_range=[-1000e-6,1000e-6], 
                             abes_ref = 'ABES-20',
                             filter_type='Int',
                             filter_low=1e3,
                             filter_high=1e4,
                             tau_diff=100*1e-6,
                             tau_int=16*1e-6,
                             interval_n=1,
                             corr_range=[-0.3,0.3],
                             subplot=None,
                             label=None):

    
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['figure.titlesize'] = 'medium'
    
    if (type(interval) is not list):
        _interval = [interval]
    else:
        _interval = interval
    read_sweeps(exp_id=exp_id)
    df = flap.get_data_object('CR_Freq',exp_id=exp_id)
    int_range, int_low, int_high = df.coordinate('Time',index=[np.array(_interval)-1])
    time_interval = [int_low[0],int_high[-1]]
    proc_intervals=flap.Intervals(int_low,int_high)

    print("Reading CR data.")
    flap.get_data('W7X_MDSPlus',
                  exp_id=exp_id,
                  coordinates={'Time':time_interval},
                  name=['CR-B','CR-D','CR-E','CR-C'],
                  object_name='CR') 
    print("Reading ABES data.")
    flap.get_data('W7X_ABES',exp_id=exp_id,coordinates={'Time':time_interval},name=abes_channels,object_name='ABES')
    flap.add_coordinate('ABES','Device R')
    change_coord_name('ABES','Device R', 'R')
    flap.list_data_objects()

    print("Filtering data.")
    if (filter_type == 'Bandpass'):        
        flap.filter_data('ABES',coordinate='Time',
                         intervals=proc_intervals,
                         options={'Type':'Bandpass','f_low':filter_low,'f_high':filter_high},
                         output_name='ABES')
        flap.filter_data('CR',coordinate='Time',
                         intervals=proc_intervals,
                         options={'Type':'Bandpass','f_low':filter_low,'f_high':filter_high},
                         output_name='CR')
    else:
        flap.filter_data('ABES',intervals=proc_intervals,options={'Type':'Diff','Tau':tau_diff},output_name='ABES')
        flap.filter_data('ABES',intervals=proc_intervals,options={'Type':'Int','Tau':tau_int},output_name='ABES')
        flap.filter_data('CR',intervals=proc_intervals,options={'Type':'Diff','Tau':tau_diff},output_name='CR')
        flap.filter_data('CR',intervals=proc_intervals,options={'Type':'Int','Tau':tau_int},output_name='CR')

    flap.slice_data('ABES',
                    slicing={'Time':flap.get_data_object('CR')},
                    options={'Interp':'Linear'},
                    output_name='ABES')
    print("Calculating correlations.")    
    d_ccf = flap.ccf('ABES',
             ref='CR',
             intervals=proc_intervals,
             coordinate='Time',
             options={'Range':time_lag_range,'Res':time_lag_res,'Interval':interval_n,'Norm':True}
             )
    tlag = d_ccf.get_coordinate_object('Time lag')
    tlag.unit.unit = 'microsecond'
    tlag.start = tlag.start * 1E6
    tlag.step[0] = tlag.step[0] * 1e6
    flap.add_data_object(d_ccf,'CR_CCF_ABES')
    
    flap.abs_value('CR_CCF_ABES',output_name='CR_CCF_ABES_abs')
    bes_chlist = flap.get_data_object('ABES').coordinate('Signal name',index=[0,...])[0].flatten()
    bes_rlist = flap.get_data_object('ABES').coordinate('R',index=[0,...])[0].flatten()
    refl_chlist = flap.get_data_object('CR').coordinate('Signal name',index=[0,...])[0].flatten()

    if (subplot is not None):
        plt.subplot(subplot[0])
    else: 
        gs = GridSpec(1, 7)  
        plt.subplot(gs[0,:3])
    flap.plot('CR_CCF_ABES_abs', 
              axes=['Time lag','R'], 
              slicing={'Signal name (Ref)':refl_channel},
              plot_type='contour',
              options={'Z range':corr_range,'Colormap':'bwr','Level':250,'Colorbar':True})
    plt.title("{:s}  [{:5.3f}-{:5.3f}s]  Ref:{:s}".format(exp_id,time_interval[0],time_interval[1],refl_channel))
    plt.xlabel('Time lag [$\mu$s]')
#    plt.plot([0,0],plt.ylim(),color='k')
    xlim = plt.xlim()
    ylim = plt.ylim()
#    plt.text(xlim[0]+(xlim[1]-xlim[0])/20,ylim[1]-(ylim[1]-ylim[0])/15,'(a)',color='k',fontsize=25)
    plt.text(xlim[1]-(xlim[1]-xlim[0])/3,ylim[1]-(ylim[1]-ylim[0])/15,'CR-ABES',color='k',fontsize=25)
    plt.plot(plt.xlim(),[6.21,6.21],linestyle='dashed',color='k')
    plt.plot(plt.xlim(),[6.26,6.26],linestyle='dashed',color='k')
    plt.plot(plt.xlim(),[6.235,6.235],linestyle='dotted',color='k')
    if (label is not None):
        xr = plt.xlim()
        yr = plt.ylim()
        plt.text(xr[0]+(xr[1]-xr[0])*0.05,yr[0]+(yr[1]-yr[0])*0.9,label[0],color='k',fontsize=40)

    d_ccf = flap.ccf('ABES',
             intervals=proc_intervals,
             coordinate='Time',
             options={'Range':time_lag_range,'Res':time_lag_res,'Interval':interval_n,'Norm':True}
             )
    tlag = d_ccf.get_coordinate_object('Time lag')
    tlag.unit.unit = 'microsecond'
    tlag.start = tlag.start * 1E6
    tlag.step[0] = tlag.step[0] * 1e6
    flap.add_data_object(d_ccf,'CCF_ABES')
    
    if (subplot is not None):
        plt.subplot(subplot[1])
    else: 
        plt.subplot(gs[0,4:])
    p = flap.plot('CCF_ABES', 
                  slicing={'Signal name (Ref)':abes_ref},
                  axes=['Time lag','R'], 
                  plot_type='contour',
                  options={'Z range':[-1,1],'Colormap':'bwr','Level':250})
    ref_r = float(bes_rlist[np.nonzero(bes_chlist == abes_ref)[0]])
    plt.title("{:s}  [{:5.3f}-{:5.3f}s]  Ref R={:5.3f} m".format(exp_id,time_interval[0],time_interval[1],ref_r))
#    plt.plot([0,0],plt.ylim(),color='k')
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.xlabel('Time lag [$\mu$s]')
#    plt.text(xlim[0]+(xlim[1]-xlim[0])/20,ylim[1]-(ylim[1]-ylim[0])/15,'(b)',color='k',fontsize=25)
    plt.text(xlim[1]-(xlim[1]-xlim[0])/3,ylim[1]-(ylim[1]-ylim[0])/15,'ABES-ABES',color='k',fontsize=25)
    plt.plot(0,ref_r,'kd-',markersize=20)
    plt.plot(plt.xlim(),[6.21,6.21],linestyle='dashed',color='k')
    plt.plot(plt.xlim(),[6.26,6.26],linestyle='dashed',color='k')
    plt.plot(plt.xlim(),[6.235,6.235],linestyle='dotted',color='k')
    if (label is not None):
        xr = plt.xlim()
        yr = plt.ylim()
        plt.text(xr[0]+(xr[1]-xr[0])*0.05,yr[0]+(yr[1]-yr[0])*0.9,label[1],color='k',fontsize=40)

def read_sweeps(exp_id='20181018.003'):
    try:
        flap.get_data_object('CR_Freq',exp_id=exp_id)
    except:
        mat = scipy.io.loadmat(exp_id[2:8]+exp_id[9:]+'_BNCSettings.mat')
        time = mat['BNC'][0][0][2][0,:]
        freq = mat['BNC'][0][0][2][1,:]
        c = flap.Coordinate(mode=flap.CoordinateMode(equidistant=True,range_symmetric=False),
                            start=0.0005,
                            step=0.02,
                            name='Time',
                            unit='Second',
                            value_ranges=[0,0.019],
                            dimension_list=[0]
                            )
        c1 = flap.Coordinate(mode=flap.CoordinateMode(equidistant=True,range_symmetric=False),
                            start=1,
                            step=1,
                            name='Interval number',
                            unit='',
                            dimension_list=[0]
                            )
        df = flap.DataObject(data_array=freq,
                              coordinates=[c,c1],
                              data_unit=flap.Unit(name='RF frequency',unit='GHz'),
                              exp_id=exp_id
                              )
        flap.add_data_object(df,'CR_Freq')
        
def eps2019(fig_no=None):
    
    if (fig_no == 3):
        plt.close('all')
        fig=plt.figure(figsize=(25,10))
        exp_id = '20181018.003'
        flap.get_data('W7X_ABES',exp_id=exp_id,coordinates={'Time':[4.12,4.13]},name='ABES-[6-20]',object_name='ABES')
        flap.filter_data('ABES',coordinate='Time',options={'Type':'Int','Tau':2e-5},output_name='ABES')
        flap.add_coordinate('ABES',exp_id=exp_id,coordinates='Device R')
        change_coord_name('ABES','Device R', 'R')
        gs = GridSpec(1,10)  
        plt.subplot(gs[0,:4])
        flap.plot('ABES',plot_type='contour',axes=['R','Time'],
                  options={'Colormap':'Reds','Z range':[0,1.5],'Colorbar':True,'Levels':255})
        plt.title('Sodium beam light')
        xr = plt.xlim()
        yr = plt.ylim()
        plt.text(xr[0]+(xr[1]-xr[0])*0.85,yr[0]+(yr[1]-yr[0])*0.9,'(a)',color='k',fontsize=40)
        plt.plot([6.21,6.21],plt.ylim(),linestyle='dashed',color='k')
        plt.plot([6.26,6.26],plt.ylim(),linestyle='dashed',color='k')
        plt.plot([6.235,6.235],plt.ylim(),linestyle='dotted',color='k')
        plt.subplot(gs[0,6:])
        flap.slice_data('ABES',summing={'Time':'Mean'},output_name='ABES_profile')
        d = flap.get_data_object('ABES')
        dp = flap.get_data_object('ABES_profile')
        d.data = d.data - dp.data.reshape((1,len(dp.data)))
        minmax = max([math.fabs(np.amin(d.data)),math.fabs(np.amax(d.data))])
        zrange = [-minmax,minmax]
        d.plot(plot_type='contour',axes=['R','Time'],options={'Colormap':'bwr','Z range':zrange,'Levels':255})
        plt.title('Deviation from mean')
        xr = plt.xlim()
        yr = plt.ylim()
        plt.text(xr[0]+(xr[1]-xr[0])*0.85,yr[0]+(yr[1]-yr[0])*0.9,'(b)',color='k',fontsize=40)
        plt.plot([6.21,6.21],plt.ylim(),linestyle='dashed',color='k')
        plt.plot([6.26,6.26],plt.ylim(),linestyle='dashed',color='k')
        plt.plot([6.235,6.235],plt.ylim(),linestyle='dotted',color='k')


        fig.savefig('eps2019_sz_3.png',dpi=300)

    if (fig_no == 4):
        flap.delete_data_object('*')
        plt.close('all')
        fig=plt.figure(figsize=(13,10))
        three_spectra('20181018.003',[4.,4.75])
        fig.savefig('eps2019_sz_4.png',dpi=300)
        
    if (fig_no == 5):
        # Two correlation plots, int-diff filter
        plt.close('all')
        flap.delete_data_object('*')
        exp_id='20181018.003'
        abes_time_interval = [4,4.7]
        abes_channels='ABES-[6-21]'
        print("Reading ABES data.")
        flap.get_data('W7X_ABES',exp_id=exp_id,coordinates={'Time':abes_time_interval},name=abes_channels,object_name='ABES')
        flap.add_coordinate('ABES',coordinates='Device R')
        change_coord_name('ABES','Device R', 'R')
        d_beam_on=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                         options={'State':{'Chop': 0, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_on')
        d_beam_off=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                         options={'State':{'Chop': 1, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_off')
        flap.filter_data('ABES',intervals={'Time':d_beam_on},options={'Type':'Diff','Tau':100e-6},output_name='ABES')
        flap.filter_data('ABES',intervals={'Time':d_beam_on},options={'Type':'Int','Tau':16e-6},output_name='ABES')
        flap.filter_data('ABES',intervals={'Time':d_beam_off},options={'Type':'Diff','Tau':100e-6},output_name='ABES')
        flap.filter_data('ABES',intervals={'Time':d_beam_off},options={'Type':'Int','Tau':16e-6},output_name='ABES')
        print('Calculating CCF')
        d_ccf = flap.ccf('ABES',intervals={'Time':d_beam_on},options={'Norm':True,'Range':[-1000e-6,1000e-6],'Res':3e-6})
        coord = d_ccf.get_coordinate_object('Time lag')
        coord.unit.unit = '$\mu$s'
        coord.start = coord.start * 1e6
        coord.step[0] = coord.step[0]*1e6
        d_ccf.get_coordinate_object
        flap.add_data_object(d_ccf,'ABES_CCF')
        
        fig = plt.figure(figsize=(35,10))
        gs = GridSpec(1,9)  
        plt.subplot(gs[0,:4])
        ref = 'ABES-12'
        corr2d = flap.slice_data('ABES_CCF',slicing={'Signal name (Ref)':ref})
        ref_r = corr2d.get_coordinate_object('R (Ref)').values
        corr2d.plot(plot_type='contour',axes=['Time lag','R'],
                    options={'Colormap':'bwr','Z range':[-1,1],'Levels':250})
        plt.title(exp_id+' [{:5.3f},{:5.3f}]s'.format(abes_time_interval[0],abes_time_interval[1]))
        plt.plot(0,ref_r,'kd-',markersize=20)
        plt.plot(plt.xlim(),[6.21,6.21],linestyle='dashed',color='k')
        plt.plot(plt.xlim(),[6.26,6.26],linestyle='dashed',color='k')
        plt.plot(plt.xlim(),[6.235,6.235],linestyle='dotted',color='k')
        xr = plt.xlim()
        yr = plt.ylim()
        plt.text(xr[0]+(xr[1]-xr[0])*0.05,yr[0]+(yr[1]-yr[0])*0.9,'(a)',color='k',fontsize=40)

        plt.subplot(gs[0,5:])
        ref = 'ABES-20'
        corr2d = flap.slice_data('ABES_CCF',slicing={'Signal name (Ref)':ref})
        ref_r = corr2d.get_coordinate_object('R (Ref)').values
        corr2d.plot(plot_type='contour',axes=['Time lag','R'],
                    options={'Colormap':'bwr','Z range':[-1,1],'Levels':250})
        plt.title(exp_id+' [{:5.3f},{:5.3f}]s'.format(abes_time_interval[0],abes_time_interval[1]))
        plt.plot(0,ref_r,'kd-',markersize=20)
        plt.plot(plt.xlim(),[6.21,6.21],linestyle='dashed',color='k')
        plt.plot(plt.xlim(),[6.26,6.26],linestyle='dashed',color='k')
        plt.plot(plt.xlim(),[6.235,6.235],linestyle='dotted',color='k')
        xr = plt.xlim()
        yr = plt.ylim()
        plt.text(xr[0]+(xr[1]-xr[0])*0.05,yr[0]+(yr[1]-yr[0])*0.9,'(b)',color='k',fontsize=40)

        fig.savefig('eps2019_sz_5.png',dpi=300)        

    if (fig_no == 6):
        flap.delete_data_object('*')
        plt.close('all')
        fig=plt.figure(figsize=(25,10))
        plt.subplot(1,2,1)
        three_spectra_highiota('20180912.014',[18,20],signals = ['ABES-16','ABES-19','ABES-23'],label='(a)')

        # Low frequency correlations in ABES, high iota conf
        plt.subplot(1,2,2)
        flap.delete_data_object('*')
        exp_id='20180912.014'
        abes_time_interval = [18,20]
        abes_channels='ABES-[10-27]'
        print("Reading ABES data.")
        flap.get_data('W7X_ABES',exp_id=exp_id,coordinates={'Time':abes_time_interval},name=abes_channels,object_name='ABES')
        flap.add_coordinate('ABES',coordinates='Device R')
        change_coord_name('ABES','Device R', 'R')
        d_beam_on=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                         options={'State':{'Chop': 0, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_on')
        d_beam_off=flap.get_data('W7X_ABES',exp_id=exp_id,name='Chopper_time',
                         options={'State':{'Chop': 1, 'Defl': 0},'Start':1000,'End':-1000},\
                         object_name='Beam_off')
        flap.filter_data('ABES',intervals={'Time':d_beam_on},options={'Type':'Int','Tau':16e-6},output_name='ABES')
        flap.filter_data('ABES',intervals={'Time':d_beam_on},options={'Type':'Diff','Tau':100e-6},output_name='ABES')
        print('Calculating CCF')
        d_ccf = flap.ccf('ABES',intervals={'Time':d_beam_on},options={'Norm':True,'Range':[-300e-6,300e-6],'Res':3e-6})
        coord = d_ccf.get_coordinate_object('Time lag')
        coord.unit.unit = '$\mu$'
        coord.start = coord.start * 1e6
        coord.step[0] = coord.step[0]*1e6
        coord.unit.unit='$\mu$s'
        d_ccf.get_coordinate_object
        flap.add_data_object(d_ccf,'ABES_CCF')
        ref=['ABES-23']
        corr2d = flap.slice_data('ABES_CCF',slicing={'Signal name (Ref)':ref})
        ref_r = corr2d.get_coordinate_object('R (Ref)').values
        corr2d.plot(plot_type='contour',axes=['Time lag','R'],
                    options={'Colormap':'bwr','Z range':[-1,1],'Levels':250})
        plt.plot(0,ref_r,'kd-',markersize=20)
        plt.plot(plt.xlim(),[6.19,6.19],linestyle='dashed',color='k')
        plt.plot(plt.xlim(),[6.21,6.21],linestyle='dashed',color='k')
        plt.plot(plt.xlim(),[6.20,6.20],linestyle='dotted',color='k')
        xr = plt.xlim()
        yr = plt.ylim()
        plt.text(xr[0]+(xr[1]-xr[0])*0.85,yr[0]+(yr[1]-yr[0])*0.85,'(b)',color='k',fontsize=40)
        plt.title(exp_id+' [{:5.3f},{:5.3f}]s'.format(abes_time_interval[0],abes_time_interval[1])+' ABES Corrrelations')
        fig.savefig('eps2019_sz_6.png',dpi=300)

    if (fig_no == 7):
        # power spectra and radial correlation with large island
        flap.delete_data_object('*')
        plt.close('all')
        fig=plt.figure(figsize=(25,10))
        exp_id='20171207.027'
        timerange=[1,1.5]

        flap_w7x_abes.proc_chopsignals(exp_id=exp_id,
                           signals=['ABES-[6-21]'],
                           timerange=timerange,
                           on_options={'Start':1,'End':-1.5}, 
                           off_options={'Start':1,'End':-1.5},
                           test=False)
        flap.add_coordinate('ABES',coordinates='Device R')
        change_coord_name('ABES','Device R', 'R')
        plt.subplot(1,2,1)
        three_spectra_fastchop(exp_id,timerange=timerange,signals=['ABES-13','ABES-15','ABES-19'],label='(a)')
        # Low frequency correlations in ABES
        plt.subplot(1,2,2)
        flap.filter_data('ABES',options={'Type':'Diff','Tau':100e-6},output_name='ABES')
        flap.filter_data('ABES',options={'Type':'Int','Tau':16e-6},output_name='ABES')
        d_ccf = flap.ccf('ABES',
                         coordinate='Time',
                         options={'Range':[-3e-4,3e-4],'Res':1e-5,'Norm':True})
        coord = d_ccf.get_coordinate_object('Time lag')
        coord.unit.unit = '$\mu$'
        coord.start = coord.start * 1e6
        coord.step[0] = coord.step[0]*1e6
        coord.unit.unit='$\mu$s'
        d_ccf.get_coordinate_object
        flap.add_data_object(d_ccf,'ABES_CCF')
        ref=['ABES-19']
        corr2d = flap.slice_data('ABES_CCF',slicing={'Signal name (Ref)':ref})
        ref_r = corr2d.get_coordinate_object('R (Ref)').values
        corr2d.plot(plot_type='contour',axes=['Time lag','R'],
                    options={'Colormap':'bwr','Z range':[-1,1],'Levels':250})
        plt.plot(0,ref_r,'kd-',markersize=20)
        plt.plot(plt.xlim(),[6.21,6.21],linestyle='dashed',color='k')
        plt.plot(plt.xlim(),[6.26,6.26],linestyle='dashed',color='k')
        plt.plot(plt.xlim(),[6.235,6.235],linestyle='dotted',color='k')
        xr = plt.xlim()
        yr = plt.ylim()
        plt.text(xr[0]+(xr[1]-xr[0])*0.85,yr[0]+(yr[1]-yr[0])*0.85,'(b)',color='k',fontsize=40)
        plt.title(exp_id+' [{:3.1f},{:3.1f}]s'.format(timerange[0],timerange[1])+' ABES Corr.')
        fig.savefig('eps2019_sz_7.png',dpi=300)

    if (fig_no == 9):
        # CR-ABES correlations
        plt.close('all')
        flap.delete_data_object('*')
        fig = plt.figure(figsize=(25,25))
        gs = GridSpec(7, 7)  
        plot_cr_abes_correlation(refl_channel='CR-B',
                                 interval=[191],
                                 time_lag_res = 1e-6,
                                 time_lag_range=[-200e-6,200e-6], 
                                 abes_ref = 'ABES-11',
                                 filter_type='Bandpass',
                                 filter_low=1e4,
                                 filter_high=3e4,
                                 subplot=[gs[:3,:3],gs[:3,4:]],
                                 label=['(a)','(b)'])
        plot_cr_abes_correlation(refl_channel='CR-B',
                                 interval=[178],
                                 time_lag_res = 1e-6,
                                 time_lag_range=[-200e-6,200e-6], 
                                 abes_ref = 'ABES-21',
                                 filter_type='Bandpass',
                                 filter_low=1e4,
                                 filter_high=3e4,
                                 subplot=[gs[4:,:3],gs[4:,4:]],
                                 label=['(c)','(d)'])
        fig.savefig('eps2019_sz_9.png',dpi=300)


    if (fig_no == 10):
        # Density profile
        # Reads from a reconstructed data file rom M. Vecsei
        
        recon_data=h5py.File('20181018.003_4.00000-4.80000.hdf5', 'r', libver='earliest')
        shot_data = {
                "time_list": np.matrix(recon_data["/Time list"].value),
                "meas_light_r_coord": recon_data['/Measured light R coord'].value,
                "meas_light_beam_coord": recon_data['/Measured light beam axis coord'].value,
                "meas_light_xyz_coord": recon_data['/Measured light xyz coord'].value,
                "recon_r_coord": recon_data['/Reconstructed R coord'].value,
                "recon_beam_coord": recon_data['/Reconstructed beam axis coord'].value,
                "recon_xyz_coord": recon_data['/Reconstructed xyz coord'].value,
                "meas_light_profile": recon_data['/Measured light profile'].value,
                "meas_light_profile_error": recon_data['/Measured light profile error'].value,
                "recon_light_profile": recon_data['/Reconstructed light profile'].value,
                "recon_ne_profile": recon_data['/Reconstructed density profile'].value,
                "recon_ne_low_error_profile": recon_data['/Reconstructed density profile low error'].value,
                "recon_ne_high_error_profile": recon_data['/Reconstructed density profile high error'].value,
                "recon_ne_error_pos_profile": recon_data['/Reconstructed density profile error position'].value
                 }
        shot_data["recon_ne_error_pos_profile_r"] = recon_data['/Reconstructed density profile error position r'].value
        plt.close('all')
        fig = plt.figure(figsize=(12,10))
        ax = plt.axes()
        ax.plot(shot_data['recon_r_coord'],shot_data['recon_ne_profile'].flatten(),color='b')
        ax.plot(shot_data['recon_ne_error_pos_profile_r'].flatten(),shot_data['recon_ne_low_error_profile'].flatten(),linestyle='--',linewidth=2,color='b')
        ax.plot(shot_data['recon_ne_error_pos_profile_r'].flatten(),shot_data['recon_ne_high_error_profile'].flatten(),linestyle='--',linewidth=2,color='b')
        ax.set_xlim(6.15,6.28)
        ax.set_ylim(0,3.5)
        ax.set_title('20181018.003 (4-4.8s)')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('$n_e/10^{19}[m^{-3}]$')
        ax.plot([6.21,6.21],plt.ylim(),linestyle='dashed',color='k')
        ax.plot([6.26,6.26],plt.ylim(),linestyle='dashed',color='k')
        ax.plot([6.235,6.235],plt.ylim(),linestyle='dotted',color='k')
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
        fig.savefig('eps2019_sz_10.png',dpi=300)
  
#eps2019(fig_no =7)
