# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:38:14 2021

@author: ShendR
"""

import os
import matplotlib.pyplot as plt

import flap
import flap_apdcam
import flap_w7x_abes

flap_apdcam.register()
flap_w7x_abes.register()

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_test_apdcam.cfg")
flap.config.read(file_name=fn)

data = flap.get_data('W7X_ABES', exp_id='20181018.027', name='ABES-10', object_name='ADC10')
# data = flap.get_data('APDCAM', exp_id='20181018.027', name='ADC39', object_name='ADC39')
# data = flap.get_data('APDCAM', exp_id='20181018.027', name='ADC1*')


# flap.save('ADC39', filename='adc39_datobj.dat')
# flap.load('adc39_datobj.dat')


array = data.data

# data.plot()

coordinates = data.coordinate_names()

time = data.coordinate("Time")
time_data = data.coordinates[1]    # is equal to time variable
sample_data = data.coordinates[0]  # is equal to array variable

time_data_name = time_data.unit.name
time_data_unit = time_data.unit.unit
sample_data_name = sample_data.unit.name
sample_data_unit = sample_data.unit.unit

unit = data.data_unit
unit_name = unit.name
unit_unit = unit.unit

