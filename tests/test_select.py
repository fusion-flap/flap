# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:52:43 2019

@author: Zoletnik
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

plt.close('all')
flap.get_data('TESTDATA',name='TEST-1-1',options={'Length':0.02},object_name='TEST')
d_sel = flap.select_intervals('TEST',coordinate='Time')
d_sel.plot(axes=['__Data__',0],plot_type='scatter',options={'Force':True})

