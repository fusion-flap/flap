global VERBOSE
VERBOSE = True
from .tools import select_signals,SignalList,interpret_signals,time_unit_translation,spatial_unit_translation,unit_conversion
from .coordinate import *
from .data_object import *
from .plot import *
from .flap_xml import *
from .spectral_analysis import *
import flap.config
from .select import *
from .time_frequency_analysis import *