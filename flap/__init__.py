global VERBOSE
VERBOSE = True
if (VERBOSE):
    print("Importing flap")
from .tools import *
from .coordinate import *
from .data_object import *
from .plot import *
from .flap_xml import *
from .spectral_analysis import *
import flap.config
from .select import *
