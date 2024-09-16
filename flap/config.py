# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 21:45:43 2019

Configuration handling for FLAP

@author: Sandor Zoletnik  (zoletnik.sandor@ek-cer.hu)
Centre for Energy Research
"""
import configparser
import copy
import flap.tools

class Config:
    """Class containing configuration for FLAP.

    Attributes
    ----------
    config: flap.ConfigParser
        The configuration itself.
    file_name: str
        The configuration filename.
    """
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.file_name = None

def read(file_name='flap_defaults.cfg'):
    """Read configuration file.

    Parameters
    ----------
    file_name : str, optional
        File name of configuration file, by default 'flap_defaults.cfg'
    """
    __flap_config.file_name = file_name

    read_ok = __flap_config.config.read(__flap_config.file_name)
    if (read_ok == []):
        raise OSError("Error reading configuration file "+__flap_config.file_name)

def get(section, element, datatype=str, default=None, evaluate=False):
    """Get the value of an element of a specified section
    from the configuration.

    Parameters
    ----------
    section : str
        Section of the configuration to get.
    element : str
        Element (of section specified) of the configuration to get.
    datatype : optional, default=str
        Type to cast configuration value of the element into.  Only applied if
        `evaluate` is False.
    default : optional, default=None
        Default value to use if `section` or `element` does not exist in the
        configuration.
    evaluate : bool, optional, default=False
        Whether to evaluate the value obtained instead of type casting it.

    Returns
    -------
    object
        The value of the element `element` of section `section` 
        in the configuration, type-cast or evaluated.
    """
    if (__flap_config.file_name is None):
        raise ValueError("Configuration has not been read.")
    try:
        txt = __flap_config.config[section][element]
    except KeyError:
        if (default is None):
            raise ValueError("No '" + element
                             + "' entry in section [" + section+"] in config file:"
                             + __flap_config.file_name)
        return default

    try:
        if evaluate is False:
            return(datatype(txt))
        else:
            return(eval(txt))
    except (ValueError, TypeError):
        raise ValueError("Invalid value (" + txt + ") for " + element
                             + " entry in section [" + section+"] in config file:"
                             + __flap_config.file_name)

def get_all_section(section):
    """Get all configuration elements from a section of the configuration.

    Parameters
    ----------
    section : str

    Returns
    -------
    dict
        Dictionary containing the configuration elements and associated values.
    """
    if (__flap_config.file_name is None):
        raise ValueError("Configuration has not been read.")
    try:
        l = __flap_config.config.items(section)
        d = {}
        if (l is not []):
            for e in l:
                val = interpret_config_value(e[1])
                d.update({e[0] : val})

        return d
    except configparser.NoSectionError:
        return {}

def interpret_config_value(value_str):
    """Determine the data type from the input string and convert.

    Parameters
    ----------
    value_str : str
        The value string to convert.

        Conversions:
        - 'True', 'Yes' converts to ``True``
        - 'False', 'No' converts to ``False``
        - Starting and ending with ``'`` or ``"`` converts to str
        - If can be converted to int, float or complex, it is converted to the
        corresponding converted numeric value
        - Starting and ending with ``[]`` converts to list
        - If all the above fails, keep as str

    Returns
    -------
    bool | str | int | float | complex | list
        Converted value.
    """
    if ((value_str == 'True') or (value_str == 'Yes')):
        return True
    if ((value_str == 'False') or (value_str == 'No')):
        return False
    if ((value_str[0] == "'") and (value_str[-1] == "'")
        or (value_str[0] == '"') and (value_str[-1] == '"')):
        return value_str[1:-1]
    convert_types = [int, float, complex]
    if ((value_str[0] == '[') and (value_str[-1] == ']')):
        values = value_str[1:-1].split(',')
        values_list = []
        for val in values:
            if ((val == 'True') or (val == 'Yes')):
                values_list.append(True)
                continue
            if ((val == 'False') or (val == 'No')):
                values_list.append(True)
                continue
            if ((val[0] == "'") and (val[-1] == "'")
                or (val[0] == '"') and (val[-1] == '"')):
                values_list.append(val[2:-1])
                continue
            for t in convert_types:
                try:
                    value = t(val)
                    values_list.append(value)
                    break
                except ValueError:
                    pass
            else:
                values_list.append(val)
        return values_list
    for t in convert_types:
        try:
            value = t(value_str)
            return value
        except ValueError:
            pass
    return value_str

def merge_options(default_options, input_options, data_source=None, section=None):
    """Merge options dictionaries.

    Uses default options of function, input options of function and options read
    from config file from section `section`. If `exp_id` is set, will also look
    for options in section Module `exp_id` for options starting with `section`.
    
    The precedence of options is: default_options < section options < module
    options < input_options

    Parameters
    ----------
    default_options : dict
        Default options in a function. This should contain all the possible
        options.
    input_options : dict
        Contents of options argument of function. Option keys can also be
        abbreviated.
    data_source : dict, optional, default=None
        The data source of the measurement. (May be None.)
    section : dict, optional, default=None
        Name of the section in the config file related to the function. (May be
        None.)

    Returns
    -------
    dict
        The merged options dictionary. Abbreviated keys are expanded to full
        name.
    """
    if (default_options is None):
        return {}
    if (section is not None):
        section_options = get_all_section(section)
    else:
        section_options = {}
    if (data_source is not None):
        module_options = get_all_section("Module "+data_source)
    else:
        module_options = {}
    options =  copy.deepcopy(default_options)

    # MOdule separator keys
    module_sep = '{}'

    if (section is not None):
        options.update(section_options)
    if ((section is not None) and (data_source is not None)):
        # Looking for options in the data source which refer to this section,
        # that is start with {section}
        module_selected_options = {}
        for module_key in module_options.keys():
            if ((module_key[0] == module_sep[0]) and (module_key.find(module_sep[1]) != 0)):
                i = module_key.find(module_sep[1])
                if ((module_key[1:i] == section) and (len(module_key) > i+1)):
                    option_name = module_key[i+1:]
                    module_selected_options[option_name] = module_options[module_key]
        options.update(module_selected_options)
    if ((section is None) and (data_source is not None)):
        # Looking for module options which do not have {...} at the beginning
        module_selected_options = {}
        for module_key in module_options.keys():
            if ((module_key[0] == module_sep[0]) and (module_key.find(module_sep[1]) != 0)):
                continue
            module_selected_options[module_key] = module_options[module_key]
        options.update(module_selected_options)

    # Processing input options
    if (input_options is not None):
        if (type(input_options) is not dict):
            raise TypeError("Options must be a dictionary or None.")
        default_keys = list(default_options.keys())
        for input_key in input_options.keys():
            n_match = 0
            for i in range(len(default_keys)):
                if ((len(default_keys[i]) >= len(input_key))  \
                    and (default_keys[i][0:len(input_key)] == input_key)):
                    n_match += 1
                    if (n_match > 1):
                        break
                    i_match = i
                    match_input_key = input_key
            if (n_match > 1):
                raise ValueError("Input option key '"+input_key+"' matches multiple possible options.")
            if (n_match == 0):
                raise ValueError("Input option key '"+input_key+"' does not match any possible option.")
            options[default_keys[i_match]] = input_options[match_input_key]
    return options

def test_select_signals():
    """Test signal selection.
    """
#    signals = chlist(chrange=[1,20],prefix='ABC-',postfix='-SD')
    signals = ['ABC-3-SD','ABC-3-SD-3','ABC-4-SD-5']
    try:
        chl, ilist = flap.tools.select_signals(signals,'ABC-[1-22]-SD-[4-7]')
    except Exception as e:
        print(e)
    print(chl)



try:
    __flap_config
except NameError:
    __flap_config = Config()
    try:
        read()
    except OSError:
        print("Warning: could not read configuration file '"+__flap_config.file_name+"'.")
        print("Default location of configuration file is working directory.")
