# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:38:28 2022

@author: ShendR
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
from pathlib import Path
import traceback, sys
from tkinter import filedialog
from tkinter import *
import difflib

from flap_gui_7 import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QVBoxLayout, QDialog, QMessageBox, QSystemTrayIcon


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


import flap
import flap.testdata
flap.testdata.register()

import flap_apdcam
import flap_w7x_abes

flap_apdcam.register()
flap_w7x_abes.register()

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_tests.cfg")
# fn = os.path.join(thisdir,"flap_test_apdcam.cfg")
flap.config.read(file_name=fn)


class FlapGui(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.flap_variable = {}
               
                
        
# ----------------/// Buttons and actions \\\------------------#

# *** Create Data Object *** #

        self.get_data_comboBox.currentIndexChanged.connect(lambda: 
                                                           self.source_combo_box_change()
                                                           )


        self.create_obj_button.clicked.connect(lambda: 
                                               self.create_data_obj(
                                                                    self.get_data_comboBox.currentText(),
                                                                    self.data_name_text.text(),
                                                                    self.obj_name_text.text())
                                               )

        self.add_coor_button.clicked.connect(lambda: 
                                             self.add_coordinates(
                                                                self.obj_list.currentItem().text()
                                                                )
                                             )
            
            

# *** Active Data Object *** #

        self.save_obj_button.clicked.connect(lambda: 
                                             self.save_data_obj(
                                                                self.obj_list.currentItem().text()
                                                                )
                                             )
        
        self.load_obj_button.clicked.connect(lambda: 
                                             self.load_data_obj()
                                             )
        
        self.delete_obj_button.clicked.connect(lambda: 
                                               self.del_data_obj(
                                                   self.obj_list.currentItem().text()
                                                   )
                                               )

        self.clear_obj_button.clicked.connect(lambda: 
                                              self.obj_list.clear()
                                              )
            
            
        self.obj_list.itemClicked.connect(lambda: 
                                          self.active_obj(self.obj_list.currentItem().text()
                                                          )
                                          )


        self.plot_obj_button.clicked.connect(lambda: 
                                             self.plot_data_obj(
                                                                self.obj_list.currentItem().text()
                                                                )
                                             )


# *** Advanced plot *** #
        

        self.plot_type_combo.currentIndexChanged.connect(lambda: 
                                                         self.plot_combo_change()
                                                         )
            
        
        self.close_all_plot_button.clicked.connect(lambda: 
                                                   self.close_all_plots()
                                                   )
        
        self.plot_spec_button.clicked.connect(lambda: 
                                              self.plot_advanced(
                                                                  self.obj_list.currentItem().text()
                                                                  )
                                              )
            
        self.slice_data_button.clicked.connect(lambda:
                                               self.slice_data(
                                                               self.obj_list.currentItem().text()
                                                               )
                                               )
        
        self.flap_func_button.clicked.connect(lambda: self.flap_functions(
                                                                self.flap_func_variable.text()))
        
        self.flap_func_button.clicked.connect(lambda: print(self.flap_variable))
            
            
            
        self.flap_apsd_button.clicked.connect(lambda:
                                               self.flap_apsd(
                                                               self.obj_list.currentItem().text()
                                                               )
                                               )
 
        self.flap_cpsd_button.clicked.connect(lambda:
                                               self.flap_cpsd(
                                                               self.obj_list.currentItem().text()
                                                               )
                                               )
        
            
            
    def logbook(self, text='', text_color='black', font="MS Shell Dlg 2", size = '9pt'): 
        set_text = "<span style=\" font-family:text_font; font-size:text_size; color : text_color \" >" + text + "</span>"
        write = set_text.replace('text_font', font)
        write = write.replace('text_color', text_color)
        write = write.replace('text_size', size)                
        self.Big_textBrowser.moveCursor(QtGui.QTextCursor.End)
        self.Big_textBrowser.ensureCursorVisible()
        self.Big_textBrowser.setAlignment(QtCore.Qt.AlignLeft)
        self.Big_textBrowser.append(write)
        self.Big_textBrowser.append('')

    
    def check_in(self, in_put):
        if in_put in self.flap_variable.keys():
            best_out = in_put
        else:       
            if True in [char.isdigit() for char in in_put]:   
                try:
                    best_out = eval(in_put)
                except:
                    print('Cannot eval input')
                    best_out = in_put
            else:    
                my_file = open("options_list.txt", "r")  
                data = my_file.read()
                opt_list = data.replace('\n', ',').split(",")
                
                best = difflib.get_close_matches(in_put, opt_list)
                if best == []:
                    best_out = in_put
                else:        
                    best_out = best[0]
                    
                    if best_out == 'True':
                        best_out = True
                        
                    elif best_out == 'False':
                        best_out = False
                        
                    elif (best_out == 'None' or best_out==''):
                        best_out = None
                        
                my_file.close()
            
        return best_out


    def check_if_list(self, in_put):
        if ('[' in in_put and ']' in in_put):
            a = in_put[in_put.find('[')-1 + 1:in_put.find(']')+1]
            b = eval(a)
            in_put = in_put.replace(in_put[in_put.find('[')-1 + 1:in_put.find(']')+1], 'list')
        
        if ',' in in_put:
            if ' ' in in_put:
                in_put = in_put.replace(' ','')
            in_put = in_put.split(',')
            out_put = []
            for i in in_put:
                out_put.append(self.check_in(i))
            if 'list' in out_put:
                out_put= [b if x=='list' else x for x in out_put]
            # if 'Error' in out_put:
            #     out_put.remove('Error')
            #     print('Invalid input element was removed from list')
        elif in_put == 'list':
            out_put = b
        else:
            out_put = self.check_in(in_put)
            
        return out_put
        

    def make_dict(self, key, value):
        try:
            keys = self.check_if_list(key)
            values = self.check_if_list(value)
            if (type(keys) is list and type(values) is list):
                if len(keys) != len(values):
                    print('Numer of keys does not mach with number of values!')
                else:
                    dict_in = {keys[i]: values[i] for i in range(len(keys))}
            else:
                dict_in = {keys : values}
        except:
            print('Could not make Dictionary!')
            
        return dict_in

 
    
 # -------------------/// Create Data Object \\\-------------------#
 
 # *** Data source *** #
 
 
    def source_combo_box_change(self):
        print('Data source changed to: ' + self.get_data_comboBox.currentText())
        self.logbook(text='Data source changed to: ' + self.get_data_comboBox.currentText())
        self.update_source()
        
    
    def get_default_options(self, source):
        if source == 'APDCAM':
            default_options = {'Datapath':'',
                                'Scaling':'Digit'
                                }
            
        if source == 'TESTDATA':
            default_options = {'Row number': 10,
                                'Column number': 15,
                                'Matrix angle': 0.0,
                                'Scaling': 'Volt',
                                'Signal': 'Sin',
                                'Image':'Gauss',
                                'Spotsize':100,
                                'Width': 640,
                                'Height': 480,
                                'Frequency': 1e3,
                                'Length': 0.1,
                                'Samplerate':1e6
                                }
            
        if source == 'W7X_ABES':
            default_options = {'Datapath': '',
                                'Scaling':'Digit'
                                # 'State': '',
                                # 'Start' : 1000,
                                # 'End': -1000,
                                # 'State_key': 'Chop, Defl',
                                # 'State_val': '0, 0'
                                }    
        
        return default_options
            
        
 
    def update_source(self):
        default_options = self.get_default_options(self.get_data_comboBox.currentText())
        if (self.get_data_comboBox.currentText() == 'APDCAM' or
            self.get_data_comboBox.currentText() == 'W7X_ABES' ):
            fn = os.path.join(thisdir,"flap_test_apdcam.cfg")
            flap.config.read(file_name=fn)
            
            self.tableWidget.setRowCount(len(default_options))
            _translate = QtCore.QCoreApplication.translate
            self.tableWidget.setSortingEnabled(False)
            for i in range((len(default_options))):    
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setVerticalHeaderItem(i, item)    
                                        
                item = self.tableWidget.verticalHeaderItem(i)
                item.setText(_translate("MainWindow", str(list(default_options.keys())[i])))
                self.tableWidget.setSortingEnabled(False)
                
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i, 0, item)   
                item = self.tableWidget.item(i, 0)
                item.setText(_translate("MainWindow", str(list(default_options.values())[i])))
                
                
            
        if self.get_data_comboBox.currentText() == 'TESTDATA':
            fn = os.path.join(thisdir,"flap_tests.cfg")
            flap.config.read(file_name=fn)
            
            self.tableWidget.setRowCount(len(default_options))
            _translate = QtCore.QCoreApplication.translate
            self.tableWidget.setSortingEnabled(False)
            for i in range((len(default_options))):    
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setVerticalHeaderItem(i, item)    
                                        
                item = self.tableWidget.verticalHeaderItem(i)
                item.setText(_translate("MainWindow", str(list(default_options.keys())[i])))
                self.tableWidget.setSortingEnabled(False)
                
                item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(i, 0, item)   
                item = self.tableWidget.item(i, 0)
                item.setText(_translate("MainWindow", str(list(default_options.values())[i])))
            
        return default_options
    
 
    
 
    def create_data_obj(self,data_source, name, obj_name):
        try:
        
            default_options = self.get_default_options(self.get_data_comboBox.currentText())
            
            exp_id = self.exp_id_text.text()
            
            if exp_id == '':
                exp_id = None
            else:
                exp_id = exp_id
           
            if self.option_checkBox.isChecked():        
                options = {}
                for i in range(len(default_options)):
                    
                    item = self.tableWidget.item(i, 0)
                    # print(item.text())
                    if (type(list(default_options.values())[i]) == int or 
                        type(list(default_options.values())[i]) ==float):
                        if item.text() == '':
                            item_val = item.text()
                        else:
                            item_val = eval(item.text())
                    else:
                        item_val = item.text()
                    
                    options[list(default_options.keys())[i]]=item_val
                
                if 'State_key' in options.items():                   
                    for k,v  in list(options.items()):
                        if (v == None or v==''):
                          options.pop(k)
                        if k == 'State_key':
                            v_key = v
                            options.pop(k)
                        if k == 'State_val':
                            v_val = v
                            options.pop(k)
                    options['State'] = self.make_dict(v_key, v_val)
                
                print(options) 
                      
                coor_key = self.option_key_line.text()
                coor_val = self.option_value_line.text()
                # coor_val_eval=eval(coor_val)
                # print(f'coor_val_eval type: {type(coor_val_eval)}')
                if (coor_key=='' or coor_val==''):
                    coordinates = None
                else:
                    # coordinates = {'Time':[3,4]}
                    coordinates = self.make_dict(coor_key, coor_val)
                    print(f'coordinates: {coordinates}')
                    print(f'coordinates type is: {type(coordinates)}')
                    print(type(coordinates['Time']))
                    self.logbook(f'coordinates: {coordinates}')
                    
                d=flap.get_data(data_source = data_source, 
                                exp_id = exp_id,
                                name = name,
                                options = options, 
                                coordinates = coordinates,
                                object_name = obj_name)
                self.obj_list.addItem(obj_name)
                # d = flap.get_data('APDCAM', name='ADC39', object_name='ADC39', coordinates={'Time':[4, 6]})
            else:
                d=flap.get_data(data_source, 
                                exp_id=exp_id, 
                                name=name, 
                                object_name=obj_name)
                self.obj_list.addItem(obj_name)       
                
            self.Big_textBrowser.setAlignment(QtCore.Qt.AlignLeft)  
            self.Big_textBrowser.append(str(flap.list_data_objects(d)))
            self.Big_textBrowser.append('\n')
            
        except Exception as e:
            # self.Big_textBrowser.append("<span style=\"color:#ff0000\" >"+'Exeption error message: '+ "</span>" + 
            #                             "<span style=\"color:#800080\" >" + '-- ' + str(e) + ' --' +"</span>" )
            self.logbook('Exeption error message: ', text_color='red', size='10pt')
            self.logbook('--- ' + str(e) + ' ---', text_color='purple')

        
       
        
       
 
 # *** Mudule options and coordinates *** #
 
    def add_coordinates(self, obj_name):
        coor = self.add_coor_line.text()
        if coor == '':
            coor = None
        else:
            coor = self.check_if_list(coor)
        
        flap.add_coordinate(obj_name, coordinates=coor)
        print(f'Coordinates {coor} added to {obj_name}')
        self.logbook(f'Coordinates {coor} added to {obj_name}')
        
        self.active_obj(obj_name)
 
 
 
 
 
 # -------------------/// Active Data Object and basic plot \\\----#
 
 # *** DataOjb. handling *** #
 
 
    def save_data_obj(self, obj_name):
        flap.save(obj_name, filename=obj_name+'.dat')
        
        
    def load_data_obj(self):
        # Pop-up dialog    
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        root.filename = filedialog.askopenfilename(initialdir='C:\ShendR\Python\Flap GUI' , title='Select a file', filetypes=(('dat files', '*.dat'),('All files','*.*')))
        fn = root.filename
        root.destroy()
        root.mainloop()
        head, file_name = os.path.split(fn)
        d=flap.load(file_name)
        self.obj_list.addItem(file_name.replace('.dat',''))
        self.Big_textBrowser.append(str(flap.list_data_objects(d)))
        self.Big_textBrowser.append('\n')

        
        
    def del_data_obj(self, obj_name):
        flap.flap_storage.delete_data_object(obj_name)
        self.obj_list.takeItem(self.obj_list.currentRow())
        self.logbook('Element deleted from list and Flap Storage')
        
        
        
    def clear_list(self):
        self.obj_list.clear()
        flap.delete_data_object('*')
        print(flap.list_data_objects())
        self.Big_textBrowser.append(flap.list_data_objects())
        self.Big_textBrowser.append('\n')
        
        
 
 # *** Active DataObj. *** #
 
 
    def active_obj(self, obj_name):
        self.active_obj_line.setText(obj_name)
        d = flap.get_data_object(obj_name)
        self.psd_on_obj_line.setText(obj_name)
        coordinates = d.coordinate_names()
        units = []
        for i in range(len(d.coordinates)):
            e=d.coordinates[i].unit 
            units.append(e.unit)

        # Fill Coordinate and units list
        self.coor_name_list.clear()
        self.coor_unit_list.clear()
        self.coor_name_list.addItems(coordinates) 
        self.coor_unit_list.addItems(units)
        # Plot comboBoxes
        self.data1_combo.clear()
        self.data1_combo.addItem('None')
        self.data1_combo.addItem('Data')
        self.data1_combo.addItems(coordinates)
        self.data2_combo.clear()
        self.data2_combo.addItem('None')
        self.data2_combo.addItem('Data')
        self.data2_combo.addItems(coordinates)
        self.data3_combo.clear()
        self.data3_combo.addItem('None')
        self.data3_combo.addItem('Data')
        self.data3_combo.addItems(coordinates)
        self.data3_combo.setDisabled(True)
        self.data4_combo.clear()
        self.data4_combo.addItem('None')
        self.data4_combo.addItem('Data')
        self.data4_combo.addItems(coordinates)
        self.data4_combo.setDisabled(True)

 
 
 
    def plot_data_obj(self, obj_name): 
        try:
            plt.figure()
            d=flap.get_data_object(object_name=obj_name)
            # plt.close('all')
            d.plot()
            self.logbook('Plot finished!', text_color='green')
        except Exception as e:
            self.logbook('Exeption error message: ', text_color='red', size='11pt')
            self.logbook('--- ' + str(e) + ' ---', text_color='orange')
            
 
    
 
 
 # -------------------/// Advanced plot \\\------------------------#
 
 # *** Plot type *** #
 
    def plot_combo_change(self):
        if self.plot_type_combo.currentText() == 'xy':
            self.active_obj(self.obj_list.currentItem().text())
            
        if self.plot_type_combo.currentText() == 'multi':
            self.active_obj(self.obj_list.currentItem().text())
            self.data3_combo.setEnabled(True)
            
        if self.plot_type_combo.currentText() == 'image':
            self.active_obj(self.obj_list.currentItem().text())
            self.data3_combo.setEnabled(True)
            
        if self.plot_type_combo.currentText() == 'grid xy':
            self.active_obj(self.obj_list.currentItem().text())
            self.data3_combo.setEnabled(True)
            self.data4_combo.setEnabled(True)
 
 
 
 # *** Plot options *** #
 
    def close_all_plots(self):
        plt.close('all')
        self.logbook('All plots are closed!')

        
 
 # *** Slicing options *** #
 
    def slice_data(self, obj_name):
        if self.slicing_opt_checkBox.isChecked():
            sliced_name = self.sliced_name_line.text()
            
            slice_key = self.slicing_opt_key.text()
            slice_val = self.slicing_opt_val.text()

            if (slice_key=='' or slice_val==''):
                slicing = None
            else:
                slicing = self.make_dict(slice_key,slice_val)
                print(f'slicing: {slicing}')
                self.logbook(f'slicing: {slicing}')
                
                
            sliced_opt_key = self.slicedata_opt_key.text()
            sliced_opt_val = self.slicedata_opt_val.text()
            
            if (sliced_opt_keyy=='' or sliced_opt_val==''):
                sliced_opt = None
            else:
                sliced_opt = self.make_dict(sliced_opt_key, sliced_opt_val)
                print(f'sliced_opt: {sliced_opt}')
                self.logbook(f'sliced_opt: {sliced_opt}')
            
        else:
            slicing = None
            sliced_opt = None
            
        flap.slice_data(obj_name, 
                        output_name=sliced_name, 
                        slicing=slicing, 
                        options=sliced_opt)
        
        self.obj_list.addItem(sliced_name)
        
        
 
 # *** Advanced Plot *** #
 
    def plot_advanced(self, obj_name):
        try:            
            self.active_obj_line.setText(obj_name)
            d = flap.get_data_object(obj_name)
            
            if self.plot_type_combo.currentText() == 'None':
                plot_type = None
            else:
                plot_type =  self.plot_type_combo.currentText()
    
            
            if self.slicing_opt_checkBox.isChecked():
                slice_key = self.slicing_opt_key.text()
                slice_val = self.slicing_opt_val.text()
                if (slice_key=='' or slice_val==''):
                    slicing = None
                else:
                    slicing = self.make_dict(slice_key,slice_val)
                    print(f'slicing: {slicing}')
                    self.logbook(f'slicing: {slicing}')
                    
                sliced_opt_key = self.slicedata_opt_key.text()
                sliced_opt_val = self.slicedata_opt_val.text()
                
                if (sliced_opt_key=='' or sliced_opt_val==''):
                    sliced_opt = None
                else:
                    sliced_opt = self.make_dict(sliced_opt_key, sliced_opt_val)
                    print(f'slicing: {slicing}')
                    self.logbook(f'slicing: {slicing}')
                
            else:
                slicing = None
                sliced_opt = None
                
                
            if self.plot_opt_checkBox.isChecked():
                axes = self.axes_line.text()
                axes = self.check_if_list(axes)
                print(f'Plot options:  {axes}')
                self.logbook(f'axes: {axes}')

                
                p_options_key = self.p_options_key.text()
                p_options_val = self.p_options_val.text()

                if (p_options_key=='' or p_options_val==''):
                    options = None
                else:
                    options = self.make_dict(p_options_key,p_options_val)
                    print(f'options: {options}')
                    self.logbook(f'options: {options}')

                    
                plot_opt_key = self.plot_opt_key.text()
                plot_opt_val = self.plot_opt_val.text()

                if (plot_opt_key=='' or plot_opt_val==''):
                    plot_options = None
                else:
                    plot_options = self.make_dict(plot_opt_key, plot_opt_val)
                    print(f'plot_options: {plot_options}')
                    self.logbook(f'plot_options: {plot_options}')
                
            else:
                axes = None
                options = None
                plot_options = None
                
            data1 = self.data1_combo.currentText()
            data2 = self.data2_combo.currentText()
            if data1=='None':
                data1=None
            if data1=='Data':
                data1 = '__Data__'
            if data2=='None':
                data2=None
            if data2=='Data':
                data2 = '__Data__'
            if (not data1 or not data2):
                
                plt.figure()
                flap.plot(obj_name, 
                          plot_type=plot_type, 
                          slicing=slicing, 
                          axes=axes, 
                          options=options,
                          plot_options=plot_options)
                
            else:
                plt.figure()
                flap.plot(obj_name, 
                          slicing=slicing, 
                          axes=[data1,data2],
                          plot_options=plot_options)
                
            self.logbook('Plot finished!', text_color='green')
                
        except Exception as e:
            # self.Big_textBrowser.append("<span style=\"color:#ff0000\" >"+'Exeption error message: '+ "</span>" + 
            #                             "<span style=\"color:#800080\" >" + '-- ' + str(e) + ' --' +"</span>" )
            self.logbook('Exeption error message: ', text_color='red', size='10pt')
            self.logbook('--- ' + str(e) + ' ---', text_color='orange')
 
    
# *****  FLAP functions ***** #

    def flap_functions(self, var_name):
        
        combo_item = self.flap_func_combo.currentText()
        print(combo_item)
        if combo_item == 'flap.Intervals':
            args = self.arg_name_line.text()
            print(args)
            value = self.arg_val_line.text()
            print(value)
            kwargs = self.make_dict(args, value)
            print(kwargs)
            var_object = flap.Intervals(**kwargs)
            print(var_object)
            print(var_name)
        # # var_name = self.flap_func_variable.text()
        # if var_name == '':
        #     return None
        
        # else:
        #     self.obj_list.addItem(var_name + '   <--- variable')
            
            
        #     # if combo_item == 'flap.Intervals':
        #     #     self.arg_name_line.setText('start, stop, step, number')
        #     args = self.arg_name_line.text()
        #     # args = self.check_if_list(args)
        #     value = self.arg_val_line.text()
        #     # value = self.check_if_list(value)
        #     # print(args)
        #     # print(value)
        #     kwargs = self.make_dict(args, value)
        #     print(kwargs)
        #     fun = flap.Intervals(**kwargs)
        #     print(fun)
            
        #     # if (type(args)==list and type(value)==list):
                
        #     # if (args==[] or value==[]):
        #     #     print('No arguments or values given!')
        #     # elif (len(args) != len(value)):
        #     #     print('Number of argumnets does not maches numbers of values!')
        #     # else:
        #     #     print(args, value)
        
            
        return self.flap_variable.update({var_name : var_object})
        print(self.flap_variable)




# *****  FLAP Power Spectrum  ***** #

#------- APSD ---------#

    def flap_apsd(self, obj_name):
        try:            
            self.active_obj_line.setText(obj_name)
            d = flap.get_data_object(obj_name)
            # self.apsd_on_obj_line.setText(obj_name)
            
            apsd_name = self.apsd_name_line.text()
            if apsd_name == '':
                print('No apsd name is given!')
                self.logbook('No apsd name is given!', text_color='red', size='10pt')
            else:
                apsd_name = apsd_name
                
            apsd_coor = self.apsd_coor_line.text()
            if apsd_coor == '':
                apsd_coor = None
            else:
                apsd_coor = self.check_in(apsd_coor)
                
            apsd_inter = self.apsd_inter_line.text()
            if apsd_inter == '':
                apsd_inter = None
            else:
                apsd_inter = self.check_in(apsd_inter)                
            
            
            apsd_opt_key = self.apsd_opt_key.text()
            apsd_opt_val = self.apsd_opt_val.text()
            if (apsd_opt_key=='' or apsd_opt_val==''):
                apsd_options = None
            else:
                apsd_options = self.make_dict(apsd_opt_key, apsd_opt_val)
                print(f'apsd_options: {apsd_options}')
                self.logbook(f'apsd_options: {apsd_options}')
                
            flap.apsd(object_name=obj_name, 
                      output_name=apsd_name,
                      coordinate=apsd_coor,
                      intervals=apsd_inter,
                      options=apsd_options)
            self.obj_list.addItem(apsd_name)
            
            
        except Exception as e:
            # self.Big_textBrowser.append("<span style=\"color:#ff0000\" >"+'Exeption error message: '+ "</span>" + 
            #                             "<span style=\"color:#800080\" >" + '-- ' + str(e) + ' --' +"</span>" )
            self.logbook('Exeption error message: ', text_color='red', size='11pt')
            self.logbook('--- ' + str(e) + ' ---', text_color='orange')
            
            
#------- APSD ---------#

    def flap_cpsd(self, obj_name):
        try:            
            self.active_obj_line.setText(obj_name)
            d = flap.get_data_object(obj_name)
            # self.apsd_on_obj_line.setText(obj_name)
            
            cpsd_name = self.cpsd_name_line.text()
            if cpsd_name == '':
                print('No cpsd name is given!')
                self.logbook('No cpsd name is given!', text_color='red', size='10pt')
            else:
                cpsd_name = cpsd_name
                
            cpsd_coor = self.cpsd_coor_line.text()
            if cpsd_coor == '':
                cpsd_coor = None
            else:
                cpsd_coor = self.check_in(cpsd_coor)
            
            cpsd_inter = self.cpsd_inter_line.text()
            if cpsd_inter == '':
                cpsd_inter = None
            else:
                cpsd_inter = self.check_in(cpsd_inter)
                
            
            cpsd_opt_key = self.cpsd_opt_key.text()
            cpsd_opt_val = self.cpsd_opt_val.text()
            if (cpsd_opt_key=='' or cpsd_opt_val==''):
                cpsd_options = None
            else:
                cpsd_options = self.make_dict(cpsd_opt_key, cpsd_opt_val)
                print(f'cpsd_options: {cpsd_options}')
                self.logbook(f'cpsd_options: {cpsd_options}')
                
            flap.cpsd(object_name=obj_name, 
                      output_name=cpsd_name,
                      coordinate=cpsd_coor,
                      intervals=cpsd_inter,
                      options=cpsd_options)
            self.obj_list.addItem(cpsd_name)
            
            
        except Exception as e:
            # self.Big_textBrowser.append("<span style=\"color:#ff0000\" >"+'Exeption error message: '+ "</span>" + 
            #                             "<span style=\"color:#800080\" >" + '-- ' + str(e) + ' --' +"</span>" )
            self.logbook('Exeption error message: ', text_color='red', size='11pt')
            self.logbook('--- ' + str(e) + ' ---', text_color='orange')



def GUI():
    app = QtWidgets.QApplication([])
    # app_icon = QtGui.QIcon()
    # app_icon.addPixmap(QtGui.QPixmap('pstc-256x256.png'), QtGui.QIcon.Selected, QtGui.QIcon.On)
    # app.setWindowIcon(app_icon)
#    app.setWindowTitle("SPI sensor data")
    # trayIcon = QSystemTrayIcon(QtGui.QIcon('pstc-256x256.png'), parent=app)
    widget = FlapGui()
    # trayIcon.show()
    widget.showMaximized()   
    app.exec_()    
         
if __name__ == '__main__':
    GUI()


