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

from flap_gui_4 import Ui_MainWindow
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

flap_apdcam.register()

thisdir = os.path.dirname(os.path.realpath(__file__))
fn = os.path.join(thisdir,"flap_tests.cfg")
# fn = os.path.join(thisdir,"flap_test_apdcam.cfg")
flap.config.read(file_name=fn)


class FlapGui(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
                
# ----------------/// Buttons and actions \\\------------------
# *** Create Data Object *** #

# *** Active Data Object *** #

# *** Advanced plot *** #

        
        self.create_obj_button.clicked.connect(lambda: 
                                               self.create_data_obj(
                                                                    self.get_data_comboBox.currentText(),
                                                                    self.data_name_text.toPlainText(),
                                                                    self.obj_name_text.toPlainText())
                                               )
        
        self.plot_obj_button.clicked.connect(lambda: 
                                             self.plot_data_obj(
                                                                self.obj_list.currentItem().text()
                                                                )
                                             )
             
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
        
        self.close_all_plot_button.clicked.connect(lambda: 
                                                   self.close_all_plots()
                                                   )
        
        self.clear_obj_button.clicked.connect(lambda: 
                                              self.obj_list.clear()
                                              )
 
        self.get_data_comboBox.currentIndexChanged.connect(lambda: 
                                                           self.combo_box_change()
                                                           )
        
        self.obj_list.itemClicked.connect(lambda: self.active_obj(self.obj_list.currentItem().text()))
        
        self.plot_type_combo.currentIndexChanged.connect(lambda: self.plot_combo_change())
        
        self.plot_spec_button.clicked.connect(lambda: self.plot_advanced(self.obj_list.currentItem().text()))
 
    
 
    
 # -------------------/// Create Data Object \\\-------------------#
 
 # *** Data source *** #
 
 # *** Mudule options and coordinates *** #
 
 
 # -------------------/// Active Data Object and basic plot \\\----#
 # *** DataOjb. handling *** #
 
 # *** Active DataObj. *** #
 
 
 # -------------------/// Advanced plot \\\------------------------#
 # *** Plot type *** #
 
 # *** Plot options *** #
 
 # *** Slicing options *** #
 
 # *** Coordinates Combo box *** #
 
 
    def plot_data_obj(self, obj_name):       
        d=flap.get_data_object(object_name=obj_name)
        # data = d.data
        # time = d.coordinate('Time')
        plt.close('all')
        d.plot()
        
    
    def update_source(self):
        if self.get_data_comboBox.currentText() == 'APDCAM':
            fn = os.path.join(thisdir,"flap_test_apdcam.cfg")
            flap.config.read(file_name=fn)
            
            _translate = QtCore.QCoreApplication.translate
            item = self.tableWidget.verticalHeaderItem(0)
            item.setText(_translate("MainWindow", "Datapath"))
            item = self.tableWidget.verticalHeaderItem(1)
            item.setText(_translate("MainWindow", "Scaling"))
            item = self.tableWidget.verticalHeaderItem(2)
            item.setText(_translate("MainWindow", ""))
            item = self.tableWidget.verticalHeaderItem(3)
            item.setText(_translate("MainWindow", ""))
            item = self.tableWidget.verticalHeaderItem(4)
            item.setText(_translate("MainWindow", "                         "))
            self.tableWidget.setSortingEnabled(False)
            item = self.tableWidget.item(0, 0)
            item.setText(_translate("MainWindow", "data"))
            item = self.tableWidget.item(1, 0)
            item.setText(_translate("MainWindow", "Digit"))
            item = self.tableWidget.item(2, 0)
            item.setText(_translate("MainWindow", ""))
            item = self.tableWidget.item(3, 0)
            item.setText(_translate("MainWindow", ""))
            item = self.tableWidget.item(4, 0)
            item.setText(_translate("MainWindow", ""))
            
            # default_options = {'Datapath':'data',
            #                    'Scaling':'Digit'
            #                    }
            
        if self.get_data_comboBox.currentText() == 'TESTDATA':
            fn = os.path.join(thisdir,"flap_tests.cfg")
            flap.config.read(file_name=fn)
            
            _translate = QtCore.QCoreApplication.translate
            item = self.tableWidget.verticalHeaderItem(0)
            item.setText(_translate("MainWindow", "Signal"))
            item = self.tableWidget.verticalHeaderItem(1)
            item.setText(_translate("MainWindow", "Scaling"))
            item = self.tableWidget.verticalHeaderItem(2)
            item.setText(_translate("MainWindow", "Frequency"))
            item = self.tableWidget.verticalHeaderItem(3)
            item.setText(_translate("MainWindow", "Length"))
            item = self.tableWidget.verticalHeaderItem(4)
            item.setText(_translate("MainWindow", "Samplerate               "))

            self.tableWidget.setSortingEnabled(False)
            item = self.tableWidget.item(0, 0)
            item.setText(_translate("MainWindow", "Sin"))
            item = self.tableWidget.item(1, 0)
            item.setText(_translate("MainWindow", "Volt"))
            item = self.tableWidget.item(2, 0)
            item.setText(_translate("MainWindow", "1e3"))
            item = self.tableWidget.item(3, 0)
            item.setText(_translate("MainWindow", "1e-2"))
            item = self.tableWidget.item(4, 0)
            item.setText(_translate("MainWindow", "1e3"))

            # default_options = {'Signal': 'Sin',
                               # 'Scaling': 'Volt',
                               # 'Frequency': 30,
                               # 'Length': 0.1,
                               # 'Samplerate':1e3,
                               # 'Image':'Gauss',
                               # 'Width': 1280,
                               # 'Height': 1024,
                               # 'Spotsize': 100
                               # }
                               
                               
        # W7X_abes
        # default_options = {'Datapath': 'data',
        #                    'Scaling':'Digit',
        #                    'Offset timerange': None,
        #                    'Amplitude calibration': False,
        #                    'Amplitude calib. path': 'cal',
        #                    'Amplitude calib. file': None,
        #                    'Phase' : None,
        #                    'State' : None,
        #                    'Start delay': 0,
        #                    'End delay': 0,
        #                    'Spatial calibration': False,
        #                    'Partial intervals': False
        #                    }
        
        
        
    def combo_box_change(self):
        print('Data source changed to: ' + self.get_data_comboBox.currentText())
        self.Big_textBrowser.append('Data source changed to: ' + self.get_data_comboBox.currentText())
        self.update_source()
        
        
    
    def create_data_obj(self,data_source, name, obj_name):
 
        if self.option_checkBox.isChecked():
            signal = self.tableWidget.item(0, 0)
            signal_val = signal.text()
            scaling = self.tableWidget.item(1, 0)
            scaling_val = scaling.text()
            frequensy = self.tableWidget.item(2, 0)
            if frequensy.text() == '':
                frequensy_val = ''
            else:
                frequensy_val = eval(frequensy.text())
            length = self.tableWidget.item(3, 0)
            if length.text() == '':
                length_val = ''
            else:
                length_val = eval(length.text())
            samplerate = self.tableWidget.item(4, 0)
            if samplerate.text() == '':
                samplerate_val = ''
            else:
                samplerate_val = eval(samplerate.text())            

            options = {'Signal':signal_val,
                        'Scaling':scaling_val,
                        'Frequency':frequensy_val,
                        'Length':length_val,
                        'Samplerate':samplerate_val}
            for k,v  in list(options.items()):
                if (v == None or v==''):
                  options.pop(k)
                  
            coor_key = self.option_key_line.text()
            coor_val = self.option_value_line.text()
            if (coor_key=='' or coor_val==''):
                coordinates = None
            else:    
                coor_val_val = eval(coor_val)
                coordinates = {coor_key : coor_val_val}
            d=flap.get_data(data_source, name=name, options=options, coordinates=coordinates, object_name=obj_name)
            self.obj_list.addItem(obj_name)
                
        else:
            d=flap.get_data(data_source, name=name, object_name=obj_name)
            self.obj_list.addItem(obj_name)
            
        self.Big_textBrowser.append(str(flap.list_data_objects(d)))
        self.Big_textBrowser.append('\n')
               
        
    
    def active_obj(self, obj_name):
        self.active_obj_line.setText(obj_name)
        d = flap.get_data_object(obj_name)
        coordinates = d.coordinate_names()
        units = []
        for i in range(len(d.coordinates)):
            e=d.coordinates[i].unit 
            units.append(e.unit)
        print('Coordinates in selected DataObject:')
        print(coordinates)
        self.Big_textBrowser.append('Coordinates in selected DataObject:')
        self.Big_textBrowser.append(str(coordinates))
        self.Big_textBrowser.append('\n')
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
            
            
            
    def plot_advanced(self, obj_name):
        self.active_obj_line.setText(obj_name)
        d = flap.get_data_object(obj_name)
        
        if self.slicing_opt_checkBox.isChecked():
            slice_key = self.slicing_opt_key.text()
            slice_val = self.slicing_opt_val.text()
            if (slice_key=='' or slice_val==''):
                slicing = None
            else:
                slicing = {slice_key : slice_val}
        else:
            slicing = None
            
            
        if self.plot_opt_checkBox.isChecked():
            axes = self.axes_line.text()
            if axes=='':
                axes = None
            else:
                axes = [axes]
                
            plot_opt_key = self.plot_opt_key.text()
            plot_opt_val = self.plot_opt_val.text()
            if (plot_opt_key=='' or plot_opt_val==''):
                plot_options = None
            else:
               plot_options = {plot_opt_key : plot_opt_val}
        else:
            axes = None
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
            flap.plot(obj_name, slicing=slicing, axes=axes, plot_options=plot_options)
            
        else:
            d.plot(axes=[data1,data2])
        
        


    def clear_list(self):
        self.obj_list.clear()
        flap.delete_data_object('*')
        print(flap.list_data_objects())
        self.Big_textBrowser.append(flap.list_data_objects())
        self.Big_textBrowser.append('\n')
        
        
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
        # text = fn.replace('C:/ShendR/Python/Flap GUI/','')
        # dialog = QtWidgets.QInputDialog()
        # text, ok = dialog.getText(self, "Load a DataObject", "Enter DataObject file name:")
        # home = Path.home()   
        # file = os.path.join(home, 'spi_gui_datapath.cfg')
        d=flap.load(file_name)
        self.obj_list.addItem(file_name.replace('.dat',''))
        self.Big_textBrowser.append(str(flap.list_data_objects(d)))
        self.Big_textBrowser.append('\n')
        coordinates = d[0].coordinate_names()
        print(coordinates)
        self.Big_textBrowser.append(str(coordinates))
        self.Big_textBrowser.append('\n')
        
        
    def del_data_obj(self, obj_name):
        flap.flap_storage.delete_data_object(obj_name)
        self.obj_list.takeItem(self.obj_list.currentRow())
        self.Big_textBrowser.append('Element deleted from list and Flap Storage')
        self.Big_textBrowser.append('/n')
        
        
        
    def close_all_plots(self):
        plt.close('all')
        self.Big_textBrowser.append('All plots are closed!')
        self.Big_textBrowser.append('/n')
        
    




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


