# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:38:28 2022

@author: ShendR
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
from scipy import signal
import copy
import time
import math
from pathlib import Path
import traceback, sys
from tkinter import filedialog
from tkinter import *

from flap_gui_canvas import Ui_MainWindow
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
                
            # Canvas on GUI       
        self.Big_graphicsView = Canvas(parent=self.centralwidget)
        self.Big_graphicsView.setMinimumSize(QtCore.QSize(550, 500))
        self.Big_graphicsView.setObjectName("Big_graphicsView")
        self.gridLayout_4.addWidget(self.Big_graphicsView, 0, 0, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_4, 1, 0, 1, 2)
        
            # Toolbar on GUI
        self.toolbar = NavigationToolbar(self.Big_graphicsView, self.Big_graphicsView) 
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)
        
        self.create_obj_button.clicked.connect(lambda: self.create_data_obj(self.get_data_comboBox.currentText(),
                                                                            self.data_name_text.toPlainText(),
                                                                            self.obj_name_text.toPlainText()))
        
        self.plot_obj_button.clicked.connect(lambda: self.plot_data_obj(self.Big_graphicsView,
                                                                        self.obj_list.currentItem().text()))
             
        self.save_obj_button.clicked.connect(lambda: self.save_data_obj(self.obj_list.currentItem().text()))
        
        self.load_obj_button.clicked.connect(lambda: self.load_data_obj())
        
        self.clear_obj_button.clicked.connect(lambda: self.obj_list.clear())
 
        self.get_data_comboBox.currentIndexChanged.connect(lambda: self.combo_box_change())
 
    
    def plot_data_obj(self, canvas, obj_name):       
        d=flap.get_data_object(object_name=obj_name)
        data = d.data
        time = d.coordinate('Time')
        # d.plot()
        canvas.fig.clf()
        axes = canvas.fig.subplots(1,1)   
        part1 = axes.twinx()
        axes.plot(time[0], data)
        canvas.draw() 

        
    
    def update_gui_data(self):
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
        self.update_gui_data()
        
        
    
    def create_data_obj(self,data_source, name, obj_name):
 
        if self.option_checkBox.isChecked():
            signal = self.tableWidget.item(0, 0)
            signal_val = signal.text()
            scaling = self.tableWidget.item(1, 0)
            scaling_val = scaling.text()
            frequensy = self.tableWidget.item(2, 0)
            frequensy_val = eval(frequensy.text())
            length = self.tableWidget.item(3, 0)
            length_val = eval(length.text())
            samplerate = self.tableWidget.item(4, 0)
            samplerate_val =  eval(samplerate.text())
            options = {'Signal':signal_val,
                       'Scaling':scaling_val,
                       'Frequency':frequensy_val,
                       'Length':length_val,
                       'Samplerate':samplerate_val}
            d=flap.get_data(data_source, name=name, options=options, object_name=obj_name)
            self.obj_list.addItem(obj_name)
                
        else:
            d=flap.get_data(data_source, name=name, object_name=obj_name)
            self.obj_list.addItem(obj_name)
            
        self.Big_textBrowser.append(str(flap.list_data_objects(d)))
        coordinates = d.coordinate_names()
        print(coordinates)
        self.Big_textBrowser.append(str(coordinates))
       
        
    
    def list_item_clicked(self):
        pass



    def clear_list(self):
        self.obj_list.clear()
        flap.delete_data_object('*')
        print(flap.list_data_objects())
        self.Big_textBrowser.append(flap.list_data_objects())
        
        
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
        coordinates = d[0].coordinate_names()
        print(coordinates)
        self.Big_textBrowser.append(str(coordinates))
        
        
        
    


class Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.fig.clear()
        self.fig.patch.set_facecolor('None')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('None')
        self.axes.axis('off')
        super(Canvas, self).__init__(self.fig)      



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


