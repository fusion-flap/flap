# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:13:45 2018

XML file handling for FLAP.

@author: Sandor Zoletnik  (zoletnik.sandor@ek-cer.hu)
Centre for Energy Research
"""

import xml.etree.ElementTree as ET
import copy

try:
    if (VERBOSE):
        print("Importing flap_xml")
except NameError:
    pass

class FlapXml:
    def __init__(self):
        self.sections = []
        self.section_elements = []
        self.head = None

    def create_head(self, head_name='FLAPXml', attrib=None):
        attrib_xml = copy.deepcopy(attrib)
        if (attrib is None):
            attrib_xml = {'FLAPVersion':'1.0'}
        else:
            if (type(attrib) is not dict):
                raise TypeError("FLAPXml.creade_head: attrib argument should be a dictionary.")
            attrib_xml['FLAPVersion'] = '1.0'
        self.top = ET.Element(head_name,attrib=attrib_xml)

    def add_element(self,
                    section=None,
                    element=None,
                    value=None,
                    unit=None,
                    comment=None,
                    value_type=None):
        if (section is None) or (element is None) or (value is None):
            raise ValueError("FLAPXml.add_element: Missing input data")

        if (type(value) == int):
            value_str = str(value)
            type_str = 'long'
        elif (type(value) == float):
            value_str = str(value)
            type_str = "float"
        elif (type(value) == str):
            value_str = value
        else:
            raise ValueError(" ABES_Xml.addElement: unsuitable input data type")

        if (value_type is not None):
            type_str = value_type

        if (unit is None):
            unit_str = "none"
        else:
            unit_str = unit

        try:
            section_index = self.sections.index(section)
            s = self.section_elements[section_index]
        except Exception:
            s = ET.SubElement(self.top, section)
            self.sections.append(section)
            self.section_elements.append(s)
            section_index = len(self.section_elements) - 1

        if (comment is None):
            child = ET.SubElement(s, element, attrib={"Value":value_str,        #DEFINED BUT UNUSED
                                                      "Unit":unit_str,
                                                      "Type":type_str,})
        else:
            child = ET.SubElement(s, element, attrib={"Value":value_str,        #DEFINED BUT UNUSED
                                                      "Unit":unit_str,
                                                      "Type":type_str,
                                                      "Comment":comment})

    def get_element(self,section,element):
        try:
            i = self.sections.index(section)
        except ValueError:
            raise ValueError("Section " + section + " not found in xml file.")
        for e in self.section_elements[i]:
            if e.tag == element:
                return e.attrib
        raise ValueError("Element '{:s}' not found in section '{:s}' in xml file.".format(element,section))                       
        return None

    def write_file(self,filename):
        ET.ElementTree(self.top).write(filename)

    def read_file(self,filename):
        self.__init__()
        tree = ET.parse(filename)
        self.head = tree.getroot()
        for child in self.head:
            self.sections.append(child.tag)
            self.section_elements.append([])
            sect = self.sections.index(child.tag)
            for element in child:
                self.section_elements[sect].append(element)


def test_write():
    m = FlapXml()
    m.create_head()
    m.add_element(section="ADCSettings",
                  element="Trigger",
                  value = -1,
                  unit="s",\
                  comment="Trigger: <0: manual,otherwise external with this delay")
    m.add_element(section="ADCSettings",
                  element="ADCMult",
                  value=20,
                  value_type='int')
    m.write_file("xx.xml")


def test_read():
    m = FlapXml()
    m.read_file("xx.xml")
