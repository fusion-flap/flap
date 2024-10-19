# FLAP Version 1.2 User's Guide

S. Zoletnik, M. Vécsei, Centre for Energy Research

M. Vavrik, BME NTI

zoletnik.sandor@ek-cer.hu

Document version 1.71, 19 April, 2022

## Contents {#contents .TOC-Heading}

[Introduction [3](#introduction)](#introduction)


## Introduction

The Fusion Library of Analysis Programs (FLAP) is a Python framework to work with large multi-dimensional data sets especially for turbulence data evaluation in fusion experiments. Data are stored in [data objects](#flap.data_object.DataObject) together with coordinates and data names, thus the built-in plotting functions create figures with correct axes. The data set can be sliced to reduce dimensions and thus enable visualization of more than 2D data sets. FLAP is a modular package: data are read through modules which register themselves to the FLAP framework. This way data are read through a uniform interface by defining data source, experiment ID and data name. Also coordinate conversion is done by functions in the data source modules.
