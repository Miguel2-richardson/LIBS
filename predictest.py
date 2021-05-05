# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:16:41 2021

@author: Miguel
"""

from spectraasc import dataread, spectraread, getpeaks, featuresread, featurecompare
import matplotlib.pyplot as plt
import numpy as np
import openpyxl as px
import tensorflow as tf
from tensorflow import keras

File = ["Pure Aluminum_1.asc","Pure Aluminum_2.asc","Pure Aluminum_3.asc",
        "Pure Aluminum_4.asc","Pure Aluminum_5.asc","Pure Aluminum_6.asc",
        "Stainless1219_1.asc","Stainless1219_2.asc","Stainless1219_3.asc",
        "Stainless1219_4.asc","Stainless1219_5.asc","Pure Copper_1.asc"]
Data = []
Wavelen = []
trig = 0.03

for i in File:
    wavelen, data = spectraread(i)
    Data.append(data)
    Wavelen.append(wavelen)



stuff = featurecompare(Wavelen, Data, trig)

model = keras.models.load_model("test.h5")

predictions = model.predict(stuff)
Max = np.argmax(predictions, axis = 1)


plt.plot(stuff[8])
# plt.plot(stuff[7])
# plt.plot(stuff[8])
# plt.plot(stuff[10])
#
