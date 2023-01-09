#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: expert
"""
#IMPORTING THE FILE TO LOAD THE MODEL
import pickle
filename="store.sav"
loaded_model=pickle.load(open(filename,"rb"))

#MAKING PREDICTION BASED ON INPUT VALUE
pred_result=loaded_model.predict([[3.07]])
print(pred_result)
