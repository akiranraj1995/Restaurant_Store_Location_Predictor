#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: expert
"""

#STREAMLIT WEB APPLICATION
#IMPORTING THE REQUIRED LIBRARIES FOR WEB APP

import streamlit as st
import numpy as np
import pickle

#USING STREAMLIT WEB APP 
#ADDING TITLE AND HEADER
st.title("FOOD RESTAURANT LOCATION PREDICTOR")
st.header("Best Place to put up a Store!")

#LOADED THE PICKLE MODEL
loaded_model=pickle.load(open("store.sav","rb"))  

#IMPORTING LIBRARY TO PUT A IMAGE
from PIL import Image

#LOADING THE IMAGE FROM LOCAL DIRECTORY
image = Image.open('HT.jpg')
st.image(image, caption='FOOD RESTAURANT LOCATION PREDICTOR')
 

#DEFINED FUNCTION FOR PREDICTION  
def location_predict(x):
    pred_result=loaded_model.predict([[x]])
    print(pred_result)

#DEFINED A MAIN FUNCTION FOR TAKING THE NEW INPUT               
def main():
    #USING x AS INPUT VARIABLE
    x=st.slider("Enter the Population",0.000,25.000,1.000)

    #CONVERTING INPUT VALUE TO ARRAY
    input_data_as_nparray=np.asarray(x)

    #RESHAPED FROM 1D to 2D
    input_data_as_nparray_and_reshaped=input_data_as_nparray.reshape(1,1)
    
    #MAKING PREDICTION
    pred_result=loaded_model.predict(input_data_as_nparray_and_reshaped)
    print(pred_result)
    
    #CONDITION IF PREDICT BUTTON IS CLICKED
    output="" 
    if st.button("Predict"): 
        
        st.success("The Probability of Profit for Store Location for the given Population is {}".format(pred_result))
        #CONDITION FOR PROFITABLE BUSINESS OR NOT
        st.subheader("Your Result: ")
        if pred_result[0]>20.0: 
            output="Excellent Location to put up the Store..Profitable Business"
        elif pred_result[0]<0.0:  
             output= "Profit is coming Negative!! It would be a huge Loss..Bad Location to put up the Store"
        else :
             output= "Good Location to put up the Store" 
             
    st.write(output)
    
if __name__=='__main__': 
    main()               
            

        


    
