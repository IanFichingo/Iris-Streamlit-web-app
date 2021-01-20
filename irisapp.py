import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn

#loading the trained model
pickle_in = open('bestmodel.pkl', 'rb')
classifier = pickle.load(pickle_in)


#defining the heading of the app
st.write("""
# Iris Flower Prediction App

The App predicts the type of Iris flower Species
""")

#pandas dataframe
st.write('This is a wonderful App')


#sidebar inputs
def user_input():
    st.sidebar.header("User Input Features")
    sepal_length = st.sidebar.slider('SepalLengthCm', 4.3, 7.9, 5.3)
    sepal_width = st.sidebar.slider('SepalWidthCm', 2.0, 4.4, 3.1)
    petal_length = st.sidebar.slider('PetalLengthCm', 1.0, 6.9, 4.0)
    petal_width = st.sidebar.slider('PetalWidthCm', 0.1, 2.5, 1.1)
    data = {'sepal_length': sepal_length,
             'sepal_width': sepal_width,
             'petal_length': petal_length,
             'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df1 = user_input()

st.subheader('The user Input parameter')
st.write(df1)

#predict
prediction = classifier.predict(df1)


st.subheader('predicting the Flower Type')
st.write(prediction)
