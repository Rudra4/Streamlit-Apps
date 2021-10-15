# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.title("GetHomePrice.com")
st.text("Get best price for the home of your dreams!")
mydata = pd.read_csv("kc_house_data.csv")
st.image("housePrice.jpg")
st.video("https://www.youtube.com/watch?v=CUTrb3aYh-o")
set_price = st.slider("Price Range", min_value=int(mydata['price'].min()), max_value=int(mydata['price'].max()), step=50, value=int(mydata['price'].min()))
st.text("You have selected a price of USD " + str(set_price))
fig = px.scatter_mapbox(mydata.loc[mydata['price']<set_price], lat = 'lat', lon = 'long', color = 'sqft_living', size = 'price')
fig.update_layout(mapbox_style = 'open-street-map')
st.plotly_chart(fig)

st.header('Price Predictor')
sel_box = st.selectbox("Select a prediction method", ['Linear', 'Ridge', 'Lasso'], index = 0)
multi_sel = st.multiselect("Select Parameters to be used for better prediction of price", ['sqft_living', 'sqft_lot', 'sqft_basement'])
mydata_new = []
mydata_new = mydata[multi_sel]
mydata_new['bedrooms'] = mydata['bedrooms']
mydata_new['bathrooms'] = mydata['bathrooms']
if sel_box == 'Linear': 
    x = mydata_new
    y = mydata['price']
    model = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    reg = model.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    st.text('Intercept = ' + str(reg.intercept_))
    st.text("Coefficients: " + str(reg.coef_))
    st.text('R^2 Score = ' + str(r2_score(y_test, y_pred)))
    st.text('Mean Squared Error = ' + str(mean_squared_error(y_test, y_pred)))
elif sel_box == 'Ridge': 
    x = mydata_new
    y = mydata['price']
    model = Ridge()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    reg = model.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    st.text('Intercept = ' + str(reg.intercept_))
    st.text("Coefficients: " + str(reg.coef_))
    st.text('R^2 Score = ' + str(r2_score(y_test, y_pred)))
    st.text('Mean Squared Error = ' + str(mean_squared_error(y_test, y_pred)))
else: 
    x = mydata_new
    y = mydata['price']
    model = Lasso()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    reg = model.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    st.text('Intercept = ' + str(reg.intercept_))
    st.text("Coefficients: " + str(reg.coef_))
    st.text('R^2 Score = ' + str(r2_score(y_test, y_pred)))
    st.text('Mean Squared Error = ' + str(mean_squared_error(y_test, y_pred)))
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.regplot(y_test, y_pred)
st.pyplot()

count = 0
pred_value = 0
for i in mydata_new.keys():
    try:
        val = st.text_input('Enter number/value of ' + i)
        pred_value += float(val) * reg.coef_[count]
        count += 1
    except:
        pass
st.text('The predicted price is USD ' + str(pred_value + reg.intercept_))

st.header("Application Details:")
img = st.file_uploader("Upload Application")
st.text("Enter your details where our representative can contact you:")
st.text("Enter your address")
address = st.text_area("Your address here")
date = st.date_input("Enter a suitable date")
date = st.time_input("Enter a suitable date")
if st.checkbox("I confirm that the above entered details are correct", value = False):
    st.write("Thanks for your confirmation!")
st.number_input("Rate your experience on our website", min_value = 1, max_value = 5, value = 3, step = 1)