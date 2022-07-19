"""

    Simple Streamlit webserver application for serving developed time series
	models.

    Author: McMunashe Munemo

    Note:

    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. 

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from datetime import datetime, date, time
import streamlit as st
import joblib,os
from PIL import Image
import plost # streamlit visualisation
from st_aggrid import GridOptionsBuilder, AgGrid 

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")
# The main function where we will build the actual app
def main():
	"""Lettuce App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Lettuce Price Forecasting")
	# st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Forecasting", "EDA", "Models"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Home":
		st.info("General Information")
		# st.metric(label="Lettuce price today", value="£0.71", delta="£0.2")
		col1, col2, col3 = st.columns(3)
		col1.metric(label="Lettuce price yesterday", value="£0.70")
		col2.metric(label="Lettuce price today", value="£0.72", delta="£0.2")
		col3.metric(label="Lettuce price tommorow", value="£0.69", delta="-£0.3") 

		st.header("Background")
		st.markdown("Lettuce (Lactuca sativa) is one of the world's most economically important leafy vegetable crops. It is one of the most popular salad crops and is well-known for its delicate, crispy texture and slightly bitter taste, as well as its fresh condition of milky juice. It is among the most widely grown salad vegetable crop because of its high demand. Lettuce is rich in vitamins as well as minerals such as calcium and iron. It is commonly served alone or with dressing as a salad with tomato, carrot, cucumber, or other salad vegetables. ")

		image = Image.open('resources/imgs/iceberg.png')
		st.image(image, caption='Iceberg lettuce')

		st.markdown("This app forecasts the price of iceberg lettuce in the UK, using data collected by the Office of National Statistics (ONS). Price forecasting is the prediction of a commodity or product price based on various factors such as its characteristics, demand, seasonal trends, the prices of other commodities, offers from various suppliers, and so on.")

	# Building out the predication page
	if selection == "Forecasting":
		st.info("Forecasting price with ML Models")

	
		st.markdown("Lettuce is one of the most versatile and valuable crops to grow in a polytunnel in the UK. ")
		
		# st.header("Official Date Picker")
		train = pd.read_csv("./data/train.csv")
		train['date'] = train['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
		dates = list(train['date'])
		st.date_input('Pick a date to forecast', value=dates)
		# st.date_input('2023')
		# st.date_input("Select date to forecast", datetime.date(2019, 7, 6))
		# st.date_input("Select date to forecast", value=datetime.date, min_value=datetime.date, max_value=datetime.date)
		models = ["Vector Auto Regression","Facebook Prophet"]
		# seasons = ["Winter", "Autumn", "Summer", "Spring"]
		regions = ["London", "South East", "South West", "East Anglia", "East Midlands", "Yorkshire & Humber", "North West", "Scotland"]
		# selection_season = st.selectbox("Select your season", seasons)
		selection_region = st.selectbox("Select your region", regions)
		selection_model = st.selectbox("Select your model", models)
		

		if st.button("Forecast"):
		
			st.success("The price for Lettuce is: £0.71")
	

	# Building out the EDA page
	if selection == "EDA":
		st.info("Exploratory Data Analysis")
		# Data
		train = pd.read_csv("./data/train.csv")
		train['date'] = train['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
		train = train.set_index('date')
		# st.dataframe(lettuce)
		# Row C
		# c1, c2 = st.columns((7,3))
		# with c1:
		col1, col2 = st.columns(2)
		col1.metric(label="Average Iceberg lettuce price", value="£0.72")
		col2.metric(label="Average Round lettuce price", value="£0.49", delta="-£0.23")

		regions = ["All regions","London", "South East", "South West", "East Anglia", "East Midlands", "Yorkshire & Humber", "North West", "Scotland"]
		selection_region = st.selectbox("Select your region", regions)
		if selection_region == "All regions":
			fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,6))
			for i, ax in enumerate(axes.flatten()):
				data = train[train.columns[i]]
				ax.plot(data, color='red', linewidth=1)
				# Decorations
				ax.set_title(train.columns[i])
				ax.xaxis.set_ticks_position('none')
				ax.yaxis.set_ticks_position('none')
				ax.spines["top"].set_alpha(0)
				ax.tick_params(labelsize=6)
			plt.set_loglevel('WARNING') 
			plt.tight_layout()
			st.pyplot(fig)
		if selection_region == "London":	
			st.line_chart(train['London'])
		if selection_region == "South East":	
			st.line_chart(train['South East'])
		if selection_region == "South West":	
			st.line_chart(train['South West'])	
		if selection_region == "East Anglia":	
			st.line_chart(train['East Anglia'])
		if selection_region == "East Midlands":	
			st.line_chart(train['East Midlands'])
		if selection_region == "Yorkshire & Humber":	
			st.line_chart(train['Yorkshire & Humber'])
		if selection_region == "North West":	
			st.line_chart(train['North West'])
		if selection_region == "Scotland":	
			st.line_chart(train['Scotland'])
		
		
		# st.markdown('### Bar chart chart')
		# st.line_chart(train)
		# plost.bar_chart(
		# data=train,
		# bar='region',
		# value= 'price',
		# color ='region',
		# direction='horizontal',
		# legend = None)

		# ob = GridOptionsBuilder.from_dataframe(train)
		# ob.configure_column('region', rowGroup=True)
		# ob.configure_column('price', aggFunc='sum')
		# st.markdown('# AgGrid')
		# AgGrid(train, ob.build(), enable_enterprise_modules=True)

		# y='price')
		# y_unit='day',
		# color='temp_max',
		# aggregate='median',
		# legend=None)
		# with c2:
		# st.markdown('### Donut chart')
		# plost.donut_chart(
		# 	data=lettuce,
		# 	theta='price',
		# 	color='item')
		

	# Building out the Models page 
	if selection == "Models":
		st.subheader("Models used")
		models = ["Vector Autoregression","Facebook Prophet"]
		selection_model = st.selectbox("Select your model", models)
		if selection_model == "Vector Autoregression":
			st.info("Vector Autoregression")
			st.markdown("Vector Autoregression (VAR) is a multivariate forecasting algorithm that is used when two or more time series influence each other.\n\n"
			              "That means, the basic requirements in order to use VAR are:\n\n"

							"\t 1. You need at least two time series (variables)\n"
							"\t2. The time series should influence each other.\n"
							"###### Alright. So why is it called ‘Autoregressive’?\n\n"

							"It is considered as an Autoregressive model because, each variable (Time Series) is modeled as a function of the past values, that is the predictors are nothing but the lags (time delayed value) of the series.\n\n"

							"###### Ok, so how is VAR different from other Autoregressive models like AR, ARMA or ARIMA?\n\n"

							"The primary difference is those models are uni-directional, where, the predictors influence the Y and not vice-versa. Whereas, Vector Auto Regression (VAR) is bi-directional. That is, the variables influence each other.")
		if selection_model == "Facebook Prophet":
			st.info("Facebook Prophet")
			st.markdown("The Multinomial Naive Bayes algorithm is a Bayesian learning approach popular in Natural Language Processing (NLP). The program guesses the tag of a text, such as an email or a newspaper story, using the Bayes theorem. It calculates each tag's likelihood for a given sample and outputs the tag with the greatest chance ")
		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()