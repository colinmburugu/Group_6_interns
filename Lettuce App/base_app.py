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
from datetime import datetime
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt





# The main function where we will build the actual app
def main():
	"""Lettuce App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Lettuce Price Forecasting")
	# st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Forecasting", "Information", "EDA", "Models"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		

	# Building out the predication page
	if selection == "Forecasting":
		st.info("Forecasting price with ML Models")

	
		st.markdown("Lettuce is one of the most versatile and valuable crops to grow in a polytunnel in the UK. ")
		# st.date_input("Select date to forecast", datetime.date(2019, 7, 6))
		# st.date_input("Select date to forecast", value=datetime.date, min_value=datetime.date, max_value=datetime.date)
		models = ["Multinomial Naive Bayes", "Support Vector Classifier", "K-nearest neighbours", "Random Forest Classifier","Logistic regression"]
		seasons = ["Winter", "Autumn", "Summer", "Spring"]
		regions = ["London", "South East", "South West", "East Anglia", "East Midlands", "West Midlands", "Yorkshire & Humber", "North West", "North", "Wales", "Scotland", "Northern Ireland"]
		selection_season = st.selectbox("Select your season", seasons)
		selection_region = st.selectbox("Select your region", regions)
		selection_model = st.selectbox("Select your model", models)
		

		if st.button("Forecast"):
		
			st.success("The price for Lettuce is: £0.71")
	

	# Building out the EDA page
	if selection == "EDA":
		st.info("Exploratory Data Analysis")
		

	# Building out the Models page 
	if selection == "Models":
		st.subheader("Models used")
		models = ["Multinomial Naive Bayes", "Support Vector Classifier", "K-nearest neighbours", "Random Forest Classifier","Logistic regression"]
		selection_model = st.selectbox("Select your model", models)
		if selection_model == "Support Vector Classifier":
			st.info("Support Vector Classifier")
			st.markdown("SVM offers very high accuracy compared to other classifiers such as logistic regression, and decision trees. It is known for its kernel trick to handle nonlinear input spaces")
		if selection_model == "Multinomial Naive Bayes":
			st.info("Multinomial Naive Bayes")
			st.markdown("The Multinomial Naive Bayes algorithm is a Bayesian learning approach popular in Natural Language Processing (NLP). The program guesses the tag of a text, such as an email or a newspaper story, using the Bayes theorem. It calculates each tag's likelihood for a given sample and outputs the tag with the greatest chance ")
		if selection_model == "Random Forest Classifier":
			st.info("Random Forest Classifier")
			st.markdown("Random forests is a supervised learning algorithm. It can be used both for classification and regression. It is also the most flexible and easy to use algorithm. A forest is comprised of trees. It is said that the more trees it has, the more robust a forest is. Random forests creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting.")
		if selection_model == "Logistic regression":
			st.info("Logistic regressionr")
			st.markdown("Logistic Regression is one of the most simple and commonly used Machine Learning algorithms.It is easy to implement and can be used as the baseline for any binary classification problem. Its basic fundamental concepts are also constructive in deep learning.")
		if selection_model == "K-nearest neighbours":
			st.info("K-nearest neighbours")
			st.markdown("The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. ")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()