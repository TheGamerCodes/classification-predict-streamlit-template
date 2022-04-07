"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Text manipulation
import re
import string # For punctuation removal
import nltk # Toolkit for language processing
from nltk.corpus import stopwords # Redundant words
import re # Regular expression for text extraction
from nltk.tokenize import TreebankWordTokenizer # Tokenizing words
from nltk import WordNetLemmatizer # Lemmatizing tool
import time

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
sent = pd.read_csv("resources/sentiments.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.image("resources/ID-logo.gif")
	
	#st.subheader("Climate change Tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "About Author", "Sentiment Description"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Sentiment Description" page
	if selection == "Sentiment Description":
		st.info("General Information about the Class/Sentiment Labels from the raw dataset.")
		# You can read a markdown file from supporting resources folder

		st.write(sent[['Sentiment', 'Label']]) # will write the df to the page
		

		st.subheader('Class Distribution')

		val_count  = raw['sentiment'].value_counts()
		fig = plt.figure(figsize=(10,5))
		sns.barplot(val_count.index, val_count.values)
		# Add Figure
		st.pyplot(fig)

	# Building out the "About" page
	if selection == "About Author":
		st.info("General Information about the Author")
		# You can read a markdown file from supporting resources folder
		st.markdown("Ibrahim is an experienced Data Scientist and Software Developer with a passion for helping people and organizations to find solutions to their challenges.")

	# Building out the predication page
	if selection == "Prediction":
		st.title("Climate Change Tweet Classifer")
		st.info("Machine Learning Classification")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text")

		# Cleanup / Remove pattern:
		def cleaner(text):
			text  = re.sub(r'@[A-Za-z0-9_]+', '', text)
			text = re.sub(r'[#]', '', text)
			text = re.sub(r'RT : ', '', text)
			text = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', 'url', text)
			text = re.sub(r"www.\S+", "", text)
			text = re.sub("[^A-Za-z]"," ", text)
			text = re.sub(r'[...]', '', text)
			return text

		# Function to remove punctuation
		def remove_punctuation(message):
			return ''.join([l for l in message if l not in string.punctuation])

		def remove_stop_words(tokens):
			return [t for t in tokens if t not in stopwords.words('english')]    
    		

		classifier = st.selectbox('Select Classifier',('Logistic Regression', 'Random Forest'))

		st.write('You chose:', classifier)
		

		if st.button("Classify"):
			with st.spinner('One moment...'):
				time.sleep(1)
				# Apply preprocessing functions to user input:
				clean_txt = cleaner(tweet_text)
				# Remove punctuation:
				clean_txt = remove_punctuation(clean_txt)
				# Tokenize:
				clean_txt = TreebankWordTokenizer().tokenize(clean_txt)
				# Remove Stop Words:
				clean_txt = remove_stop_words(clean_txt)
				# Lemmatize:
				clean_txt = [WordNetLemmatizer().lemmatize(word) for word in clean_txt]
				# Restore to sentence:
				clean_txt = " ".join(clean_txt)


				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_txt]).todense()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice

					
				
				if classifier == 'Logistic Regression':
					predictor = joblib.load(open(os.path.join("resources/lr_model.pkl"),"rb"))
					prediction = predictor.predict(vect_text)
					

				if classifier == 'Random Forest':
					predictor = joblib.load(open(os.path.join("resources/rf_clf.pkl"),"rb"))
					prediction = predictor.predict(vect_text)
					

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				
				#st.balloons()

				if prediction == 1:
					st.success("Text Classified as: {}".format('Pro - Believer'))
				elif prediction == 2:
					st.success("Text Classified as: {}".format('News'))
				elif prediction == 0:
					st.success("Text Classified as: {}".format('Neutral'))
				else:
					st.success("Text Classified as: {}".format('Anti - Non Believer'))


				#score = str(prediction.score()) + "%"

				#st.metric(label="Accuracy", value=score) 

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
