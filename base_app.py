"""
Simple Streamlit web application for serving developed classification models.

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
import joblib, os

# Data dependencies
import pandas as pd





# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("The Analysts Hive - Tweet Classifier")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "About"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"), "rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            st.success("Text Categorized as: {}".format(prediction))


    

            # Handle the new "About" selection
    if selection == "About":
        
        st.title('Company Information')
    
        # Display company logo
        logo_path = "resources/imgs/logo.jpg"
        st.image(logo_path, caption='Company Logo', use_column_width=True)

    
        st.subheader('About Us')
    
        # Display company description using st.text() or st.markdown()
        company_description = ("Our company is dedicated to reducing environmental impact and promoting sustainability. We offer a range of products and services designed to help individuals and businesses lessen their carbon footprint and contribute to a greener future.")
        st.markdown(company_description)
    
        

        st.title("About This App")
        st.info("This app is developed by The Analysts Hive, aiming to classify tweets on climate change into different sentiments. It serves as a tool for raising awareness about climate change and understanding public sentiment on this critical issue.")
        st.markdown(company_description)




        # Adding an image
        image_paths = ['resources/imgs/onneile.jpg', 'resources/imgs/masego.jpg', 'resources/imgs/mpho.jpg', 'resources/imgs/minnie.jpg', 'resources/imgs/zinhle.jpg', ]  # Make sure to use the correct path or URL to your image
        st.image(image_paths, caption=['Onneile', 'Masego', 'Mpho', 'Minnie', 'Zinhle'], use_column_width=True, width=300)

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
