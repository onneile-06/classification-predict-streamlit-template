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
news_vectorizer = open("resources/count_vectorizer.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("The Analysts Hive - Twitter Classifier")
    st.subheader("Climate change tweets classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "EDA", "About"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":

        logo_image = "resources/imgs/twetterpic.png"
        st.image(logo_image, caption='Twitter image', use_column_width=True)

        st.info("General Information")
        st.info('• Testing data is used to determine the performance of the trained model, whereas training data is used to train the machine learning model.')
        st.info('• Training data is the power that supplies the model in machine learning, it is larger than testing data. Because more data helps to more effective predictive models.') 
        st.info('• When a machine learning algorithm receives data from our records, it recognizes patterns and creates a decision-making model.')
        st.info('• To avoid overfitting, it essential to use separate training and testing data. When a machine learning model learns the training data too well, it becomes hard to generalize to new data. This may happen if the training data is insufficient or not representative of the real-world data on which the model will be used.')

        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page
        
        st.subheader("Class Description")
        if st.checkbox('Classes'):
            st.write('These classes are categorizations used to classify tweets based on their content regarding climate change:'

                   
                   '• News (Class 2): Tweets falling into this category provide factual news about climate change. These tweets are important as they disseminate accurate and up-to-date information about climate-related events, scientific findings, policy changes, and other significant developments. They help educate the public and raise awareness about the realities and impacts of climate change.'

                   
                   '• Pro (Class 1): Tweets categorized as "Pro" support the belief in man-made climate change. They may advocate for actions to address climate change, highlight the scientific consensus on the issue, or promote sustainable practices. These tweets play a crucial role in fostering public understanding and acceptance of the reality of anthropogenic climate change. They contribute to building momentum for collective action and policy initiatives aimed at mitigating climate change.'

                   
                   '• Neutral (Class 0): Tweets labeled as "Neutral" neither support nor refute the belief in man-made climate change. While they may mention climate-related topics, they do not explicitly take a stance on the issue. These tweets are important for providing balanced perspectives and fostering open dialogue about climate change. They offer opportunities for individuals to engage in discussions, share diverse viewpoints, and critically evaluate information.'

                   
                   '• Anti (Class -1): Tweets categorized as "Anti" express disbelief in man-made climate change. They may deny the existence of climate change, challenge the scientific consensus, or oppose climate-related policies and initiatives. While these tweets represent dissenting views, they can also contribute to misinformation and confusion surrounding climate change. It is important to critically assess and address misinformation to ensure that accurate information prevails in public discourse.')



    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with our Machine Learning Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        # Model selection
        model_options = ["Logistic_Regression", "SVM_model", "Random_Forest"]
        model_choice = st.selectbox("Select Model", model_options)

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
        
            # Dictionary to map model choice to pickle file
            model_files = {
                "Logistic_Regression": "logistic_regression_model.pkl",
                "SVM_model": "SVM_model.pkl",
                "Random_Forest": "Random_Forest_model.pkl"
            }
        
            # Load your .pkl file with the model of your choice + make predictions
            model_file = model_files.get(model_choice, "logistic_regression_model.pkl")  # Default to logistic regression if not found
            predictor = joblib.load(open(os.path.join("resources", model_file), "rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            st.success(f"Text Categorized as: {prediction}")

    #Adding 'EDA' dropdown
            
    # Handle the new "EDA" selection
    if selection == "EDA":
        st.title("Exploratory Data Analysis (EDA)")
        st.info("Visual Insights into the Dataset")

        # Placeholder for images
        graph_image_paths = [
            "resources/imgs/dist_sent.png",  # Replace these paths with actual paths to your images
            "resources/imgs/twt_dist_sent.png",
            "resources/imgs/top20common.png",
            "resources/imgs/wordcloud.png",
            "resources/imgs/sent_msg.png",
            "resources/imgs/piechart.png",
        ]

        # Placeholder for captions - modify these with actual captions for your graphs
        graph_captions = [
            "Distribution of Sentiments",
            "Tweet Length Distribution by Sentiment",
            "Top 20 Most Common Words",
            "Word Cloud",
            "Sentiment by Message Length",
            "Sentiment Pie Chart"
        ]

        # Displaying images with captions
        for path, caption in zip(graph_image_paths, graph_captions):
            st.image(path, caption=caption, use_column_width=True)

    # Handle the new "About" selection
    if selection == "About":

        st.title('Company Information')


        st.subheader('About Us')
    
        # Display company description using st.text() or st.markdown()
        company_description = ("Our company is dedicated to reducing environmental impact and promoting sustainability. We offer a range of products and services designed to help individuals and businesses lessen their carbon footprint and contribute to a greener future.")
        st.markdown(company_description)


         # Display company logo
        logo_path = "resources/imgs/logo.jpg"
        st.image(logo_path, caption='Company Logo', use_column_width=True)




        st.title("About This App")
        st.info("This app is developed by The Analysts Hive, aiming to classify tweets on climate change into different sentiments. It serves as a tool for raising awareness about climate change and understanding public sentiment on this critical issue.")
        

        # Adding an image
        image_paths = ['resources/imgs/onneile.jpg', 'resources/imgs/masego.jpg', 'resources/imgs/mpho.jpg', 'resources/imgs/minnie.jpg', 'resources/imgs/zinhle.jpg', 'resources/imgs/zenani.jpg' ]
        st.image(image_paths, caption=['Onneile', 'Masego', 'Mpho', 'Minnie', 'Zinhle', 'Zenani'], use_column_width=True, width=300)

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
