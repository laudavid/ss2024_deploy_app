import streamlit as st 
import pandas as pd

from streamlit_gsheets import GSheetsConnection
from preprocess import text_preprocessing
from inference import predict
from utils import load_artifact

st.set_page_config(layout="wide")


# Load saved models
tfidf_vectorizer = load_artifact("tfidf-vectorizer.sav")
model = load_artifact("logistic_regression.sav")

# Page
st.image("images/movie-header.jpg")
st.markdown("# Movie sentiment analysis app 🎬")
st.markdown("""This app provides a **sentiment analysis model** trained on movie reviews. <br>
            Users can **write their own review** or select one from an **external database**.""", 
            unsafe_allow_html=True)

option = st.radio("**Select an option**", ["Write a review", "Find a movie review"])

st.divider()


# Option 1: Write a review
if option == "Write a review":
    st.markdown("### Write a review 📝")
    review = st.text_area('Write the review here', 
    'I hated the movie. The story and the actors were terrible.') 

    st.markdown("  ")
    run_model = st.button("Run model", type="primary", key="write_review")

    if run_model:
        st.markdown(" ")
        result, probas = predict(review, tfidf_vectorizer, model)

        if result[0] == "positive":
            st.success(f"**Result** 👍: The review is {result[0]}.")
        
        else:
            st.error(f"**Result** 👎: The review is {result[0]}")



# Option 2: Use external data
elif option == "Find a movie review":

    # Connect app to google sheet 
    conn = st.connection("gsheets", type=GSheetsConnection)
    reviews_df = conn.read(ttl="30m", usecols=[0,1,2], nrows=11)

    st.markdown("### Find a movie review 🔎")
    movie = st.selectbox('Select a movie', reviews_df["Title"].to_list())
    review = reviews_df.loc[reviews_df["Title"]==movie, "Review"].to_list()[0]     
    st.markdown(f"""**Review**: <br> {review }""", unsafe_allow_html=True)

    st.markdown("  ")
    run_model = st.button("Run model", type="primary", key="database")

    if run_model:
        st.markdown(" ")
        
        result, probas = predict(review, tfidf_vectorizer, model)
        
        if result[0] == "positive":
            st.success(f"**Result** 👍: The review is {result[0]}.")
        
        else:
            st.error(f"**Result** 👎: The review is {result[0]}")
