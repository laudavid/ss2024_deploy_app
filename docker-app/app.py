# import pandas as pd

# from preprocess import text_preprocessing
# from inference import predict
# from utils import load_artifact
import os, re, string
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum

app=FastAPI()
handler=Mangum(app)

def load_artifact(file):
    path_artifact = r"saved_models/"
    artifact = joblib.load(os.path.join(path_artifact, file))
    return artifact

# Load saved models
tfidf_vectorizer = load_artifact("tfidf-vectorizer.sav")
model = load_artifact("logistic_regression.sav")

nltk.data.path.append("nltkdata")

# review = 'I hated the movie. The story and the actors were terrible.'
# review = 'I love the move. The story and the actors were excellent.'
# review_clean = text_preprocessing(review)
# result, probas = predict(review_clean, tfidf_vectorizer, model)

# if result[0] == "positive":
#     print(f"**Result** 👍: The review is {result[0]}.")

# else:
#     print(f"**Result** 👎: The review is {result[0]}")

def load_artifact(file):
    path_artifact = r"saved_models/"
    artifact = joblib.load(os.path.join(path_artifact, file))
    return artifact

# TEXT CLEANING 
def clean_text(text):
    text = text.replace('<br /><br />','')
    text = text.lower() 
    text = text.strip()  
    text = re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

def remove_stopwords(text):
    stopwords_eng = list(set(stopwords.words("english")))
    text = ' '.join([i for i in text.split() if i not in stopwords_eng])
    return text

def apply_lemmatizer(text):
    lem = WordNetLemmatizer()
    text = ' '.join([lem.lemmatize(word) for word in text.split()])
    return text

def text_preprocessing(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = apply_lemmatizer(text)
    return text
def predict(text, vectorizer, model):
    if isinstance(text, str):
        text = [text]
    
    embedding = vectorizer.transform(text)
    prediction = model.predict(embedding)
    probas = model.predict_proba(embedding)
    
    return prediction, probas


@app.get('/')
def my_function(text:str):
#   pred=predict_result(text)
    review_clean = text_preprocessing(text)
    result, probas = predict(review_clean, tfidf_vectorizer, model)
    print(result[0])
    return JSONResponse({"prediction":result[0]})

if __name__=="__main__":
  uvicorn.run(app,host="0.0.0.0",port=9000)

### Commands to download nltk package

# python -m nltk.downloader stopwords
# python -m nltk.downloader wordnet
