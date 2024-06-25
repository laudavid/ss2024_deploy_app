import pandas as pd

from preprocess import text_preprocessing
from inference import predict
from utils import load_artifact

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum

app=FastAPI()
handler=Mangum(app)

# Load saved models
tfidf_vectorizer = load_artifact("tfidf-vectorizer.sav")
model = load_artifact("logistic_regression.sav")



# review = 'I hated the movie. The story and the actors were terrible.'
review = 'I love the move. The story and the actors were excellent.'
review_clean = text_preprocessing(review)
result, probas = predict(review_clean, tfidf_vectorizer, model)

if result[0] == "positive":
    print(f"**Result** 👍: The review is {result[0]}.")

else:
    print(f"**Result** 👎: The review is {result[0]}")

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
