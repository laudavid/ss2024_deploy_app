
import os
import json
import joblib 
from preprocess import text_preprocessing

def model_fn(model_dir):
    """
    Load pre-trained models
    """
    model = joblib.load(os.path.join(model_dir, 'logistic_regression.sav'))
    tfidf = joblib.load(os.path.join(model_dir, 'tfidf-vectorizer.sav'))
    model_dict = {"vectorizer":tfidf, "model":model}
    
    return model_dict


def predict_fn(input_data, model):
    """
    Apply text vectorizer and model to the incoming request
    """
    tfidf = model['vectorizer']
    lr_model = model['model']
        
    clean_text = text_preprocessing(input_data)
    embedding = tfidf.transform(clean_text)
    prediction = lr_model.predict(embedding)

    return prediction.tolist()


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request


def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """

    if response_content_type == "application/json":
        response = json.dumps(prediction)
    else:
        response = str(prediction) 

    return response
