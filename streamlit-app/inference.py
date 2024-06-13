# PREDICTION FUNCTION

def predict(text, vectorizer, model):
    if isinstance(text, str):
        text = [text]
    
    embedding = vectorizer.transform(text)
    prediction = model.predict(embedding)
    probas = model.predict_proba(embedding)
    
    return prediction, probas
    

