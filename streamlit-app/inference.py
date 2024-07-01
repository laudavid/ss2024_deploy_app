from preprocess import text_preprocessing

def predict(text, vectorizer, model):
    # Text cleaning/pre-processing 
    cleaned_text = text_preprocessing(text)
    embedding = vectorizer.transform(cleaned_text)
    
    # Polarity prediction with Logistic Regression
    prediction = model.predict(embedding)
    probas = model.predict_proba(embedding)
    
    return prediction, probas
    

