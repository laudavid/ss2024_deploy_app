import re 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


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
    return [text]


    

