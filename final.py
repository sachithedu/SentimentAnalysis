import pandas as pd
import numpy as np
import re
import string  
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack 


# Check for empty or missing values
def preprocess_text(text):
    if not text or pd.isna(text):  
        return ""
    
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text

def label_sentiment(score):
    if score >= 4:
        return "positive"
    elif score <= 2:
        return "negative"
    else:
        return "neutral"

#load the dataset here
data = pd.read_csv("tiktok_user_reviews.csv", dtype={'reviewID': 'str', 'userName': 'str', 'content': 'str','score':'int', 'thumbsUpCount': 'int'})

#preprocess text and label sentiment
data["cleaned_content"] = data["content"].apply(preprocess_text)
data["sentiment"] = data["score"].apply(label_sentiment)

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[["cleaned_content", "thumbsUpCount"]], data["sentiment"], test_size=0.2, random_state=42)

#feature extraction using Bag of Words (BoW)
vectorizer = CountVectorizer(stop_words='english')
X_train_bow = vectorizer.fit_transform(X_train["cleaned_content"])
X_test_bow = vectorizer.transform(X_test["cleaned_content"])

#feature extraction using TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_bow)
X_test_tfidf = tfidf_transformer.transform(X_test_bow)

#scale 'thumbsUpCount' feature
scaler = MinMaxScaler()
X_train_thumbs_up = scaler.fit_transform(X_train[["thumbsUpCount"]])
X_test_thumbs_up = scaler.transform(X_test[["thumbsUpCount"]])

#combine TF-IDF matrix with scaled 'thumbsUpCount' feature 
X_train_combined = hstack([X_train_tfidf, X_train_thumbs_up])
X_test_combined = hstack([X_test_tfidf, X_test_thumbs_up])

#train model using log. regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_combined, y_train)

#predict model
y_pred = logreg.predict(X_test_combined)

#evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["positive", "negative", "neutral"]))