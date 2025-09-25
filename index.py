import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("fake_news_data.csv",nrows=40000)  # Replace with your dataset path

# Assume the dataset has 'text' and 'class'  columns
# 'text' contains the news content, and 'class' contains 0 (real) and 1 (fake)
X = data['text']
y = data['class'].astype('int') 

# Convert text data into numerical format using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf_vectorizer.fit_transform(X).toarray()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the Decision Tree classifier
tree_clf = DecisionTreeClassifier(random_state=0)
tree_clf.fit(X_train, y_train)

# Save model and TF-IDF vectorizer with pickle
pickle.dump(tree_clf, open('fake_news_tree_model.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

# Load model and vectorizer to check the prediction
model = pickle.load(open('fake_news_tree_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Test model accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
