#Day 2
import pandas as pd

print("Program started")

data = pd.read_csv("data/spam.csv", encoding="latin-1")

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

print(data.head())
print("Rows and columns:", data.shape)

#Day 3
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text messages into numerical form
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(data['message'])
y = data['label']

print("Text converted into numbers")
print("X shape:", X.shape)
