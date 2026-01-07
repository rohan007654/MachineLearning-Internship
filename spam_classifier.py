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

#Day 4
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training completed")

#Day 5
from sklearn.metrics import accuracy_score, confusion_matrix

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test with your own message
sample_message = ["Congratulations! You have won a free prize"]
sample_vector = vectorizer.transform(sample_message)
prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: NOT SPAM")
