import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load and preprocess the dataset
def load_data():
    df = pd.read_csv('data/emails_dataset.csv')
    # Additional datasets can be merged here for data augmentation
    # Example: df = pd.concat([df, pd.read_csv('data/additional_emails.csv')])
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(df['email'])
    y = df['label']
    return X, y, vectorizer

# Train the Naive Bayes model
def train_model():
    X, y, vectorizer = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Save the model and vectorizer
    joblib.dump(model, 'model/spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    
    return model, vectorizer

# Predict if the email is spam or ham
def predict_email(model, vectorizer, email):
    email_transformed = vectorizer.transform([email])
    prediction = model.predict(email_transformed)
    return prediction[0]

# Only train the model if this script is run directly
if __name__ == "__main__":
    model, vectorizer = train_model()
