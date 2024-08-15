from flask import Flask, request, render_template
import joblib
from modelmodi import predict_email

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model/spam_classifier_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email = request.form['email'] #storing the entered content in the website's form
        prediction = predict_email(model, vectorizer, email) #using the model to predict the given mail
        return render_template('index.html', prediction=prediction, email=email) 
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
