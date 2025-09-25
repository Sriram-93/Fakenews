from flask import Flask, request, render_template
import numpy as np
import pickle
app = Flask(__name__)
# Load the model and vectorizer
model = pickle.load(open('fake_news_tree_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html', title="About")

@app.route('/login')
def login():
    return render_template('login.html', title="Login")

@app.route('/signin')
def signup():
    return render_template('signup.html', title="Signin")

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input text from the HTML form
    input_text = request.form.get("news_text", "")  # Expecting a form field with name 'news_text'
    print("Input Text:", input_text)
    
    # Transform the input text using the TF-IDF vectorizer
    final_input = tfidf_vectorizer.transform([input_text]).toarray()
    
    # Predict the probability of the news being fake
    prediction = model.predict_proba(final_input)
    fake_prob = prediction[0][1]  # Probability of being 'fake'
    
    # Format the output with two decimal places
    output = f"{fake_prob:.2f}"
    
    # Determine if the news is likely fake or real based on the probability
    if fake_prob > 0.5:
        return render_template(
            'index.html', 
            pred=f'This news is likely fake. Probability of being fake: {output}',
            action="Consider verifying the source."
        )
    else:
        return render_template(
            'index.html', 
            pred=f'This news is likely real. Probability of being fake: {output}',
            action="The news seems credible."
        )

if __name__ == "__main__":
    app.run(debug=True)
