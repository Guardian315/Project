"""
This module provides emotion detection functionality.

The functions in this module include:
detection_emotion function.
helper function to validate text.

Usage:
make sure to import the following libraries.

Note: This module uses nltk & sklearn instead of watson.
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from flask import Flask, request, jsonify

app = Flask(__name__)

def detect_emotion(text):
    '''
    gets the emotion that represents the text.

    Parameters:
    - text.

    Returns:
    - str: the emotion.
    '''
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token)
                       for token in tokens if token.isalpha() and token not in stop_words]
    preprocessed_text = ' '.join(filtered_tokens)

    vectorizer = TfidfVectorizer()
    model = LinearSVC(dual=True)
    x_train = vectorizer.fit_transform(["I am happy", "I am sad", "I am angry", "I am surprised"])
    y_train = ["happy", "sad", "angry", "surprised"]
    model.fit(x_train, y_train)

    x_test = vectorizer.transform([preprocessed_text])
    predicted_emotion = model.predict(x_test)[0]

    return predicted_emotion

def validate_text(text):
    '''
    validates the test if empty or not.

    Parameters:
    - text.

    Returns:
    - bool: whether empty or not.
    '''
    return not text == ''


@app.route('/detect_emotion', methods=['POST'])
def get_emotion():
    '''
    the hosting function for the detection app.

    Parameters:
    - the route.
    - the method.

    Returns:
    - json: of the emotion.
    - int: status code.
    '''
    data = request.get_json()
    text = data['text']
    if not validate_text(text):
        return jsonify({'error': 'Text is empty'}), 400

    emotion = detect_emotion(text)
    return jsonify({'emotion': emotion}), 200

if __name__ == '__main__':
    app.run(debug=True)
