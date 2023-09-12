"""
This module provides emotion detection functionality using client-server approach.

The functions in this module include:
helper function to validate text.
flask hosting function.

Usage:
make sure to import the following libraries.

Note: This module uses nltk & sklearn instead of watson.
"""
from flask import Flask, request, jsonify
from detection import detect_emotion

app = Flask(__name__)

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
