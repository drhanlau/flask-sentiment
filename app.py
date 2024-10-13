from flask import Flask, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon if you haven't already
nltk.download("vader_lexicon")

# Initialize the Flask app
app = Flask(__name__)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


@app.route("/sentiment", methods=["POST"])
def sentiment_analysis():
    # Get the input string from the POST request's JSON payload
    data = request.json
    input_text = data.get("text", "")

    # Perform sentiment analysis
    sentiment_scores = analyzer.polarity_scores(input_text)

    # Return the result as a JSON object
    return jsonify(sentiment_scores)


if __name__ == "__main__":
    app.run(debug=True)
