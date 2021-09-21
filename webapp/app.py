import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model.
with open(f'model/movie_reviews_sentiment_analysis.pkl', 'rb') as f:
    model = pickle.load(f)

# Use pickle to load in vectorizer.
with open(f'model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        review = flask.request.form['review']
        predict_text = "Prediction sentiment for movie: "
        movie = flask.request.form.get("movie")
        prediction = model.predict(vectorizer.transform([review]))
        return(flask.render_template('main.html', predict_text=predict_text, movie=movie, result=prediction))

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run()
