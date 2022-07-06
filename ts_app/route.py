from flask import Blueprint,  render_template, request, jsonify
from .model import model, tokenizer
from .process import summarize_replies

main = Blueprint('main', __name__)

# Homepage handling
@main.route("/")
def homepage():
    # get input from HTML form
    return render_template('home.html')

@main.route("/summary", methods=['POST'])
def summary():
    if 'url' in request.form:
        url = request.form['url']
        if url:
            # get the summary
            summary = summarize_replies(tokenizer, model, url)
            return jsonify(message=f'{summary}')
    return '', 400