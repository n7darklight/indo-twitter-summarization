from flask import Flask
#import secrets (alternative)
from .model import model, tokenizer
from .route import main
#from .processor import text, ner_tagged_doc, pos_tagged_doc, stemmed_sentence

# init the flask app
app = Flask(__name__)
#secret = secrets.token_urlsafe(64) (alternative)

#app.secret_key = secret (alternative)

# register the app route
app.register_blueprint(main)