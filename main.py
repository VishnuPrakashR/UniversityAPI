#  Copyright (c) 2023. This is the property of Vishnu Prakash

from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import socket

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://mongo:27017/dev"
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
mongo = PyMongo(app)
db = mongo.db
app.config.update(
    SECRET_KEY=b'\xa8G\x1c\x84@EQ\xdd\xa2\xf8\xe2\xed\x9e\x9ft\x8f'
)
CORS(app, expose_headers="content-disposition", supports_credentials=True)


@app.route("/")
def hello():
    return "<p>Hello, Everyone! This is the api gateway of my flask api gateway. Enjoy :)</p>"


def run():
    app.run(host="0.0.0.0", port=5000, debug=True, load_dotenv='development')


if __name__ == "__main__":
    run()
