#  Copyright (c) 2023. This is the property of Vishnu Prakash

from flask import Flask, request
from flask_cors import CORS
from db import Mongo
from format import JSONEncoder as jsone
import requests

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
api = Mongo('api')
app.config.update(
    SECRET_KEY=b'\xa8G\x1c\x84@EQ\xdd\xa2\xf8\xe2\xed\x9e\x9ft\x8f'
)
CORS(app, expose_headers="content-disposition", supports_credentials=True)


@app.route("/")
def hello():
    return "<h1>Welcome to API gateway!</h1><p>This is a university App made as part of Software Engineering in Service Computing Module</p>"


@app.route('/<service>/<path:path>', methods=['GET', 'POST'])
def processAPI(service, path):
    data = api.getaftercount({"Status": 1, "Service": service}, "CallCount")
    auth_header = request.headers.get('Authorization')
    headers = {'X-API-Key': data.get('Key'), 'Referer': 'Gateway', 'Authorization': auth_header}
    url = f'{data.get("Url")}/{path}'
    response = requests.request(request.method, url, headers=headers, data=request.form)
    return response.text, response.status_code


def run():
    app.run(host="0.0.0.0", port=5001, debug=True, load_dotenv='development')


if __name__ == "__main__":
    run()
