import time
from concurrent.futures import ThreadPoolExecutor
import json
import os
from multiprocessing.managers import BaseManager
from flask import Flask, request, jsonify, make_response, Response
from flask_cors import CORS
import uuid
from concurrent.futures import as_completed
from werkzeug.utils import secure_filename

# some imports above may not be needed

app = Flask(__name__)
app.response_buffering = False
CORS(app)

# initialize manager connection
# TODO: you might want to handle the password in a less hardcoded way 
manager = BaseManager(("", 5602), b"password")
manager.register("initialize_index")
manager.register("query_index")

# Try to connect
for _ in range(10):
    try:
        manager.connect()
        break
    except ConnectionRefusedError:
        print("Connecting to index server has failed, waiting before retrying...")
        time.sleep(3)

executor = ThreadPoolExecutor()
    
@app.route("/query", methods=["GET"])
def query_index():
    global manager
    query_text = request.args.get("text", None)
    query_doc_id = request.args.get("doc_id", None)
    
    if query_text is None:
        return "No text found, please include a ?text=blah parameter in the URL", 400
    
    manager.initialize_index() 
    # we aren't make a unique index for each user, so not sure we need UUID
    response = manager.query_index(query_text, query_doc_id)._getvalue()
    response_json = {
        "text": str(response),
    }
    return make_response(jsonify(response_json)), 200



















