import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.managers import BaseManager
from flask import Flask, request, jsonify, make_response, Response
from flask_cors import CORS
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
manager.register("query_cases")
manager.register("query_regs")

# Try to connect
# NOTE: we need a more graceful load bar while we wait for the indexes to be built because
# until they're built, we'll just keep getting 'Connecting to index server has failed' 
for _ in range(10):
    try:
        manager.connect()
        break
    except ConnectionRefusedError:
        print("Connecting to index server has failed, waiting before retrying...")
        time.sleep(5)
else:  # This will run if the for loop completes without a break
    print("Failed to connect after all retries. Exiting.")
    exit(1)

executor = ThreadPoolExecutor()
    
@app.route("/query_cases", methods=["GET"])
def query_cases():
    global manager
    query_text = request.args.get("text", None)

    if query_text is None:
        return "No text found, please include a ?text=blah parameter in the URL", 400
    
    response = manager.query_cases(query_text)._getvalue()
    response_json = {
        "text": str(response),
    }
    return make_response(jsonify(response_json)), 200

@app.route("/query_regs", methods=["GET"])
def query_regs():
    global manager
    query_text = request.args.get("text", None)

    if query_text is None:
        return "No text found, please include a ?text=blah parameter in the URL", 400
    
    response = manager.query_regs(query_text)._getvalue()
    response_json = {
        "text": str(response),
    }
    return make_response(jsonify(response_json)), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)


















