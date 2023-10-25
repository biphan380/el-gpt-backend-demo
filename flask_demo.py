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
    
    regs_prompt = """
        You are an expert in employment law in Ontario.
        As such, you know there's the Human Rights Code,
        that governs violations of s. 5 of the Code (which 
        you have in your documents database) and any special
         defences, such as exemptions from the clergy or
        'bona fide occupational requirements.'
        You also know about the Employment Standards Act, 2000,
        which prescribes minimum employment standards. They
        work together to create legislative rights.
        As you answer the following query, 
        remember to caution the reader at the end that
        you are providing legal information, not legal
        advice, and relying on your response is a bad
        idea. It would be great if you added that at the 
        end in its own paragraph with a warning icon in front
        of it. 
        For example, if asked about the minimum wage
        for a bartender, a model response would be:
        
        The current minimum wage is $16.55 per hour,
        including liquor servers. (See the Your Guide to
        the Employment Standards Act.) Although I'm 
        reviewing relevent documents, you should always
        double-check my answer with the Ministry of Labour,
        a lawyer or paralegal, or against the statute and
        regulations themselves. They can be found at: 
        https://canlii.ca/t/55zlc. I hope that helps!
        
        
        Having said that, give it your best guess to
        answer with specific information, with
        dollar values, times, etc. Now answer this query:
        """
    query_text = regs_prompt + query_text
    response = manager.query_regs(query_text)._getvalue()
    response_json = {
        "text": str(response),
    }
    return make_response(jsonify(response_json)), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)


















