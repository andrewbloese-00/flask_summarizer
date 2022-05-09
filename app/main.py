from copyreg import pickle
from flask import Flask , abort , request, jsonify
from util import summarize_text
import numpy as np 
import requests
import pickle

RESOURCE_URL = "https://github.com/brokegti/vectors/raw/697e925299c0155b6f827938ccc13582c9388624/vectors"


vectors = None
app = Flask(__name__)

@app.route("/")
def hello():
        return "hello"

@app.route("/api/sum", methods=["POST"])
def do_summary():
    text=request.json.get("text")
    reduceTo=float(request.json.get("reduceTo"))
    if text is None or reduceTo is None: 
        return ""    
    return summarize_text( text , reduce_to=reduceTo, vectors=vectors)




if __name__ == "__main__":
    if vectors is None: 
        print("Downloading vector text . . . ")
        response = requests.get(RESOURCE_URL)
        vectors = pickle.loads(response.content)
            

        

    print("App ready!")
    app.run()   
