from flask import Flask , abort , request, jsonify

from util import summarize_text


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
    return summarize_text( text , reduce_to=reduceTo)

if __name__ == "__main__":
    app.run()
