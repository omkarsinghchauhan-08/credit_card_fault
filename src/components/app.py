from flask import Flask
from urllib.parse import quote as url_quote


app = Flask(__name__)

@app.route("/")
def helloworld():
    return "<h1>Hello World!</h1>"

if __name__=="__main__":
    app.run(host="0.0.0.0")