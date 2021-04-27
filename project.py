from flask import Flask,redirect,url_for,render_template,request,session,flash
import os
from second import second
from paragraph import paragraph
from wikisearch import wikisearch
from pdfsearch import pdfsearch

app = Flask(__name__)
app.register_blueprint(second,url_prefix='/')
app.register_blueprint(paragraph,url_prefix='/')
app.register_blueprint(wikisearch,url_prefix='/')
app.register_blueprint(pdfsearch,url_prefix='/')

UPLOAD_FOLDER = '/home/subhashis/Desktop/EE390/newproject/Parisodhan_Website/UPLOAD_FOLDER'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

@app.route("/index")
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
   app.run(debug=True)  #This is to immediately update changes done in html pages#