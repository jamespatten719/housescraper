#author = "James <jamespatten1996@gmail.com>"

import Flask
from flask import Flask, flash, redirect, url_for, render_template, request, session, abort
import os
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug import secure_filename
from pipeline import geocodes 

app = Flask(__name__)
app.config.from_object(__name__)

@app.route('/')
@app.route("/index", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return render_template('map.html')
    else:
        return render_template('map.html', geocodes = geocodes)
	
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=4000)
	
