from urllib import request
from flask import Flask ,render_template, request

import model as m

app= Flask(__name__)
ap = ""

@app.route("/", methods = ["GET","POST"])
def Fun_knn():            
    return render_template("index.html")


@app.route("/sub", methods = ["GET","POST"])
def submit():
    global ap
    if request.method == "POST":
        mod = request.form['mod']
        acc_pred  = m.knnModel()
        ap= acc_pred
        
    return render_template("sub.html", mod_acc= ap) 

if __name__=="__main__":
    app.run(debug=True)     