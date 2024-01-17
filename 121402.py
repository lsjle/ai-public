# /*
#  * ----------------------------------------------------------------------------
#  * "THE TEA-WARE LICENSE" (Revision 42-1):
#  * <210114@gs.hs.ntnu.edu.tw> wrote this file.  As long as you retain this notice you
#  * can do whatever you want with this stuff. If we meet some day, and you think
#  * this stuff is worth it, you can buy me a tea (tea or beer both fine) in return.   Lapsang Souchong
#  * Please notice that some part of the code might not be written by me.
#  * File: 121402.py
#  * ----------------------------------------------------------------------------
#  */
#

from flask import Flask,request,render_template
import joblib
import pathlib

app=Flask(__name__,template_folder=pathlib.Path().resolve())
@app.route("/")
def hello():
    return "Hello, World!<a href='mainpage'>fill it</a>"
@app.route("/proceed", methods=['GET', 'POST'])
def proceed():
    if(request.method=='GET'):
        return 'f'
    else:
        model=joblib.load("2023121401.pkl")
        inputdata=[int(request.values['gender']),int(request.values['married']),int(request.values['dependents']),int(request.values['education']),int(request.values['selfemp']),int(request.values['pa'])]
        return (str(model.predict([inputdata]))+str(model.predict_proba([inputdata])))
                #+str(max(model.predict_proba([inputdata]))))
@app.route("/mainpage")
def mainpage():
    return render_template("121403.html")
if __name__ == '__main__':
    app.debug = True
    app.run()  