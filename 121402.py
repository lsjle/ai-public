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