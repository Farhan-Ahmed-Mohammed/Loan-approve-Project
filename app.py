# code for backend
from flask import Flask, request, render_template # here we are using flask for backend we can also use fast api
import pickle
import pandas as pd

app=Flask(__name__)

# load trained pipeline
with open("loan_model.pkl", "rb") as f: # now we are opening the file in which we trained the model
    model=pickle.load(f)

 # every backend server has the routes they are used to preform a particular function, routes are urls of website, when we go to a particular tab we get another word attached to url with a /(slash) that is called route. we need to define different different routes to different different functions. here in below line / means it is the home page og website without anything attached.
@app.route("/") # this is route for home page
def home():
    return render_template("frontend.html") # it means when we open the website without anything attached to / we should be able to frontend.html file every route returns something

# here predict is name of endpoint word attached to / method post means when we are giving some input to backend and other methd is GET it is to get something from backend here we are sending data to backend so POST method
@app.route("/predict",methods=["POST"]) 
def predict():

    # Get form data
    data=request.form.to_dict() # means taking the data from the form we fill on frontend

    # DataFrame for model input
    df=pd.DataFrame([data]) # we are converting this to dataframe bcoz our model takes only dataframe

    # Prediction
    proba=model.predict_proba(df)[0][1]  # this gives probability
    prediction="Loan Approved" if proba>0.6 else "Loan not Approved"

# it means when we click predict we will be redirected to result.html which gives output
    return render_template( 
        "result.html",
        prediction=prediction,
        probability=round(float(proba), 3),
        previous_values=data     # send all the previous values
    )

if __name__=="__main__":
    app.run()



