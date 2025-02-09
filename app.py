from flask import FLask,request,render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

# cat_onehot_features = ['area_type']
        # cat_ordinal_features = ['size']
        # num_features = ['bath', 'balcony', 'total_sqft']
@app.route("/predictdata",methods=["GET","POST"])
def predictdata():
    if request.method=="GET":
        return render_template("index.html")
    else:
        data=CustomData(
            area_type=request.form.get('area_type'),
            size=request.form.get('size'),
            bath=request.form.get('bath'),
            balcony=request.form.get('balcony'),
            total_sqft=request.form.get('total_sqft')

            input=data.get_data_as_data_frame()
            print(input)


            predict_pipeline=PredictPipeline()
            results=predict_pipeline.predict(input)

            return render_template("index.html",results=results[0])








        )