from flask import Flask,request,jsonify,url_for,render_template,Markup,redirect
import requests
import pandas as pd
import numpy as np
import pickle
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import torch
from PIL import Image
from utils.model import ResNet9
from torchvision import transforms
import io
import random



crop_model1=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\Crop_models\Decision_Tree.pkl','rb'))
crop_model2=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\Crop_models\Gaussian_Naive_bayes.pkl','rb'))
crop_model3=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\Crop_models\KNN.pkl','rb'))
crop_model4=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\Crop_models\Random_Forest.pkl','rb'))
crop_model5=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\Crop_models\SVM.pkl','rb'))
cropmodel=[crop_model1,crop_model2,crop_model3,crop_model4,crop_model5]

fert_model1=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\Fert_models\Decesion_Tree.pkl','rb'))
fert_model2=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\Fert_models\Gaussian_Naive_bayes.pkl','rb'))
fert_model3=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\Fert_models\Random_Forest.pkl','rb'))
fert_model4=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\Fert_models\SVM.pkl','rb'))
fertmodel=[fert_model1,fert_model2,fert_model3,fert_model3]
crop_label=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\crop_label.pkl','rb'))
soil_label=pickle.load(open('D:\Project\Crop_Project\Farmer AI Friend\models\soil_label.pkl','rb'))


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = "D:\Project\Crop_Project\Farmer AI Friend\models\plant_disease_model.pth"
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


def weather_fetch(city_name):
    api_key='c6f5279cd3ede75a9bb20ecb582205fd'
    base_url='http://api.weatherstack.com/current'
    com_url=base_url+'?access_key='+api_key+'&query='+city_name
    response = requests.get(com_url)
    x=response.json()
    y=x['current']
    temperature=y['temperature']
    humidity=y['humidity']
    # print(temperature,' ',humidity)
    return temperature,humidity

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop_reco')
def crop():
    return render_template('crop_reco.html')
@app.route('/crop_recommendation',methods=['POST'])
def crop_r():
    if(request.method=='POST'):
        city=str(request.form['city'])
        nitrogen=int(request.form['nitrogen'])
        phosphorus=int(request.form['phosphorus'])
        pottasium=int(request.form['pottasium'])
        rainfall=float(request.form['rainfall'])
        ph=int(request.form['ph'])
        temperature,humidity=weather_fetch(city)
        data=np.array([[nitrogen,phosphorus,pottasium,temperature,humidity,ph,rainfall]])
        output=[]
        for i in cropmodel:
            p=i.predict(data)
            output.append(p[0])
        output=' or '.join(set(output))
    return render_template('crop_result.html', prediction=output)



@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')
@app.route('/fertilizer_recommendation', methods=['POST'])
def fert_recommendation():
    crop_type=str(request.form['crop'])
    soil_type=str(request.form['soil'])
    city=str(request.form['city'])
    nitrogen=int(request.form['nitrogen'])
    phosphorus=int(request.form['phosphorus'])
    pottasium=int(request.form['pottasium'])
    temperature,humidity=weather_fetch(city)
    moisture=humidity-random.randint(0,30)
    crop_type=crop_label.transform([crop_type])
    soil_type=soil_label.transform([soil_type])
    data=np.array([[temperature,moisture,soil_type[0],crop_type[0],nitrogen,pottasium,phosphorus]])
    output=[]
    for i in fertmodel:
        p=i.predict(data)
        output.append(p[0])
    output=Markup(str(fertilizer_dic[output[0]]))
    return render_template('fertilizer_result.html',result=output)

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease_result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

if(__name__)=="__main__":
    app.run(debug=True) 