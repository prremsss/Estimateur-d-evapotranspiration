import numpy as np
from flask import Flask, jsonify, render_template, request
import pickle
import os
import pandas as pd
flask_app = Flask(__name__)
flask_app.config["UPLOAD_FOLDER"]="static/excel"
#4variables
_4var = pickle.load(open("models/4v/4variables", "rb"))
#2variables
HumiditeVitesse=pickle.load(open("models/2v/Humidité Vitesse du vent daily", "rb"))
RayonnementHumidite=pickle.load(open("models/2v/Rayonnement Humidité daily", "rb"))
RayonnementVitesseduvent =pickle.load(open("models/2v/Rayonnement Vitesse du vent daily", "rb"))
TemperatureHumidite=pickle.load(open("models/2v/Température Humidité daily", "rb"))
TemperatureVitesseduvent=pickle.load(open("models/2v/Température Vitesse du vent daily", "rb"))
TemperatureRayonnement=pickle.load(open("models/2v/Température Rayonnement daily", "rb"))
#3variables
RayonnementHumiditéVitesseduvent=pickle.load(open("models/3v/Rayonnement ,Humidité ,Vitesse du vent", "rb"))
TempératureRayonnementyHumidité=pickle.load(open("models/3v/Température ,Rayonnementy,Humidité", "rb"))
TempératureHumiditéVitesseduvent=pickle.load(open("models/3v/Température, Humidité,Vitesse du vent", "rb"))
TempératureRayonnementVitesseduvent=pickle.load(open("models/3v/Température, Rayonnement,Vitesse du vent", "rb"))
#1variable
Humidité=pickle.load(open("models/1v/Humidité daily", "rb"))
Rayonnement=pickle.load(open("models/1v/Rayonnement daily", "rb"))
Température=pickle.load(open("models/1v/Température daily", "rb"))
Vitesseduvent=pickle.load(open("models/1v/Vitesse du vent daily", "rb"))

#horaire

hr_4v=pickle.load(open("models/Hourly/4 variables hr", "rb"))

#3variables
RayonnementHumiditéVitesseduventhr=pickle.load(open("models/Hourly/3v/Rf 3 variables hr Radiation Solaire Vitesse du vent & Humidité", "rb"))
TempératureRayonnementyHumiditéhr=pickle.load(open("models/Hourly/3v/Rf 3 variables hr Radiation Solaire & Humidité & Température", "rb"))
TempératureHumiditéVitesseduventhr=pickle.load(open("models/Hourly/3v/Rf 3 variables hr Vitesse du vent & Humidité & Température", "rb"))
TempératureRayonnementVitesseduventhr=pickle.load(open("models/Hourly/3v/Rf 3 variables hr Radiation Solaire & Vitesse du vent & Température", "rb"))

#2variables
HumiditeVitessehr=pickle.load(open("models/Hourly/2v/Rf 2 variables hr Vitesse du vent & Humidité", "rb"))
RayonnementHumiditehr=pickle.load(open("models/Hourly/2v/Rf 2 variables hr Radiation Solaire & Humidité", "rb"))
RayonnementVitesseduventhr =pickle.load(open("models/Hourly/2v/Rf 2 variables hr Radiation Solaire & Vitesse du vent", "rb"))
TemperatureHumiditehr=pickle.load(open("models/Hourly/2v/Rf 2 variables hr Humidité & Température", "rb"))
TemperatureVitesseduventhr=pickle.load(open("models/Hourly/2v/Rf 2 variables hr Vitesse du vent & Température min", "rb"))
TemperatureRayonnementhr=pickle.load(open("models/Hourly/2v/Rf 2 variables hr Radiation Solaire & Température", "rb"))
#1variable
Humiditéhr=pickle.load(open("models/Hourly/1v/Rf 1 variable hr Humidité", "rb"))
Rayonnementhr=pickle.load(open("models/Hourly/1v/xgb 1 variable hr Radiation Solaire", "rb"))
Températurehr=pickle.load(open("models/Hourly/1v/Rf 1 variable hr Température", "rb"))
Vitesseduventhr=pickle.load(open("models/Hourly/1v/Rf 1 variable hr Vitesse du vent", "rb"))
@flask_app.route("/")
def Home():
    return render_template('projet.html')


@flask_app.route("/predict", methods=["POST"])
def predict():
    def features(float_features):
        ft = list(map(float, float_features))
        fts = np.array(ft)
        fts = fts.reshape(1, -1)
        return fts
    get=request.form.get

    var = request.form.getlist('check')
    vars = list(map(int, var))

    if (vars == [1]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin')]
        prediction=Température.predict(features(float_features))
    elif (vars == [2]):
        float_features = [get('ray')]
        prediction=Rayonnement.predict(features(float_features))

    elif (vars == [3]):
        float_features = [get('hummoy'),get('hummax'),get('hummin')]
        prediction=Humidité.predict(features(float_features))

    elif (vars == [4]):
        float_features = [get('vitmoy'),get('vitmax')]
        prediction=Vitesseduvent.predict(features(float_features))

    elif (vars == [1, 2]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('ray')]
        prediction=TemperatureRayonnement.predict(features(float_features))

    elif (vars == [1, 3]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('hummoy'),get('hummax'),get('hummin')]
        prediction=TemperatureHumidite.predict(features(float_features))

    elif (vars == [1, 4]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('vitmoy'),get('vitmax')]
        prediction=TemperatureVitesseduvent.predict(features(float_features))

    elif (vars == [2, 3]):
        float_features = [get('ray'),get('hummoy'),get('hummax'),get('hummin')]
        prediction=RayonnementHumidite.predict(features(float_features))

    elif (vars == [2, 4]):
        float_features = [get('ray'),get('vitmoy'),get('vitmax')]
        prediction=RayonnementVitesseduvent.predict(features(float_features))

    elif (vars == [3, 4]):
        float_features = [get('hummoy'),get('hummax'),get('hummin'),get('vitmoy'),get('vitmax')]
        prediction=HumiditeVitesse.predict(features(float_features))

    elif (vars == [1,2,3]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('ray'),get('hummoy'),get('hummax'),get('hummin')]
        prediction=TempératureRayonnementyHumidité.predict(features(float_features))

    elif (vars == [1,2,4]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('ray'),get('vitmoy'),get('vitmax')]
        prediction=TempératureRayonnementVitesseduvent.predict(features(float_features))

    elif (vars == [2,3,4]):
        float_features = [get('ray'),get('hummoy'),get('hummax'),get('hummin'),get('vitmoy'),get('vitmax')]
        prediction=RayonnementHumiditéVitesseduvent.predict(features(float_features))

    elif (vars == [1,3,4]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('hummoy'),get('hummax'),get('hummin'),get('vitmoy'),get('vitmax')]
        prediction=TempératureHumiditéVitesseduvent.predict(features(float_features))


    elif (vars == [1,2,3,4]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('ray'),get('hummoy'),get('hummax'),get('hummin'),get('vitmoy'),get('vitmax')]
        prediction=_4var.predict(features(float_features))


    print(prediction[0])

    return render_template("projet.html", prediction_text="Et0= {}".format(prediction[0]))


@flask_app.route("/predict2", methods=["POST"])
def predict2():
    def features(float_features):
        ft = list(map(float, float_features))
        fts = np.array(ft)
        fts = fts.reshape(1, -1)
        return fts
    get=request.form.get

    var = request.form.getlist('check')
    vars = list(map(int, var))

    if (vars == [1]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin')]
        prediction=Températurehr.predict(features(float_features))
    elif (vars == [2]):
        float_features = [get('ray')]
        prediction=Rayonnementhr.predict(features(float_features))

    elif (vars == [3]):
        float_features = [get('hummoy'),get('hummax'),get('hummin')]
        prediction=Humiditéhr.predict(features(float_features))

    elif (vars == [4]):
        float_features = [get('vitmoy'),get('vitmax')]
        prediction=Vitesseduventhr.predict(features(float_features))

    elif (vars == [1, 2]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('ray')]
        prediction=TemperatureRayonnementhr.predict(features(float_features))

    elif (vars == [1, 3]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('hummoy'),get('hummax'),get('hummin')]
        prediction=TemperatureHumiditehr.predict(features(float_features))

    elif (vars == [1, 4]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('vitmoy'),get('vitmax')]
        prediction=TemperatureVitesseduventhr.predict(features(float_features))

    elif (vars == [2, 3]):
        float_features = [get('ray'),get('hummoy'),get('hummax'),get('hummin')]
        prediction=RayonnementHumiditehr.predict(features(float_features))

    elif (vars == [2, 4]):
        float_features = [get('ray'),get('vitmoy'),get('vitmax')]
        prediction=RayonnementVitesseduventhr.predict(features(float_features))

    elif (vars == [3, 4]):
        float_features = [get('hummoy'),get('hummax'),get('hummin'),get('vitmoy'),get('vitmax')]
        prediction=HumiditeVitessehr.predict(features(float_features))

    elif (vars == [1,2,3]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('ray'),get('hummoy'),get('hummax'),get('hummin')]
        prediction=TempératureRayonnementyHumiditéhr.predict(features(float_features))

    elif (vars == [1,2,4]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('ray'),get('vitmoy'),get('vitmax')]
        prediction=TempératureRayonnementVitesseduventhr.predict(features(float_features))

    elif (vars == [2,3,4]):
        float_features = [get('ray'),get('hummoy'),get('hummax'),get('hummin'),get('vitmoy'),get('vitmax')]
        prediction=RayonnementHumiditéVitesseduventhr.predict(features(float_features))

    elif (vars == [1,3,4]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('hummoy'),get('hummax'),get('hummin'),get('vitmoy'),get('vitmax')]
        prediction=TempératureHumiditéVitesseduventhr.predict(features(float_features))


    elif (vars == [1,2,3,4]):
        float_features = [get('tempmoy'),get('tempmax'),get('tempmin'),get('ray'),get('hummoy'),get('hummax'),get('hummin'),get('vitmoy'),get('vitmax')]
        prediction=hr_4v.predict(features(float_features))

    print(vars)
    return render_template("projet.html", prediction_text2="Et0= {}".format(prediction[0]))


@flask_app.route("/predict3", methods=["POST"])
def predict3():
        get = request.form.get
        var = request.form.getlist('check')
        vars = list(map(int, var))
        f = request.files['file_upload']
        data_xls = pd.read_excel(f)


        if (vars == [1]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            prediction = Température.predict(data)
        elif (vars == [2]):
            data = pd.DataFrame()
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            prediction = Rayonnement.predict(data)

        elif (vars == [3]):
            data = pd.DataFrame()
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            prediction = Humidité.predict(data)
        elif (vars == [4]):
            data = pd.DataFrame()
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = Vitesseduvent.predict(data)

        elif (vars == [1, 2]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            prediction = TemperatureRayonnement.predict(data)

        elif (vars == [1, 3]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            prediction = TemperatureHumidite.predict(data)

        elif (vars == [1, 4]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = TemperatureVitesseduvent.predict(data)

        elif (vars == [2, 3]):
            data = pd.DataFrame()
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            prediction = RayonnementHumidite.predict(data)

        elif (vars == [2, 4]):
            data = pd.DataFrame()
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']

            prediction = RayonnementVitesseduvent.predict(data)

        elif (vars == [3, 4]):
            data = pd.DataFrame()
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = HumiditeVitesse.predict(data)

        elif (vars == [1, 2, 3]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            prediction = TempératureRayonnementyHumidité.predict(data)

        elif (vars == [1, 2, 4]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = TempératureRayonnementVitesseduvent.predict(data)

        elif (vars == [2, 3, 4]):
            data = pd.DataFrame()
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = RayonnementHumiditéVitesseduvent.predict(data)

        elif (vars == [1, 3, 4]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = TempératureHumiditéVitesseduvent.predict(data)


        elif (vars == [1, 2, 3, 4]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = _4var.predict(data)

        prediction=prediction.tolist()
        print(prediction)
        print(type(prediction))
        labels=data_xls['Date/heure'].tolist()
        return render_template("projet.html",labels=labels,  values=prediction)
@flask_app.route("/predict4", methods=["POST"])
def predict4():
        get = request.form.get
        var = request.form.getlist('check')
        vars = list(map(int, var))
        f = request.files['file_upload']
        data_xls = pd.read_excel(f)


        if (vars == [1]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            prediction = Températurehr.predict(data)
        elif (vars == [2]):
            data = pd.DataFrame()
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            prediction = Rayonnementhr.predict(data)

        elif (vars == [3]):
            data = pd.DataFrame()
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            prediction = Humiditéhr.predict(data)
        elif (vars == [4]):
            data = pd.DataFrame()
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = Vitesseduventhr.predict(data)

        elif (vars == [1, 2]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            prediction = TemperatureRayonnementhr.predict(data)

        elif (vars == [1, 3]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            prediction = TemperatureHumiditehr.predict(data)

        elif (vars == [1, 4]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = TemperatureVitesseduventhr.predict(data)

        elif (vars == [2, 3]):
            data = pd.DataFrame()
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            prediction = RayonnementHumiditehr.predict(data)

        elif (vars == [2, 4]):
            data = pd.DataFrame()
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']

            prediction = RayonnementVitesseduventhr.predict(data)

        elif (vars == [3, 4]):
            data = pd.DataFrame()
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = HumiditeVitessehr.predict(data)

        elif (vars == [1, 2, 3]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            prediction = TempératureRayonnementyHumiditéhr.predict(data)

        elif (vars == [1, 2, 4]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = TempératureRayonnementVitesseduventhr.predict(data)

        elif (vars == [2, 3, 4]):
            data = pd.DataFrame()
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = RayonnementHumiditéVitesseduventhr.predict(data)

        elif (vars == [1, 3, 4]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']

            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = TempératureHumiditéVitesseduventhr.predict(data)


        elif (vars == [1, 2, 3, 4]):
            data = pd.DataFrame()
            data['Température moy'] = data_xls['Température moy']
            data['Température max'] = data_xls['Température max']
            data['Température min'] = data_xls['Température min']
            data['Rayonnement solaire moy'] = data_xls['Rayonnement solaire moy']
            data['Humidité moy'] = data_xls['Humidité moy']
            data['Humidité max'] = data_xls['Humidité max']
            data['Humidité min'] = data_xls['Humidité min']
            data['Vitesse du vent moy'] = data_xls['Vitesse du vent moy']
            data['Vitesse du vent max'] = data_xls['Vitesse du vent max']
            prediction = hr_4v.predict(data)

        prediction2=prediction.tolist()
        print(prediction)
        print(type(prediction))
        labels2=data_xls['Date/heure'].tolist()
        return render_template("projet.html",labels2=labels2,  values2=prediction2)


if __name__ == "__main__":
    flask_app.run(debug=True)
