import pandas as pd
from joblib import load

def load_model(path_to_model):
    """ Charge le modèle entraîné à partir du chemin spécifié. """
    return load(path_to_model)

def load_scaler(scaler_filename):
    """ Charge l'objet de normalisation. """
    return load(scaler_filename)

def prepare_data(data, scaler):
    """ Normalise les données à l'aide du scaler chargé. """
    return scaler.transform(data)

def make_predictions(model, new_data):
    """ Fait des prédictions à l'aide du modèle sur les nouvelles données. """
    return model.predict(new_data)

if __name__ == "__main__":
    # Chemin vers le modèle entraîné
    model_S_path = '../models/gradient_boosting_S.joblib'
    model_L_path = '../models/gradient_boosting_L.joblib'
    scaler_S_path = '../models/scaler_S.joblib'
    scaler_L_path = '../models/scaler_L.joblib'

    #Chargement modèles
    model_S = load_model(model_S_path)
    model_L = load_model(model_L_path)
    scaler_S = load_scaler(scaler_S_path)
    scaler_L = load_scaler(scaler_L_path)

    # Charger ou créer de nouvelles données pour faire des prédictions
    new_data_S = pd.read_csv('../data/new_prediction_data_S.csv')
    new_prepared_data_S = prepare_data(new_data_S, scaler_S)
    new_data_L = pd.read_csv('../data/new_prediction_data_L.csv')
    new_prepared_data_L = prepare_data(new_data_L, scaler_L)

    # Faire des prédictions
    predictions_S = make_predictions(model_S, new_prepared_data_S)
    predictions_L = make_predictions(model_L, new_prepared_data_L)

    # Afficher ou sauvegarder les résultats des prédictions
    print("Prédictions pour Sériation 'S':", predictions_S)
    print("Prédictions pour Sériation 'L':", predictions_L)