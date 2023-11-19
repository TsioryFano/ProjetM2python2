import bentoml
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
import pandas as pd
from joblib import load
import pandas as pd

# Charger les modèles et les scalaires
model_S = load('../models/gradient_boosting_S.joblib')
model_L = load('../models/gradient_boosting_L.joblib')
scaler_S = load('../models/scaler_S.joblib')
scaler_L = load('../models/scaler_L.joblib')

# Définir une fonction pour préparer les données et faire des prédictions
def predict(data, model, scaler):
    # Normalisation des données
    data_scaled = scaler.transform(data)
    # Prédiction avec le modèle
    return model.predict(data_scaled)

# Configuration du service BentoML
@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([bentoml.artifact.PickleArtifact('model_S'),
                    bentoml.artifact.PickleArtifact('model_L'),
                    bentoml.artifact.PickleArtifact('scaler_S'),
                    bentoml.artifact.PickleArtifact('scaler_L')])
class MLService(bentoml.BentoService):

    @bentoml.api(input=DataframeInput(), batch=True)
    def predict_S(self, df: pd.DataFrame):
        return predict(df, model_S, scaler_S)
    
    @bentoml.api(input=DataframeInput(), batch=True)
    def predict_L(self, df: pd.DataFrame):
        return predict(df, model_L, scaler_L)
    
# Sauvegarde du service
if __name__ == "__main__":
    service = MLService()
    service.pack('model_S', model_S)
    service.pack('model_L', model_L)
    service.save()
    
