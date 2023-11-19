from ml_service import MLService
from predict import load_model, load_scaler

# Charger les modèles et les scalers
model_S = load_model('chemin/vers/model_S.joblib')
model_L = load_model('chemin/vers/model_L.joblib')
scaler_S = load_scaler('chemin/vers/scaler_S.joblib')
scaler_L = load_scaler('chemin/vers/scaler_L.joblib')

# Créer une instance du service et sauvegarder
ml_service = MLService()
ml_service.pack('model_S', model_S)
ml_service.pack('model_L', model_L)
ml_service.pack('scaler_S', scaler_S)
ml_service.pack('scaler_L', scaler_L)
saved_path = ml_service.save()