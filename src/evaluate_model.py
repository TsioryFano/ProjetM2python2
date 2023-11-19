import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import load

def evaluate_model(model, X_test, y_test, model_name, seriation):
    """ Évalue le modèle de régression et affiche les résultats. """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model {model_name} for Seriation '{seriation}':")    
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

# Affichage graphique des prédictions vs valeurs réelles
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Prédictions')
    plt.title('Régression - Prédictions vs Valeurs Réelles')
    plt.show()

    # Exemple d'utilisation
if __name__ == "__main__":
    # Liste des modèles à évaluer
    models_to_evaluate = ['random_forest', 'linear_regression', 'gradient_boosting']

    # Chargement des ensembles de test
    X_test_S = pd.read_csv('../data/X_test_S.csv')
    y_test_S = pd.read_csv('../data/y_test_S.csv').squeeze()
    X_test_L = pd.read_csv('../data/X_test_L.csv')
    y_test_L = pd.read_csv('../data/y_test_L.csv').squeeze()

    # Évaluation des modèles pour la sériation 'S'
    for model_name in models_to_evaluate:
        model_S = load(f'../models/{model_name}_S.joblib')
        print(f"\nÉvaluation du modèle {model_name} pour la sériation 'S':")
        evaluate_model(model_S, X_test_S, y_test_S, model_name, 'S')

    # Évaluation des modèles pour la sériation 'L'
    for model_name in models_to_evaluate:
        model_L = load(f'../models/{model_name}_L.joblib')
        print(f"\nÉvaluation du modèle {model_name} pour la sériation 'L':")
        evaluate_model(model_L, X_test_L, y_test_L, model_name, 'L')

# def evaluate_model(model, X_test, y_test, model_name):
  #   y_pred = model.predict(X_test)
    # mae = mean_absolute_error(y_test, y_pred)
    # print(f"MAE pour {model_name}: {mae}")

    # Afficher la distribution des erreurs
#     plt.hist(y_pred - y_test.to_numpy(), bins=50)
  #   plt.xlabel('Erreur de Prédiction')
    # plt.ylabel('Fréquence')
    # plt.title('Distribution des Erreurs de Prédiction')
   #  plt.show()


# model_S = load('../models/model_S.joblib')
# model_L = load('../models/model_L.joblib')

# X_test_S = pd.read_csv('../data/X_test_S.csv')
# y_test_S = pd.read_csv('../data/y_test_S.csv')
# X_test_L = pd.read_csv('../data/X_test_L.csv')
# y_test_L = pd.read_csv('../data/y_test_L.csv')


# evaluate_model(model_S, X_test_S, y_test_S, "Modele S")
# evaluate_model(model_L, X_test_L, y_test_L, "Modele L")