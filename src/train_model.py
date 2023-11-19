import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from joblib import dump

def train_and_save_model(model, X_train, y_train, model_name, seriation):
    # Utilisation de values.ravel() pour convertir y_train en un tableau 1D
    y_train_1d = y_train.values.ravel()
    model.fit(X_train, y_train_1d)
    dump(model, f'../models/{model_name}_{seriation}.joblib')

def split_and_save_test_set(X, y, seriation):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    save_test_set(X_test, y_test, seriation)
    return X_train, y_train

def save_test_set(X_test, y_test, seriation):
    X_test.to_csv(f'../data/X_test_{seriation}.csv', index=False)
    y_test.to_csv(f'../data/y_test_{seriation}.csv', index=False)


if __name__ == "__main__":
    # Chargement des données prétraitées
    X_S = pd.read_csv('../data/X_S_scaled.csv')
    y_S = pd.read_csv('../data/y_S.csv')
    X_L = pd.read_csv('../data/X_L_scaled.csv')
    y_L = pd.read_csv('../data/y_L.csv')

     # Vérifier que les ensembles de caractéristiques ne contiennent pas les colonnes cibles
    assert 'Classement.S' not in X_S.columns, "X_S contient la colonne cible 'Classement.S'"
    assert 'Classement.L' not in X_L.columns, "X_L contient la colonne cible 'Classement.L'"



# Division des données et sauvegarde des ensembles de test
    X_train_S, y_train_S = split_and_save_test_set(X_S, y_S, "S")
    X_train_L, y_train_L = split_and_save_test_set(X_L, y_L, "L")

    # Entraînement de différents modèles pour la sériation 'S'
    model_rf = RandomForestRegressor()
    model_lr = LinearRegression()
    model_gb = GradientBoostingRegressor()

  # ... Entraînement et sauvegarde des modèles ...
    train_and_save_model(model_rf, X_train_S, y_train_S, "random_forest", "S")
    train_and_save_model(model_lr, X_train_S, y_train_S, "linear_regression", "S")
    train_and_save_model(model_gb, X_train_S, y_train_S, "gradient_boosting", "S")

    # Entraînement de différents modèles pour la sériation 'L'
    train_and_save_model(model_rf, X_train_L, y_train_L, "random_forest", "L")
    train_and_save_model(model_lr, X_train_L, y_train_L, "linear_regression", "L")
    train_and_save_model(model_gb, X_train_L, y_train_L, "gradient_boosting", "L")

# Charger les données prétraitées pour la sériation 'S' et 'L'
#X_S = pd.read_csv('../data/X_S.csv')
#y_S = X_S.pop("Classement.S") 

#X_L = pd.read_csv('../data/X_L.csv')
#y_L = X_L.pop("Classement.L") 

# Division des données en ensembles d'entraînement et de test
#X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(X_S, y_S, test_size=0.4)
#X_train_L, X_test_L, y_train_L, y_test_L = train_test_split(X_L, y_L, test_size=0.4)

# Construction et entraînement du modèle pour la sériation 'S'
#model_S = make_pipeline(MinMaxScaler(), RandomForestRegressor())
#model_S.fit(X_train_S, y_train_S)

# Construction et entraînement du modèle pour la sériation 'L'
# model_L = make_pipeline(MinMaxScaler(), RandomForestRegressor())
# model_L.fit(X_train_L, y_train_L)

# Sauvegarde des modèles entraînés
# dump(model_S, '../models/model_S.joblib')
# dump(model_L, '../models/model_L.joblib')

# Optionnel : Sauvegarde également les ensembles de test pour une évaluation ultérieure
# X_test_S.to_csv('../data/X_test_S.csv', index=False)
# y_test_S.to_csv('../data/y_test_S.csv', index=False)
# X_test_L.to_csv('../data/X_test_L.csv', index=False)
# y_test_L.to_csv('../data/y_test_L.csv', index=False)
