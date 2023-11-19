# Importations nécessaires
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import fuzz, process
from joblib import dump

def load_data(filepath):
    return pd.read_excel(filepath)


def correct_regions(df, region_column, regions_correctes):
    def trouver_correspondance(region):
        if isinstance(region, str):
            correspondance = process.extractOne(region, regions_correctes, scorer=fuzz.token_set_ratio)
            if correspondance[1] >= 90:
                return correspondance[0]
        return region

    df[region_column] = df[region_column].apply(trouver_correspondance)
    return df


def filter_data(df, seriation):
    return df.query(f"`Seriation` == '{seriation}'")

def clean_data(df, threshold_ratio):
    total_rows = df.shape[0]
    thresh = total_rows * threshold_ratio
    df_cleaned = df.dropna(axis=1, thresh=thresh)
    df_cleaned = df_cleaned.select_dtypes(exclude=["object"])
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce').fillna(df_cleaned.mean())
    return df_cleaned

def select_features_and_target(df, features, target):
    X = df[features]
    y = df[target]
    return X, y

def scale_features(X):
    """ Met à l'échelle les caractéristiques à l'aide de MinMaxScaler. """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# Exemple d'utilisation
if __name__ == "__main__":
    path = '../data/Data SESAME S1 int.xlsx'
    regions_correctes = ["Fitovinany", "Atsimo Andrefana", "Menabe", "Haute Matsiatra", 
                         "Vakinankaratra", "Boeny", "SAVA", "Tsiroanomandidy", "Atsimo-Andrefana", 
                         "Sofia", "Ihorombe", "Itasy","Antsinanana", "Vatovavy", "Analanjirofo",
                          "Alaotra Mangoro", "Atsimo Atsinanana", "Melaky", "DIANA", "Anosy",
                           "Bongolava", "Amoron'i Mania", "BETSIBOKA"]
    df = load_data(path)
    df = correct_regions(df, 'Région d\'origine', regions_correctes)

    # Traitement des données pour la sériation 'S'
    df_S = filter_data(df, 'S')
    df_S = clean_data(df_S, threshold_ratio=0.01)  # seuil spécifique pour 'S'
    features_S = ['EC3-Nuc-UE2.CC', 'EC1-InfSinf-UE4.CC', 'EC1-BioMol-UE7.CC', 
    'EC2-Rac-UE4.CC', 'UE6-PhyHum.CC', 'UE2-Chim.CC', 
    'UE4-Ang.CC', 'EC1-Fle-UE9.CC', 'MoyGen-CC.S']  # liste de caractéristiques pour 'S'
    X_S, y_S = select_features_and_target(df_S, features_S, 'Classement.S')

    # Traitement des données pour la sériation 'L'
    df_L = filter_data(df, 'L')
    df_L = clean_data(df_L, threshold_ratio=0.4)  # seuil spécifique pour 'L'
    features_L = ['EC1-Inf-UE12.CC','EC1-ComMag-UE11.CC', 'EC1-Equ-UE15.CC', 'EC2-ResTex-UE11.CC',
                   'EC2-DynMon-UE13.CC', 'MoyGen-CC.L']  # liste de caractéristiques pour 'L'
    X_L, y_L = select_features_and_target(df_L, features_L, 'Classement.L')
    
    # Sélection et mise à l'échelle des caractéristiques pour 'S' et 'L'
    X_S_scaled, y_S = select_features_and_target(df_S, features_S, 'Classement.S')
    X_L_scaled, y_L = select_features_and_target(df_L, features_L, 'Classement.L')

    X_S_scaled, scaler_S = scale_features(X_S)
    X_S_scaled_df = pd.DataFrame(X_S_scaled, columns=features_S)
    dump(scaler_S, '../models/scaler_S.joblib')  # Sauvegarde du scaler pour 'S'

    X_L_scaled, scaler_L = scale_features(X_L)
    X_L_scaled_df = pd.DataFrame(X_L_scaled, columns=features_L)
    dump(scaler_L, '../models/scaler_L.joblib')  # Sauvegarde du scaler pour 'L'


     # Enregistrement des DataFrames dans des fichiers CSV
    X_S_scaled_df.to_csv('../data/X_S_scaled.csv', index=False)
    y_S.to_csv('../data/y_S.csv', index=False)
    X_L_scaled_df.to_csv('../data/X_L_scaled.csv', index=False)
    y_L.to_csv('../data/y_L.csv', index=False)

     
# Fonction pour trouver la correspondance de région
#def trouver_correspondance(region, regions_correctes):
 #   if isinstance(region, str):
  #      correspondance = process.extractOne(region, regions_correctes, scorer=fuzz.token_set_ratio)
   #     if correspondance[1] >= 90:
    #        return correspondance[0]
    #return region

# Fonction pour charger et nettoyer les données
#def load_and_clean_data(filepath):
 #   df = pd.read_excel(filepath)
  #  regions_correctes = ["Fitovinany", "Atsimo Andrefana", "Menabe", "Haute Matsiatra", 
   #                      "Vakinankaratra", "Boeny", "SAVA", "Tsiroanomandidy", "Atsimo-Andrefana", 
    ##                     "Sofia", "Ihorombe", "Itasy","Antsinanana", "Vatovavy", "Analanjirofo",
      #                   "Alaotra Mangoro", "Atsimo Atsinanana", "Melaky", "DIANA", "Anosy",
       #                  "Bongolava", "Amoron'i Mania", "BETSIBOKA"]
    #df['Région d\'origine'] = df['Région d\'origine'].apply(lambda x: trouver_correspondance(x, regions_correctes))
    #return df

# Fonction pour préparer les données pour la modélisation
#def prepare_data_for_modeling(df, seriation):
 #   df_filtered = df.query(f"`Seriation` == '{seriation}'").copy()
  #  df_filtered.dropna(axis=1, inplace=True)
   # df_filtered = df_filtered.select_dtypes(exclude="object")
    #scaler = MinMaxScaler()
    #X = scaler.fit_transform(df_filtered)
    #columns = df_filtered.columns  
    #return pd.DataFrame(X, columns=columns)  
    
# Chargement, nettoyage et préparation des données
#if __name__ == "__main__":
 #   path = '../data/Data SESAME S1 int.xlsx'
  #  df = load_and_clean_data(path)
   # X_S = prepare_data_for_modeling(df, 'S')
    #X_L = prepare_data_for_modeling(df, 'L')

#X_S.to_csv('../data/X_S.csv', index=False)
#X_L.to_csv('../data/X_L.csv', index=False)
