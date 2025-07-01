import pandas as pd
import joblib
import numpy as np

# Chargement du modèle et du scaler
model = joblib.load('modele_random_forest.joblib')
scaler = joblib.load('scaler.joblib')

# Fonction de prédiction

def predire_tumeurs(input_csv=None, input_liste=None):
    if input_csv:
        # Prédiction sur un fichier CSV
        data = pd.read_csv(input_csv)
        X = data.values
    elif input_liste:
        # Prédiction sur une liste de valeurs (1 observation)
        X = np.array(input_liste).reshape(1, -1)
    else:
        print("Aucune donnée fournie.")
        return
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:,1]
    for i, (pred, proba) in enumerate(zip(y_pred, y_proba)):
        diagnostic = 'Malin' if pred == 1 else 'Bénin'
        print(f"Observation {i+1} : {diagnostic} (probabilité d'être malin : {proba:.2f})")

# Exemple d'utilisation :
# Pour prédire à partir d'un fichier CSV (mêmes colonnes que data_clean.csv sans la colonne 'diagnosis')
# predire_tumeurs(input_csv='nouvelles_tumeurs.csv')

# Pour prédire à partir d'une liste de valeurs (remplacer par vos propres valeurs)
# exemple_valeurs = [14.5, 20.0, 95.0, ...]  # 30 variables
# predire_tumeurs(input_liste=exemple_valeurs) 