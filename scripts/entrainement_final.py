import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

# Chargement des données
print('Chargement des données...')
df = pd.read_csv('data_clean.csv')
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Séparation train/test pour évaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle Random Forest
print('Entraînement du modèle Random Forest...')
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Évaluation
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:,1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f'\nAccuracy : {acc:.3f}')
print(f'AUC : {auc:.3f}')
print('Classification report :\n', classification_report(y_test, y_pred))
print('Matrice de confusion :\n', confusion_matrix(y_test, y_pred))

# Réentraînement sur tout le dataset
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)

# Sauvegarde du modèle et du scaler
joblib.dump(model, 'modele_random_forest.joblib')
joblib.dump(scaler, 'scaler.joblib')
print('\nModèle et scaler sauvegardés.') 