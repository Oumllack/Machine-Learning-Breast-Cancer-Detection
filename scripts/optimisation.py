import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Chargement du dataset nettoyé
df = pd.read_csv('data_clean.csv')
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# 2. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Grilles d'hyperparamètres
param_grids = {
    'Régression Logistique': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}

models = {
    'Régression Logistique': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

best_results = {}

for name in models:
    print(f'\nOptimisation pour {name}...')
    grid = GridSearchCV(models[name], param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:,1] if hasattr(best_model, 'predict_proba') else best_model.decision_function(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print('Meilleurs paramètres :', grid.best_params_)
    print('Accuracy sur test :', acc)
    print('AUC sur test :', auc)
    best_results[name] = {'params': grid.best_params_, 'accuracy': acc, 'auc': auc}

# Affichage récapitulatif
tab = pd.DataFrame(best_results).T
print('\nRésumé des meilleurs modèles :')
print(tab) 