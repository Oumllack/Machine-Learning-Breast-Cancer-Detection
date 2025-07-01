import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement du dataset nettoyé
df = pd.read_csv('data_clean.csv')

# 2. Séparation X/y
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# 3. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modélisation
models = {
    'Régression Logistique': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {'accuracy': acc, 'auc': auc, 'report': classification_report(y_test, y_pred, output_dict=True)}
    print(f'\n--- {name} ---')
    print('Accuracy:', acc)
    print('AUC:', auc)
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('Matrice de confusion:\n', confusion_matrix(y_test, y_pred))

# 6. Comparaison des modèles
accs = [results[m]['accuracy'] for m in models]
aucs = [results[m]['auc'] for m in models]
plt.figure(figsize=(8,4))
sns.barplot(x=list(models.keys()), y=accs)
plt.title('Comparaison des accuracies')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(x=list(models.keys()), y=aucs)
plt.title('Comparaison des AUC')
plt.ylabel('AUC')
plt.show()

# 7. Courbe ROC du meilleur modèle
best_model_name = max(results, key=lambda m: results[m]['auc'])
best_model = models[best_model_name]
y_proba = best_model.predict_proba(X_test_scaled)[:,1] if hasattr(best_model, 'predict_proba') else best_model.decision_function(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC {best_model_name} (AUC={results[best_model_name]["auc"]:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend()
plt.show()

print(f'\nLe meilleur modèle est : {best_model_name}') 