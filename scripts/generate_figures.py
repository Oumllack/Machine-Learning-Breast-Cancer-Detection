import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv('data_clean.csv')

# 1. Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='diagnosis', data=df)
plt.title('Class Distribution (0 = Benign, 1 = Malignant)')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('figure_class_distribution.png')
plt.close()

# 2. Correlation Matrix (Top 10 Features)
corr = df.corr()
top_corr = corr['diagnosis'].abs().sort_values(ascending=False)[1:11].index
plt.figure(figsize=(10,8))
sns.heatmap(df[top_corr].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix (Top 10 Features)')
plt.tight_layout()
plt.savefig('figure_correlation_top10.png')
plt.close()

# 3. Feature Importances (Random Forest)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
importances = model.feature_importances_
indices = np.argsort(importances)[-10:][::-1]
plt.figure(figsize=(10,6))
plt.title('Top 10 Feature Importances (Random Forest)')
plt.bar(range(10), importances[indices], align='center')
plt.xticks(range(10), X.columns[indices], rotation=45)
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('figure_feature_importances.png')
plt.close()

# 4. Confusion Matrix
y_pred = model.predict(X_scaled)
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Random Forest)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('figure_confusion_matrix.png')
plt.close()

# 5. ROC Curve
proba = model.predict_proba(X_scaled)[:,1]
fpr, tpr, _ = roc_curve(y, proba)
auc = roc_auc_score(y, proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.tight_layout()
plt.savefig('figure_roc_curve.png')
plt.close()

# 6. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, proba)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend()
plt.tight_layout()
plt.savefig('figure_precision_recall.png')
plt.close()

print('All scientific figures have been generated and saved as PNG files.') 