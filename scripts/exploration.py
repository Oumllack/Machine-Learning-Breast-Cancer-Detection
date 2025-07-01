import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement du dataset
df = pd.read_csv('data.csv')

# 2. Exploration rapide
def exploration_rapide(df):
    print('Aperçu des premières lignes:')
    print(df.head())
    print('\nDimensions du dataset:', df.shape)
    print('\nTypes de données:')
    print(df.dtypes)
    print('\nValeurs manquantes:')
    print(df.isnull().sum())
    print('\nStatistiques descriptives:')
    print(df.describe())
    print('\nRépartition de la variable cible:')
    print(df['diagnosis'].value_counts())

exploration_rapide(df)

# 3. Nettoyage de base
if 'id' in df.columns:
    df = df.drop('id', axis=1)
# Suppression des colonnes inutiles (ex: Unnamed: 32)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Encodage de la variable cible (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 4. Visualisations
plt.figure(figsize=(6,4))
sns.countplot(x='diagnosis', data=df)
plt.title('Répartition des classes (0=Bénin, 1=Malin)')
plt.show()

# Sélection des 10 variables les plus corrélées avec la cible
top_corr = df.corr()['diagnosis'].abs().sort_values(ascending=False)[1:11].index
plt.figure(figsize=(10,8))
sns.heatmap(df[top_corr].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation (10 variables les plus pertinentes)')
plt.show()

# 5. Sauvegarde du dataset nettoyé
df.to_csv('data_clean.csv', index=False)

print('\nNettoyage terminé. Dataset sauvegardé sous data_clean.csv.') 