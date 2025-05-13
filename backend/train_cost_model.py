import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Création d'un dataset synthétique pour l'exemple
np.random.seed(42)
n_samples = 1000

# Génération des features
data = {
    'Resolution_Duration_Days': np.random.randint(1, 31, n_samples),  # 1-30 jours
    'years_experience': np.random.uniform(1, 35, n_samples),  # 1-35 ans
    'Lits': np.random.randint(50, 500, n_samples),  # 50-500 lits
    'Pourcentage_Lits': np.random.uniform(40, 100, n_samples)  # 40-100%
}

df = pd.DataFrame(data)

# Création de la variable cible (coût) avec une logique métier
# Le coût augmente avec la durée et l'expérience, diminue avec le nombre de lits disponibles
df['cout_estime'] = (
    df['Resolution_Duration_Days'] * 100 +  # 100€ par jour
    df['years_experience'] * 50 +           # 50€ par année d'expérience
    (100 - df['Pourcentage_Lits']) * 20 +  # Plus le % est bas, plus le coût est élevé
    (500 - df['Lits']) * 10                # Plus il y a de lits, moins c'est cher
)

# Normalisation pour avoir des coûts plus réalistes (entre 500€ et 5000€)
df['cout_estime'] = ((df['cout_estime'] - df['cout_estime'].min()) / 
                     (df['cout_estime'].max() - df['cout_estime'].min()) * 4500 + 500)

# Préparation des données
X = df[['Resolution_Duration_Days', 'years_experience', 'Lits', 'Pourcentage_Lits']]
y = df['cout_estime']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Évaluation du modèle
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Score R² sur l'ensemble d'entraînement : {train_score:.3f}")
print(f"Score R² sur l'ensemble de test : {test_score:.3f}")

# Sauvegarde du modèle et du scaler
joblib.dump(model, 'cost_model.joblib')
joblib.dump(scaler, 'cost_scaler.joblib')

# Sauvegarde des informations sur les features
feature_info = {
    'names': ['Resolution_Duration_Days', 'years_experience', 'Lits', 'Pourcentage_Lits'],
    'importance': dict(zip(['Resolution_Duration_Days', 'years_experience', 'Lits', 'Pourcentage_Lits'],
                         model.feature_importances_))
}
joblib.dump(feature_info, 'cost_feature_info.joblib')
