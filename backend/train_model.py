import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib

# Connexion à la base de données
conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=DESKTOP-E66M1OC;"
    "DATABASE=DW_HealthCare;"
    "UID=hama;"
    "PWD=Azerty@12345;"
)

# Chargement des données
patients = pd.read_sql("SELECT * FROM dim_Patient", conn)
doctors = pd.read_sql("SELECT * FROM Dim_Medecin", conn)
drugs = pd.read_sql("SELECT * FROM Dimension_PDRUG", conn)
procev = pd.read_sql("SELECT * FROM Dimension_ProEvent", conn)
admissions = pd.read_sql("SELECT * FROM dim_Admission", conn)

# Conversion des hadm_id en string
for table in [doctors, drugs, admissions]:
    table['hadm_id'] = table['hadm_id'].astype(str)

# Fusion des tables
df = patients\
    .merge(procev, on='subject_id', how='left')\
    .merge(doctors, left_on='Patient_FK', right_on='Docteur_FK', how='left')\
    .merge(drugs, on='hadm_id', how='left')\
    .merge(admissions[['hadm_id', 'discharge_location']], on='hadm_id', how='left')

# Prétraitement des données
df['BMI'] = pd.to_numeric(df['BMI'].astype(str).str.replace(',', '.'), errors='coerce')
df['BMI'].fillna(df['BMI'].median(), inplace=True)

df['gender'] = df['gender'].astype(str).str.strip().str.upper()
df = df[df['gender'].isin(['M', 'F'])]
df['gender'] = df['gender'].map({'M': 1, 'F': 0})

df['years_experience'] = pd.to_numeric(df['years_experience'], errors='coerce').fillna(0)
df['speciality'] = df['speciality'].fillna('Unknown')
df['dod_hosp'] = pd.to_numeric(df['dod_hosp'], errors='coerce')
df = df[df['dod_hosp'].isin([0, 1])]

for col in ['drug_type', 'statusdescription', 'discharge_location']:
    df[col] = df[col].fillna('Unknown')
for col in ['speciality', 'drug_type', 'statusdescription', 'discharge_location']:
    df[col] = LabelEncoder().fit_transform(df[col])

# Préparation des features
features = ['BMI', 'gender', 'years_experience', 'speciality', 'discharge_location']
X = df[features]
y = df['dod_hosp']

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=20,
    random_state=42
)
model.fit(X_train, y_train)

# Évaluation du modèle
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.6
y_pred = (y_prob >= threshold).astype(int)

print(f"\n✅ Évaluation avec threshold = {threshold}")
print("\n📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))
print("📌 ROC AUC Score:", roc_auc_score(y_test, y_prob))
print("📌 Accuracy:", accuracy_score(y_test, y_pred))

# Validation croisée
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print("\n📘 Résultats de la validation croisée (5-folds) :")
print("Scores individuels :", cv_scores)
print(" Moyenne :", round(cv_scores.mean(), 4))
print(" Écart-type :", round(cv_scores.std(), 4))

# Sauvegarde du modèle et du scaler
print("\nSauvegarde du modèle et du scaler...")
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Sauvegarde terminée !")

# Sauvegarde des informations sur les features
feature_info = {
    'names': features,
    'encoders': {
        'speciality': df['speciality'].unique().tolist(),
        'discharge_location': df['discharge_location'].unique().tolist()
    }
}
joblib.dump(feature_info, 'feature_info.joblib')
