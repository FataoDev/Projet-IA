import streamlit as st
import numpy as np
import joblib  # Charger le modèle
from sklearn.preprocessing import StandardScaler

# Charger le modèle KNN et le scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---- 🎨 Style de la page ----
st.set_page_config(
    page_title="Prédiction des Maladies Cardiaques ❤️",
    page_icon="💖",
    layout="centered"
)

# Ajout d'un fond coloré
st.markdown(
    """
    <style>
        body {
            background-color: #F8F9FA;
        }
        .stApp {
            background: #f0f2f6;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        .title {
            text-align: center;
            color: #ff4b4b;
            font-size: 32px;
            font-weight: bold;
        }
        .result {
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
        }
        .positive {
            background-color: #ffcccc;
            color: #ff0000;
        }
        .negative {
            background-color: #ccffcc;
            color: #008000;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- 🏥 Titre et Description ----
st.markdown("<h1 class='title'>🔍 Prédiction des Maladies Cardiaques</h1>", unsafe_allow_html=True)
st.write("👨‍⚕️ **Entrez vos informations médicales et obtenez une prédiction immédiate.**")

# ---- 📌 Inputs de l'utilisateur ----
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Âge", 20, 100, 50)
    sex = st.selectbox("Sexe", [("Homme 🚹", 1), ("Femme 🚺", 0)])[1]
    cp = st.selectbox("Type de douleur thoracique", [("Type 0", 0), ("Type 1", 1), ("Type 2", 2), ("Type 3", 3)])[1]
    trestbps = st.slider("Pression artérielle (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholestérol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Glycémie à jeun > 120 mg/dl ?", [("Non", 0), ("Oui", 1)])[1]

with col2:
    restecg = st.selectbox("Résultats électrocardiographiques", [("Normal", 0), ("Anormal", 1), ("Hypertrophie", 2)])[1]
    thalach = st.slider("Fréquence cardiaque max", 60, 220, 150)
    exang = st.selectbox("Angine induite par l'exercice ?", [("Non", 0), ("Oui", 1)])[1]
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Pente du segment ST", [("Bas", 0), ("Moyen", 1), ("Haut", 2)])[1]
    ca = st.slider("Nombre de vaisseaux colorés (0-3)", 0, 3, 1)
    thal = st.selectbox("Thalassémie", [("Normal", 3), ("Défaut fixé", 6), ("Défaut réversible", 7)])[1]

# ---- 📊 Préparation des données ----
user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
user_input = scaler.transform(user_input)  # Normaliser les données

# ---- 🧪 Bouton de Prédiction ----
if st.button("🔍 Prédire"):
    prediction = model.predict(user_input)
    
    # ---- 🎯 Affichage du résultat ----
    if prediction[0] == 1:
        st.markdown("<div class='result positive'>🚨 **Risque élevé de maladie cardiaque ! Consultez un médecin.**</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result negative'>✅ **Pas de signe de maladie cardiaque. Continuez à surveiller votre santé !**</div>", unsafe_allow_html=True)

# ---- 🩺 Footer ----
st.markdown("<hr>", unsafe_allow_html=True)
st.write("🔬 **Développé avec ❤️ par un passionné de l'IA**")
