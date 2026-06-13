import streamlit as st
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Species Prediction")
st.write("Predict the Iris flower species using AdaBoost Machine Learning Model.")

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\Machine_Learning_Project\AdaBoost_Iris_Project\Iris (1).csv")
    return df

df = load_data()

# ==========================
# PREPROCESSING
# ==========================
X = df[['SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm']]

y = df['Species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==========================
# TRAIN MODEL
# ==========================
model = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1,
    random_state=0
)

model.fit(X, y_encoded)

# ==========================
# USER INPUTS
# ==========================
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)

# ==========================
# PREDICTION
# ==========================
if st.button("Predict Species"):

    sample = pd.DataFrame({
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    })

    prediction = model.predict(sample)[0]

    species = le.inverse_transform([prediction])[0]

    st.success(f"Predicted Species: {species}")

# ==========================
# DATASET PREVIEW
# ==========================
with st.expander("View Dataset"):
    st.dataframe(df.head())