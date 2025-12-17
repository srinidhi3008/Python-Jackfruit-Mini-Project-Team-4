import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #A7C7E7;
    }
    </style>
    """,
    unsafe_allow_html=True)
iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.title("Iris Flower Classifier 🌼 ")
st.write("Enter the flower measurements and get the predicted Iris species.")

sepal_length = st.slider("Sepal length (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width  = st.slider("Sepal width (cm)",  float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.slider("Petal length (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width  = st.slider("Petal width (cm)",  float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_class = model.predict(input_data)[0]
    pred_name = iris.target_names[pred_class]
    st.success(f"Predicted species: {pred_name.capitalize()}")

