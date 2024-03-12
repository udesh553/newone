 
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib


# Connect to the SQLite database
conn = sqlite3.connect("image_database.db")
cursor = conn.cursor()


def fetch_image_path(dataset, model, explainable_type):
    cursor.execute("SELECT file_path FROM images WHERE dataset=? AND model=? AND explanable_type=?", (dataset, model, explainable_type))
    row = cursor.fetchone()
    if row:
        return row[0]  # Return the image path
    else:
        return None  # Return None if image not found


def sub_header(model,explanation_type):
    st.subheader(f"{explanation_type} Explanation for {model}")





model_mapper={"Random Forest":"RF", 
 "Logistic Regression" :"LR"}

type_mapper={"Lime" :"lime",
  "Shap": "Shap",
  "Feature Importance" : "Feature_imp",
    "Partial Dependence Plot" :"pdp" ,
    "Counterfactual": "weight",
     "Show All" :"show_all" }

explain_types=["lime","Shap","Feature_imp","pdp","weight"]

# st.title("Intrusion Detection Systems Model Preformance interpretation ")
st.image("newly_edited.jpg")

# Set up Streamlit sidebar
st.sidebar.title("Model Explainer")
selected_dataset = st.sidebar.selectbox("Select Dataset", ["UNSW_NB15", "NSL_KDD"])
selected_model = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
selected_result = st.sidebar.selectbox("Select Result", ["Lime", "Shap", "Feature Importance", "Partial Dependence Plot", "Counterfactual", "Show All"])

if st.sidebar.button("Show Results"):
    if type_mapper[selected_result]=="show_all":
        keys_list = list(type_mapper.keys())
        for num,exps in enumerate(explain_types):
            image_path=fetch_image_path(selected_dataset,model_mapper[selected_model],exps)
            st.subheader(f"{keys_list[num]} Explanation with  {selected_model}")
            st.image(image_path)
    else:
        st.subheader(f"{selected_result} Explanation with  {selected_model}")
        if selected_result=="Shap":
            image_path=fetch_image_path(selected_dataset,model_mapper[selected_model],type_mapper[selected_result])
            message = f"""Here are the results using the **{selected_model}** model.
            The top plot is a summary of the features and their importance for the three classes.The bottom three plots illustrate how the model's predictions change for each feature value, 
            with respect to each class (class 0, class 1, and class 2).
            """
            st.write(message)
            st.image(image_path)
        elif type_mapper[selected_result]=="pdp":
            message = f"""Here are the results using the **{selected_model}** model. The  three plots illustrate how the model's predictions change for 3  feature values, with respect to each class (class 0, class 1, and class 2).
            """
            st.write(message)
            image_path=fetch_image_path(selected_dataset,model_mapper[selected_model],type_mapper[selected_result])
            st.image(image_path)
        else:
            image_path=fetch_image_path(selected_dataset,model_mapper[selected_model],type_mapper[selected_result])
            st.write(f"Here are the results using the **{selected_model}** model with model preformance explenation")
            st.image(image_path)

####################


# Upllad test dataset
st.sidebar.subheader("Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
if st.sidebar.button("Predict Results"):
    st. subheader("Test Dataset Results :")
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

        # Display uploaded test dataset
        st.sidebar.subheader("Uploaded Test Dataset")
        st.write("Your Uploaded Dataset")
        st.write(test_df.head())

        # Load Pre trained model 
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        model = joblib.load('logistic_regression_model_nslkdd.pkl')

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Display metrics
        st.write("Performance Metrics Results with Logistic Regression : \n")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 Score: {f1}")
    ###################



 



