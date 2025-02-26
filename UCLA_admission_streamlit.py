import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
import requests
from io import BytesIO
warnings.filterwarnings("ignore")

# data prepraation function
def data_preparation(data):
    # transfer target variable 
    data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)

    # delete useless data 
    data = data.drop(['Serial_No'], axis=1)
    
    # convert University_Rating to categorical type
    data['University_Rating'] = data['University_Rating'].astype('object')
    data['Research'] = data['Research'].astype('object')

    # # Create dummy variables for all 'object' type variables except 'Loan_Status'
    clean_data = pd.get_dummies(data, columns=['University_Rating', 'Research'], dtype='int')

    # Split the Data into train and test
    x = clean_data.drop(['Admit_Chance'], axis=1)
    y = clean_data['Admit_Chance']

    # split data into trainging set and test set
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)

    # fit calculates the mean and standard deviation
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    
    # Now transform xtrain and xtest
    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    return xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, x.columns

# Neural training model function
def UCLA_Neural(xtrain_scaled, ytrain, xtest_scaled, ytest):
    # create model and train model
    MLP = MLPClassifier(hidden_layer_sizes=(3, 3), batch_size=50, max_iter=200, random_state=123)
    MLP.fit(xtrain_scaled, ytrain)

    # make predictions
    ypred_train = MLP.predict(xtrain_scaled)
    ypred = MLP.predict(xtest_scaled)

    # check accuracy of the model
    train_accuracy = accuracy_score(ytrain, ypred_train)
    test_accuracy = accuracy_score(ytest, ypred)

    # check the confusion matrix
    conf_matrix = confusion_matrix(ytest, ypred)

    # get loss curve
    loss_values = MLP.loss_curve_

    return MLP, train_accuracy, test_accuracy, conf_matrix, loss_values

# process signle student data for prediction
def prepare_student_data(student_data, feature_columns, scaler):
    # create DataFrame
    student_df = pd.DataFrame([student_data], columns=['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
    
    # convert University_Rating to categorical type
    student_df['University_Rating'] = student_df['University_Rating'].astype('object')
    student_df['Research'] = student_df['Research'].astype('object')

    # Create dummy variables
    student_clean = pd.get_dummies(student_df, columns=['University_Rating', 'Research'], dtype='int')

    # make sure align with training data
    for col in feature_columns:
        if col not in student_clean.columns:
            student_clean[col] = 0
    student_clean = student_clean[feature_columns]  

    # transform data
    student_scaled = scaler.transform(student_clean)

    return student_scaled

# Main  (Streamlit)
def main():
    st.title("UCLA Admission Prediction System")
    st.write("Predict admission possibility according to student performance and check the model")

    # load data
    #data_path = r"C:\algonquin\2025W\2216_ML\2216_project\2216_project_UCLA_Neural_Networks\data\Admission.xlsx" 
    data_url = "https://raw.githubusercontent.com/chen041081733/2216_project_UCLA_Neural_Networks/main/data/Admission.xlsx"
    try:
        response = requests.get(data_url)
        response.raise_for_status()  # make sure request successful
        data = pd.read_excel(BytesIO(response.content))
        xtrain_scaled, xtest_scaled, ytrain, ytest, scaler, feature_columns = data_preparation(data)
        model, train_accuracy, test_accuracy, conf_matrix, loss_values = UCLA_Neural(
            xtrain_scaled, ytrain, xtest_scaled, ytest
        )
        st.success("model training done based on training data")
    except FileNotFoundError:
        st.error("Training data not found, please make sure 'admission.xlsx' exist。")
        return

    # show model performance
    st.subheader("model performance(based on default data set）")
    st.write(f"Training data accuracy: {train_accuracy:.4f}")
    st.write(f"Test data accuracy: {test_accuracy:.4f}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

    #plot loss curve
    st.subheader("Loss Curve")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_values, label='Loss', color='blue')
    ax.set_title('Loss Curve')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # input student info
    st.subheader("please input student info for admission predicssion")
    gre_score = st.number_input("GRE_score 260-340）", min_value=260, max_value=340, value=320, step=1)
    toefl_score = st.number_input("TOEFL_score 0-120）", min_value=0, max_value=120, value=100, step=1)
    univ_rating = st.selectbox("University_Rating", options=[1, 2, 3, 4, 5], index=2)
    sop = st.number_input("SOP（1-5）", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
    lor = st.number_input("LOR（1-5）", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
    cgpa = st.number_input("CGPA（0-10）", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    research = st.selectbox("Research", options=[0, 1], index=0)

    if st.button("Prediction"):
        # prepare student data
        student_data = {
            'GRE_Score': gre_score,
            'TOEFL_Score': toefl_score,
            'University_Rating': univ_rating,
            'SOP': sop,
            'LOR': lor,
            'CGPA': cgpa,
            'Research': research
        }
        student_scaled = prepare_student_data(student_data, feature_columns, scaler)

        # make prediction
        prediction = model.predict(student_scaled)[0]
        probability = model.predict_proba(student_scaled)[0]

        # show prediction result
        st.write("### Prediction Result")
        st.write(f"Yes or No(0=No, 1=Yes): {prediction}")
        st.write(f"Approval Probability: {probability[1]:.4f} (Not Approved Probability: {probability[0]:.4f})")

if __name__ == "__main__":
    main()