UCLA Admission Prediction System

This application utilizes Streamlit and Machine Learning (Neural Networks) to predict a student's admission probability to UCLA based on academic and research performance. The model is trained on GRE scores, TOEFL scores, CGPA, and other factors to provide an accurate admission prediction..

Features
•	User-friendly web interface powered by Streamlit.
•	Automated dataset processing for model training and evaluation.
•	Neural Network-based prediction model using MLPClassifier.
•	Real-time student admission prediction based on input scores.
•	Performance evaluation with accuracy scores, confusion matrix, and loss curve visualization.
•	Accessible via Streamlit Community Cloud.

Dataset
The model is trained on UCLA Admission Dataset, which includes the following features:
•	GRE Score: Graduate Record Exam score (260-340).
•	TOEFL Score: Test of English as a Foreign Language score (0-120).
•	University Rating: Rating of the university (1-5).
•	SOP: Statement of Purpose rating (1-5).
•	LOR: Letter of Recommendation rating (1-5).
•	CGPA: Cumulative Grade Point Average (0-10).
•	Research: Whether the student has research experience (0 = No, 1 = Yes).
•	Admit_Chance: Binary target variable (1 = Admitted, 0 = Not Admitted).

Technologies Used
•	Streamlit – For web application development.
•	Scikit-learn – For training a neural network model.
•	Pandas & NumPy – For data processing and feature engineering.
•	Matplotlib – For visualizing model performance (loss curve).
•	Requests & BytesIO – For loading the dataset from GitHub.

Model
Preprocessing
•	Converts categorical variables (University Rating, Research) into dummy variables.
•	Uses MinMaxScaler to scale numerical features.
•	Splits data into 80% training and 20% testing sets.
Neural Network (MLPClassifier)
•	Architecture: 3 hidden layers with 3 neurons each.
•	Batch size: 50
•	Training iterations: 200
•	Activation function: ReLU
•	Optimization: Adam Optimizer
•	Evaluated using accuracy score and confusion matrix

Installation (for local deployment)
If you want to run the application locally, follow these steps:
1.	Clone the repository:
git clone https://github.com/chen041081733/2216_project_UCLA_Neural_Networks.git
cd 2216_project_UCLA_Neural_Networks
2.	Create and activate a virtual environment:
python -m venv env
source env/bin/activate  # On Windows, use `env\\Scripts\\activate`
3.	Install dependencies:
pip install -r requirements.txt
4.	Run the Streamlit application:
streamlit run UCLA_admission_streamlit.py

Usage
Dataset Loading:
•	The app automatically loads the dataset from GitHub.
•	Performs preprocessing and splits data into training and test sets.
•	Trains a neural network model.
Model Performance Analysis:
•	Displays training & test accuracy.
•	Shows confusion matrix for error analysis.
•	Plots loss curve to visualize model optimization.
Admission Prediction:
•	Users input their GRE, TOEFL, CGPA, SOP, LOR, and Research experience.
•	The model predicts whether the student will be admitted.
•	Displays probability of admission for better decision-making.

Thank you for using the UCLA Admission Prediction System!

Streamlit link:
https://2216projectuclaneuralnetworks-zqkexqzyw2csoyfege6v9x.streamlit.app/

