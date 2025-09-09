This project is a machine learning application that predicts the risk of heart disease using patient medical data.
It also includes a simple Streamlit web app for user interaction.

🚀 How it works

The user enters medical information (age, blood pressure, cholesterol, heart rate, etc.).

The trained machine learning model processes the input.

The app predicts whether there is a high risk of heart disease.

Results are shown in a user-friendly interface.

📂 Project Structure

ui/app.py → Streamlit app code

models/final_model.pkl → Trained machine learning model

notebooks/ → Jupyter Notebooks (data preprocessing, model training, etc.)

data/ → Dataset used for training (heart_disease.csv)

results/ → Evaluation metrics and results

deployment/ → Deployment instructions (e.g., ngrok setup)

requirements.txt → List of dependencies

README.md → Project documentation

.gitignore → Ignore unnecessary files

LICENSE → License info

▶️ Run the App Locally

Clone the repository:

git clone https://github.com/yourusername/Heart_Disease_Project.git
cd Heart_Disease_Project


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run ui/app.py

📊 Model

Algorithm: Random Forest Classifier

Metrics used: Accuracy, Precision, Recall, F1-score, AUC

📌 Notes

Dataset: UCI Heart Disease Dataset

The model was trained, tuned, and evaluated in multiple steps (see notebooks/).

✨ This project is for educational purposes and should not be used as medical advice.
