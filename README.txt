===============================
AI Symptom Checker with Streamlit
===============================

📌 Description:
--------------
This project is an AI-powered medical assistant that recommends medications based on selected symptoms.
It uses a machine learning model (Random Forest Classifier) trained on a custom dataset.

🛠️ Features:
------------
- Simple and intuitive web interface using Streamlit
- Multiple symptom selection using checkboxes
- Predicts the most likely medications with confidence scores
- Shows which symptoms contributed to each medication prediction

📁 Project Structure:
---------------------
streamlit_app/
├── app.py               ← Main Streamlit application
├── model 2 ai.csv       ← Dataset file
└── README.txt           ← Project information and usage instructions

⚙️ Requirements:
---------------
Make sure you have Python installed (preferably via Anaconda).
Required libraries:
- streamlit
- pandas
- numpy
- scikit-learn

Install them via:
    pip install streamlit pandas numpy scikit-learn

🚀 How to Run:
--------------
1. Open Anaconda Prompt (or Command Prompt).
2. Navigate to the project directory. Example:
       cd Desktop\streamlit_app
3. Run the Streamlit app:
       streamlit run app.py
4. A web browser will open automatically at:
       http://localhost:8501

📌 Notes:
---------
- Ensure the CSV file `model 2 ai.csv` is in the same folder as `app.py`.
- You can customize the dataset to include more symptoms or medications.

🧠 Model:
---------
- Machine Learning Algorithm: Random Forest Classifier
- Input: Binary vector representing selected symptoms
- Output: Probabilities for each possible medication

👨‍💻 Developed by:
------------------
Alaa Yaser (Graduation Project)
Faculty of [Your Faculty Name]
Department of Computer Science

📅 Year: 2025

