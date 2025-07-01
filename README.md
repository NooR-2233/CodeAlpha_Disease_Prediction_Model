Here is your **updated `README.md`** for the **Disease Prediction From Medical Data** project, reflecting the use of only `Medical_Dataset.csv` and incorporating everything you've requested across our conversation:

---

# 🏥 Disease Prediction From Medical Data

A machine learning–powered web app that predicts possible diseases based on user-reported symptoms. Built using **Random Forest**, **Streamlit**, and real-world health data, the model provides instant feedback with **top 5 predictions**, **probabilities**, and **live medical summaries** from Wikipedia.

---

## 🚀 Key Features

* ✅ **Symptom-Based Prediction** using a Random Forest Classifier.
* ✅ **Live Wikipedia Integration** for disease descriptions.
* ✅ **Top-5 Probable Diseases** displayed with confidence scores.
* ✅ **Auto-generated** symptoms, treatments, and precautions.
* ✅ **Interactive UI** via Streamlit for real-time use.

---

## 🧠 Concepts Used

* **Supervised Learning** with Random Forest for multi-class classification.
* **Pandas & Scikit-Learn** for data preprocessing and model training.
* **Streamlit** for building an interactive web interface.
* **Wikipedia API & BeautifulSoup** for fetching readable medical summaries.
* **Probability-based ranking** to show alternate diagnoses.
* **CSV-based input features** structured around 132 binary symptom flags.

---

## 📁 Project Structure

```
DiseasePredictorML/
│
├── disease_prediction_app.py     # Main app script
├── Medical_Dataset.csv           # Symptom-disease dataset (renamed from Training.csv)
├── README.md                     # You're here!
├── .gitignore                    # Python template
└── myvenv/                       # Optional local virtual environment
```

---

## 📦 Dataset Source

🔗 [Kaggle Dataset – Disease Prediction Using ML](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

Used only the `Training.csv` file, renamed to `Medical_Dataset.csv`. `Testing.csv` was excluded for simplicity and real-time demo focus.

---

## 🔧 Installation Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pRoMasteR2002/CodeAlpha_Disease_Prediction_Model
cd CodeAlpha_Disease_Prediction_Model
```

### 2️⃣ Set Up Virtual Environment

```bash
python -m venv myvenv
myvenv\Scripts\activate           # On Windows
# or
source myvenv/bin/activate       # On macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn wikipedia beautifulsoup4
```

---

## ▶️ Run the App

```bash
streamlit run disease_prediction_app.py
```

* Select symptoms from the multiselect box.
* Click **Predict** to view the top disease guess and related information.
* Scroll to see **probabilities**, **Wikipedia summaries**, and **precaution advice**.

---

## 🧪 Sample Output

* **Input Symptoms**: fever, vomiting, muscle pain, fatigue
* **Predicted Disease**: Dengue
* **Other Possibilities**: Typhoid, Hepatitis B, Malaria...
* **Extras**:

  * Summary: Pulled from Wikipedia
  * Chart: Top 5 predictions with confidence bars
  * Advice: Suggested treatments & precautions

---

## 🔓 License

This project is licensed under the **MIT License**.
You are free to use, share, and modify the code with attribution.

---

## 🙌 Acknowledgments

Big thanks to [Kaggle](https://www.kaggle.com/) for the dataset and to **@CodeAlpha** for the challenge opportunity that inspired this project.

---

## 🎥 Demo Video

📽️ *Watch the video below to see the app in action and how it predicts diseases instantly based on your symptoms!*

---

Let me know if you’d like this version exported as a `.md` file or if you want badges, project screenshots, or auto-deploy instructions added.
