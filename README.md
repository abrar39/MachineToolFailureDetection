# 🧠 Machine Tool Condition Detection using Local Outlier Factor (LOF)

### 🚀 A Flask Web App for Predicting Machine Tool Health using Anomaly Detection

![Flask](https://img.shields.io/badge/Framework-Flask-blue)
![Python](https://img.shields.io/badge/Language-Python-yellow)
![Model](https://img.shields.io/badge/Model-Local%20Outlier%20Factor-green)
![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render.com-purple)

---

## 📘 Overview

This project demonstrates the end-to-end development and deployment of a **machine learning application** that predicts the **health condition of a machine tool** based on real-time sensor data. The model identifies whether the machine is in a **"Normal"** state or has a potential **"Failure"** condition.

The model was trained using the **Local Outlier Factor (LOF)** algorithm — a robust **unsupervised anomaly detection** method ideal for highly **imbalanced datasets** where failure events are rare.

The complete pipeline includes:
- Data preprocessing and feature engineering  
- Model selection and training using scikit-learn  
- Model serialization and deployment via Flask  
- Cloud hosting on [Render.com](https://machinetoolfailuredetection.onrender.com/)

---

## 🧩 Problem Statement

Industrial machine tools are equipped with various sensors that monitor their operational parameters. However, **failure cases** are rare, making **supervised classification models** ineffective due to class imbalance.

To address this, the project applies **anomaly detection** to identify deviations from normal operational behavior using features such as:

| Feature | Description |
|----------|--------------|
| Process Temperature | Temperature during operation |
| Air Temperature | Ambient air temperature |
| Rotational Speed | RPM of the machine tool |
| Torque | Applied torque value |
| Tool Wear | Cumulative wear on the tool |
| Type | Type of tool in use |

---

## ⚙️ Machine Learning Approach

### 1. **Exploratory Data Analysis (EDA)**
- Visualized distributions and correlations of sensor readings  
- Identified skewed data and outliers in operational parameters  
- Examined imbalance between "Normal" and "Failure" cases  

### 2. **Data Preprocessing**
- Standardized numeric features using `StandardScaler`  
- Encoded categorical variable `Type` using one-hot encoding
- The data did not have noisy points or missing values.    

### 3. **Model Selection**
Due to high data imbalance, anomaly detection was a suitable approach to train the model.
Multiple anomaly detection models were tested such as *Local Outlier Factor*, *Isolation Forest*, and *One Class SVM*.
Out these, LOF was the best performing model. It was selected for **Hyperparameter tuning**.
- LOF detects rare deviations in data without requiring class labels  
- Provides a robust binary output: **-1 (Anomaly/Failure)** and **1 (Normal)**  

### 4. **Model Training & Evaluation**
- Trained LOF using features: `['Process temperature', 'Air temperature', 'Rotational speed', 'Torque', 'Tool wear', 'Type']`  
- Tuned hyperparameters such as `n_neighbors` and `contamination`  
- Evaluated using:
  - Confusion matrix for binary classification performance  
  - Precision and recall of failure detection  

### 5. **Model Serialization**
- Model saved using **Pickle** for deployment  
- Input preprocessing pipeline also serialized to ensure consistent predictions  

---

## 💻 Flask Web Application

A lightweight **Flask** interface allows users to input machine parameters and receive instant predictions.

### 🔹 Web Features
- Simple input form for entering sensor data  
- Model prediction displayed as:
  - ✅ **Normal Condition**
  - ⚠️ **Failure Condition (Potential Anomaly)**
- Backend built using Flask and scikit-learn  

### 🔹 Deployment
- App deployed on **Render.com** for free testing  
- GitHub repository integrated with Render for continuous deployment  
- Requirements managed via `requirements.txt`

---

## 🧠 Skills Demonstrated

| Category | Skills |
|-----------|--------|
| **Data Science & Machine Learning** | Data preprocessing, Feature engineering, Model selection for imbalanced data, Anomaly detection, Scikit-learn pipeline |
| **Software Engineering** | Flask app development, REST API design, Modular code structure |
| **Model Deployment (MLOps)** | Pickle model serialization, Environment setup with requirements.txt, Render.com deployment |
| **Version Control & CI/CD** | Git, GitHub integration with Render, Automated build and deploy |
| **Visualization & Reporting** | Matplotlib/Seaborn EDA, Model evaluation metrics, Clear documentation |

---

## 🧪 Example Usage

1. Go to the deployed app: [🔗 Live Demo on Render](https://machinetoolfailuredetection.onrender.com/)
2. [Youtube Demo](https://youtu.be/bnHs0urwd6c)
3. Input the following values:

| Feature | Example Input |
|----------|----------------|
| Process Temperature | 308.6 |
| Air Temperature | 295.0 |
| Rotational Speed | 1430 |
| Torque | 40.2 |
| Tool Wear | 120 |
| Type | H |

3. Click **Predict** → The app will output:  
**✅ Normal Condition** *or* **⚠️ Failure Condition**

---

## 🧰 Technologies Used

- **Python 3.10+**  
- **Flask**  
- **scikit-learn**  
- **pandas**, **numpy**  
- **matplotlib**, **seaborn**  
- **Render.com**  
- **GitHub**

---

## 🧾 Future Improvements

- Integrate **real-time IoT sensor streaming** for continuous monitoring  
- Replace LOF with **Isolation Forest** or **Autoencoder** models for comparison  
- Add visual dashboard using **Plotly Dash** or **Streamlit**  
- Containerize the app using **Docker** for scalable deployment  

---

## 👤 Author

**Abrar Asghar**  
🔗 [GitHub Profile](https://github.com/abrar39)  
💼 *Machine Learning Based Reliability Engineering*  
📧 abrar.asghar@gmail.com

