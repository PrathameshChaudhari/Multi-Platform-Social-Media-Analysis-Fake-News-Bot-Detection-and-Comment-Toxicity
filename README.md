# Multi-Platform Social Media Analysis: Fake News, Bot Detection, and Comment Toxicity

## 📌 Project Overview
This project applies CRISP-DM and ML models (Naive Bayes, Random Forest, MLP) across three datasets to detect fake news, identify social media bots, and classify YouTube comment toxicity. It leverages feature engineering and SHAP analysis to build highly interpretable, reproducible models for robust online content moderation.
**Team Members:** Prathamesh Chaudhari, Srushti Gowda, Rutik Chaudhari

## 📊 Key Results & Visualizations

### 1. Fake News Detection
* **Approach:** Lexical analysis extracting textual differences between fake and real content.
* **Models Compared:** Multinomial Naive Bayes vs. K-Nearest Neighbours (KNN).
* **Results:** Naive Bayes outperformed KNN in terms of both accuracy and F1-scores.
* **Visual Output:**
  

### 2. Twitter Bot Detection
* **Approach:** Feature engineering utilizing text properties, behavioral features, and profile metadata.
* **Models Compared:** Ensemble methods (Random Forest, Gradient Boosting).
* **Results:** Random Forest and Gradient Boosting achieved the best performance by fully exploiting the engineered behavioral features to separate humans from bots.
* **Visual Output:**


### 3. YouTube Comment Toxicity
* **Approach:** Comparative analysis utilizing interpretable models vs. deep learning architectures.
* **Models Compared:** Logistic Regression vs. Hybrid Multilayer Perceptron (MLP).
* **Results:** Logistic Regression yielded a highly competitive ROC-AUC score, whereas the Hybrid MLP model provided superior recall, precision, and overall generalization. SHAP (SHapley Additive exPlanations) analysis successfully illuminated complex feature interactions.
* **Visual Output:**


## 🧠 Methodology 
This project strictly adhered to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:
1. **Business Understanding:** Addressing the critical need for misinformation detection and content moderation.
2. **Data Understanding:** Analyzing three distinct datasets across diverse social platforms.
3. **Data Preparation:** Lexical extraction, metadata parsing, and behavioral feature engineering.
4. **Modeling:** Implementing both traditional ML (Logistic Regression, Naive Bayes, KNN) and advanced/ensemble techniques (Random Forest, Gradient Boosting, MLP).
5. **Evaluation:** Prioritizing reproducible, interpretable models utilizing SHAP analysis to validate model logic.

## 🗂️ Repository Structure
```text
├── fakenews/                 <-- Code and outputs for Fake News detection
├── TwitterBotDetection/      <-- Code and outputs for Bot classification
├── youtubeComms/             <-- Code and outputs for Comment Toxicity
└── README.md
```

## 🛠️ Technology Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, XGBoost/Gradient Boosting, Keras/TensorFlow (MLP)
* **Interpretability:** SHAP
* **Workflow:** CRISP-DM
