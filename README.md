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

  <img width="642" height="305" alt="d1confusionmatrix" src="https://github.com/user-attachments/assets/143b91af-228c-423c-b291-ae94e423c448" />
  <img width="642" height="274" alt="d2cm" src="https://github.com/user-attachments/assets/52ff75a4-6f8e-4b9b-981d-9525ddc13162" />
<img width="921" height="528" alt="NaiveBayes_top_features_heatmap" src="https://github.com/user-attachments/assets/d978a9a7-07f7-4b2f-b24a-83cd617e1e2e" />



### 2. Twitter Bot Detection
* **Approach:** Feature engineering utilizing text properties, behavioral features, and profile metadata.
* **Models Compared:** Ensemble methods (Random Forest, Gradient Boosting).
* **Results:** Random Forest and Gradient Boosting achieved the best performance by fully exploiting the engineered behavioral features to separate humans from bots.
* **Visual Output:**

<img width="584" height="384" alt="account_age_days_distribution" src="https://github.com/user-attachments/assets/82af9ee8-9882-4f83-b84d-ec3f7954b70b" />
<img width="984" height="384" alt="confusion_matrices" src="https://github.com/user-attachments/assets/8c854257-8fd9-4a9e-a1d3-5805302c6cc0" />
<img width="1108" height="784" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/666bec1b-ea03-4724-8e8e-378f966da8ad" />


### 3. YouTube Comment Toxicity
* **Approach:** Comparative analysis utilizing interpretable models vs. deep learning architectures.
* **Models Compared:** Logistic Regression vs. Hybrid Multilayer Perceptron (MLP).
* **Results:** Logistic Regression yielded a highly competitive ROC-AUC score, whereas the Hybrid MLP model provided superior recall, precision, and overall generalization. SHAP (SHapley Additive exPlanations) analysis successfully illuminated complex feature interactions.
* **Visual Output:**

<img width="584" height="484" alt="Hybrid_MLP_roccurve_20251212_214651" src="https://github.com/user-attachments/assets/98d2196c-23d1-45a1-b2fd-0bc4298338ed" />

<img width="1184" height="784" alt="shap_lr_top_tokens_20251212_214651" src="https://github.com/user-attachments/assets/e6f57d68-4a51-4b94-ba08-810de3cdf523" />
<img width="1708" height="784" alt="wordcloud_toxic_nontoxic_20251212_214651" src="https://github.com/user-attachments/assets/7cf994c9-ffe0-46d4-be22-e8784d0387ec" />



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
