
# Predicting Spotify Song Hits Using Machine Learning

## Executive Summary
This project uses Kaggle’s [Spotify Tracks Attributes and Popularity](https://www.kaggle.com/datasets/melissamonfared/spotify-tracks-attributes-and-popularity/data) dataset to test if a a supervised machine learning classification model can reliably predict song hits using attributes of the song. Reliability was defined as achieving an F1 score ≥ 0.7. A hit was defined as a popularity score ≥ 70.

The best-performing model was **XGBoost (Weighted)** with:
- **ROC AUC:** 0.87
- **Recall:** 0.81
- **Precision:** 0.14
- **F1 Score:** 0.24
- **Brier Score:** 0.14

The model is excellent at identifying most “hit” songs (high recall) and has a high ROC AUC, meaning it is effective at distinguishing between “hit” and “not hit” tracks. However, it tends to over-predict hits, leading to many false positives (low precision). This approach is suitable for advertising agencies or managers targeting a **global audience**, because global popularity metrics align with worldwide trends. It is **less suitable for national campaigns**, as cultural and language differences mean globally popular songs may not be locally popular. Ethically, this limitation should be clearly communicated to manage expectations.

---

## Data Infrastructure & Tools
- **Environment:** Google Colab (cloud-based Jupyter notebook)
- **Libraries:** pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost
- **Data Source:** Kaggle (open dataset)

Colab was chosen for its free, accessible environment and ease of sharing. Python’s ecosystem provided integrated tools for data handling, modeling, and evaluation.

---

## Data Engineering
- Created target variable: `hit = True` if popularity ≥ 70.
- Severe class imbalance (4.8% hits) handled using:
  - **Random Undersampling (RUS)**
  - **Class weights**
- Preprocessing:
  - **Scaling:** StandardScaler for numeric features
  - **Encoding:** OneHotEncoder for categorical features
- Train/test split: 80/20 with stratification.

---

## Data Visualisation
Key visuals:
- Class imbalance bar chart
- Violin plots for numeric features
- Stacked bar plots for categorical features

![Class Imbalance](Visuals/eda_class_balance_hit.png)
![Feature Ranking](Visuals/feature_ranking_clustered_bar_chart.png)

See more in the `Visuals/` folder.

---

## Data Analytics
### Hypotheses:
- **Alternative Hypothesis (H₁):** Danceability, energy, and valence are the main contributors to hit outcome.
- **Null Hypothesis (H₀):** Danceability, energy, and valence are not the main contributors to hit outcome.

### Findings:
There is insufficient evidence to reject the null hypothesis (H₀). Danceability, energy, and valence were not the main contributors to hit outcome. Stronger predictors included track genre, tempo, duration, loudness, acousticness, and instrumentalness.

Feature selection was based on Mutual Information (MI) and Point Biserial Absolute Pearson Correlation (PBAPC). Features with MI > 0.005 or PBAPC > 0.01 were retained.

### Models Tested:
- Logistic Regression (RUS and Weighted)
- Random Forest (RUS and Weighted)
- XGBoost (RUS and Weighted)

**Best Model:** XGBoost (Weighted) – high recall, low precision.

### Limitations:
- Over-predicts hits, leading to many false positives.
- Global popularity may not apply locally.
- Model needs improving before it could hyphothetically be used by marketing agencies. Permission from Kaggle and the dataset owner is also required for the suggested commercial use.

---

## Future Improvements
- Adjust classification threshold to improve precision.
- Hyperparameter tuning for XGBoost.
- Explore cultural context for popularity metrics.

  
