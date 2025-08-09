# 🇵🇹 Portuguese Bank Marketing Prediction

Predicting whether a customer will subscribe to a term deposit based on direct marketing campaign data from a Portuguese bank.

---

## 📋 Project Description

This project uses data from a Portuguese financial institution’s phone-based marketing campaigns (May 2008 – November 2010) to predict whether customers will subscribe to a term deposit. The dataset contains demographic information, campaign contact details, and economic indicators.

**Key goals:**
- Build a binary classification model to identify high-value customers.
- Address high class imbalance by focusing on recall (minimizing false negatives).
- Compare model performance using precision, recall, F1-score, and accuracy.

---

## ⚙️ Dataset Details

- **Source:** UCI Bank Marketing Dataset  
- **Records:** ~41,000 instances  
- **Features:** age, job, marital status, education, credit/mortgage info, call-related campaign features (`duration`, `campaign`, `pdays`, `previous`), economic indicators (`emp.var.rate`, `cons.price.idx`, `euribor3m`), and target (`y`: yes/no) :contentReference[oaicite:1]{index=1}  

*Note: `duration` is highly correlated with the target and only available after a call is made.*

---

## 🧩 Machine Learning Pipeline

1. **Data Preprocessing:**  
   - Clean and filter missing/irrelevant data.  
   - One-hot encode categorical variables.  
   - Standardize numeric features.  
   - Handle class imbalance using resampling techniques (e.g., SMOTE or weighted classes).

2. **Exploratory Data Analysis (EDA):**  
   - Visualize feature distributions and relationships.  
   - Assess feature importance and multicollinearity (e.g., via VIF).  
   - Identify low-impact variables (e.g., `day_of_week`) :contentReference[oaicite:2]{index=2}.

3. **Model Training & Evaluation:**  
   - Develop and compare classifiers: Logistic Regression, KNN, SVM, Decision Tree, Random Forest.  
   - Evaluate using recall-centric metrics, precision, F1-score, accuracy, and ROC-AUC.  
   - Prioritize models that minimize false negatives while maintaining high precision.

4. **Model Tuning:**  
   - Perform grid-/random-search and advanced tuning (e.g., Bayesian or TPOT genetic algorithms) for top-performing models :contentReference[oaicite:3]{index=3}.

5. **Result Interpretation:**  
   - Random Forest often achieves top performance with recall ~95% and precision ~90% :contentReference[oaicite:4]{index=4}.  
   - Visualizations include confusion matrices, ROC curves, and feature importance plots.

---

## 🥇 Modeling Summary (Example Results)

| Model              | Recall | Precision | F1-score | Accuracy |
|-------------------|:------:|:---------:|:--------:|:--------:|
| Logistic Regression | 0.84  | 0.83      | 0.83     | 0.84     |
| KNN                | 0.99  | 0.81      | 0.89     | 0.88     |
| Decision Tree      | 0.91  | 0.79      | 0.85     | 0.83     |
| **Random Forest**  | **0.95** | **0.90**  | **0.92** | **0.92** |

*Source: KaviM11’s implementation* :contentReference[oaicite:5]{index=5}

---

## 🛠️ Code Snippet (Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score

# X = preprocessed features, y = target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))
print("Recall:", recall_score(y_test, y_pred))

📁 Project Structure
├── data/
│   └── bank-additional-full.csv
├── notebooks/
│   └── Portuguese_Bank_Marketing.ipynb
├── models/
│   └── best_model.pkl
├── requirements.txt
└── README.md


🔧 Dependencies
Packages needed (via requirements.txt):

pandas, numpy

scikit-learn

imbalanced-learn (for balancing)

seaborn, matplotlib
🚀 Next Steps
Deploy model via Streamlit or Flask app.

Integrate interpretability tools (LIME/SHAP).

Update model using new data (e.g., different time periods, seasonal effects).

Extend the model to multi-channel marketing data.


🙏 Credits
Based on public implementations (e.g., KaviM11) and the UCI Bank Marketing dataset 
