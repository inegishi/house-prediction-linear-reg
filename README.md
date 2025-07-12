# 🏠 House Price Prediction Using Linear Regression

This project predicts housing prices using a cleaned dataset and a linear regression model. It includes steps for feature engineering, outlier removal, model training, and performance evaluation using scikit-learn.

---

## 📊 Dataset

- Source: `Housing.csv`
- Features include area, location, furnishing status, number of rooms, etc.
- Target: `price` (in Indian rupees, scaled to millions)

---

## 🔧 Project Workflow

1. **Load & Inspect Data**
   - Remove duplicates
   - Explore features

2. **Outlier Removal (IQR Method)**
   - Remove rows where any numerical feature is outside the [Q1 - 1.5×IQR, Q3 + 1.5×IQR] range

3. **Feature Engineering**
   - Identify categorical vs. numerical features
   - One-hot encode categorical features with ≤ 16 unique values
   - Scale target variable (`price`) to millions

4. **Model Training**
   - Use `LinearRegression` from `scikit-learn`
   - Split data using `train_test_split` with shuffling

5. **Evaluation**
   - Metrics: Mean Squared Error (MSE), R² Score
   - Typical R² after fixing data split: ~0.75+

---

## 📈 Sample Results
<img width="428" height="486" alt="image" src="https://github.com/user-attachments/assets/7b620d94-1bcf-4185-b222-d4040e15c3a3" />
