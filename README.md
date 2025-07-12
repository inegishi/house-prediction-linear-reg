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
<img width="421" height="19" alt="image" src="https://github.com/user-attachments/assets/fc2ec3ca-cdd9-4954-b64b-ed06366232e5" />

<img width="214" height="479" alt="image" src="https://github.com/user-attachments/assets/1c72b7ff-09b6-4a83-8a06-2f7b6d6dc2f0" />
