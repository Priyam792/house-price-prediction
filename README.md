# 🏠 House Price Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An end-to-end Machine Learning project for predicting median house values using **Scikit-learn** and **Random Forest Regression**.

The project covers data preprocessing, stratified sampling, model comparison, cross-validation, model persistence, and prediction on unseen test data.

## 📑 Table of Contents

- [Features](#-features)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Project Structure](#-project-structure)
- [Files Explained](#-files-explained)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Model Persistence](#-model-persistence)
- [Technologies Used](#-technologies-used)
- [What I Learned](#-what-i-learned)
- [Future Improvements](#-future-improvements)
- [Project Status](#-project-status)
- [Author](#-author)

## 🚀 Features

- 📊 Data preprocessing using Scikit-learn
- 🔀 Stratified train-test splitting
- 🧹 Missing-value handling with `SimpleImputer`
- 📏 Feature scaling using `StandardScaler`
- 🔤 Categorical feature encoding using `OneHotEncoder`
- 🌲 Random Forest Regression
- 🧪 Model comparison using:
  - Linear Regression
  - Decision Tree Regression
  - Random Forest Regression
- 📈 10-fold cross-validation using RMSE
- 💾 Model and preprocessing pipeline persistence using `joblib`
- 🔮 Prediction on test data
- 📄 Automatic generation of prediction results in `output.csv`

## 🧠 Machine Learning Workflow

```text
housing.csv
    ↓
Data Preprocessing
    ↓
Train / Test Split
    ↓
Model Comparison
    ↓
Cross-Validation (RMSE)
    ↓
Random Forest Regression
    ↓
model.pkl + pipeline.pkl
    ↓
input.csv
    ↓
Prediction
    ↓
output.csv
```

## 📂 Project Structure

```text
house-price-prediction/
│
├── main.py
├── model-selection.py
├── housing.csv
├── input.csv
├── output.csv
├── README.md
├── requirements.txt
└── .gitignore
```

> **Note:** `model.pkl` and `pipeline.pkl` are not included in this repository. They're generated automatically by `main.py` when required.

## 📁 Files Explained

| File | Description |
|---|---|
| `main.py` | Main script for training and prediction. Trains the model and creates `model.pkl` / `pipeline.pkl` if they don't exist yet; otherwise loads them and predicts directly. |
| `model-selection.py` | Used during experimentation to compare Linear Regression, Decision Tree Regression, and Random Forest Regression via 10-fold cross-validation (RMSE). |
| `housing.csv` | The original housing dataset used to train the model. |
| `input.csv` | Test data used for prediction — auto-generated from the test split on the first run if it doesn't exist. |
| `output.csv` | The input data plus the model's predicted house values. |
| `model.pkl` | The trained `RandomForestRegressor`. Not included in the repo — generated automatically by `main.py`. |
| `pipeline.pkl` | The fitted preprocessing pipeline, used to transform new input data the same way as training data. Also auto-generated, also not included. |

## 🔧 Installation

Clone the repository:
```bash
git clone https://github.com/Priyam792/house-price-prediction.git
```

Move into the project directory:
```bash
cd house-price-prediction
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

## 🏃 Running the Project

Run:
```bash
python main.py
```

**First run** — if `model.pkl` and `pipeline.pkl` aren't present, `main.py` will:
1. Load `housing.csv`
2. Prepare the dataset
3. Create a stratified train-test split
4. Build the preprocessing pipeline
5. Train the Random Forest model
6. Generate `model.pkl`
7. Generate `pipeline.pkl`
8. Generate `input.csv`

**Second run** — run the same command again:
```bash
python main.py
```

If the model files already exist, the program will instead:
1. Load `model.pkl`
2. Load `pipeline.pkl`
3. Read `input.csv`
4. Transform the input data
5. Generate predictions
6. Save the results to `output.csv`

You don't need to manually create the `.pkl` files — they're generated automatically.

## 🔄 Model Persistence

The project uses `joblib` to save and load the trained model and preprocessing pipeline:

```python
joblib.dump(model, "model.pkl")
joblib.dump(pipeline, "pipeline.pkl")
```

This means the model doesn't need to be retrained every time:
- If the `.pkl` files exist, they're loaded automatically.
- If they're missing, `main.py` trains the model and creates them.

## 🧰 Technologies Used

**Core**
- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib

**Machine Learning**
- Random Forest Regression
- Linear Regression
- Decision Tree Regression
- Cross-Validation
- RMSE

**Data Preprocessing**
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `Pipeline`
- `ColumnTransformer`
- `StratifiedShuffleSplit`

## 📚 What I Learned

Through this project, I practiced:

- Preparing a dataset for Machine Learning
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Creating preprocessing pipelines
- Using stratified sampling
- Comparing different ML algorithms
- Using cross-validation
- Evaluating regression models using RMSE
- Saving trained ML models
- Reusing preprocessing pipelines
- Making predictions on unseen data

## 🔮 Future Improvements

- [ ] Add more regression algorithms
- [ ] Perform hyperparameter tuning
- [ ] Add MAE and R² evaluation
- [ ] Add exploratory data analysis and visualizations
- [ ] Build a web interface for predictions
- [ ] Improve model performance

## 📌 Project Status

🚧 **Completed** as a Machine Learning learning project.

This project was built while learning Machine Learning and Scikit-learn, with a focus on understanding the full workflow — from data preprocessing and model selection to model training and inference.

## 👨‍💻 Author

**Priyam**
GitHub: [@Priyam792](https://github.com/Priyam792)
