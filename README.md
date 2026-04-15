# 🛡️ Car Insurance Premium Prediction

Advanced Machine Learning regression model for predicting vehicle insurance premiums with a Streamlit web interface.

## 🎯 Features

- **Advanced ML Models**: RandomForest, XGBoost, and GradientBoosting algorithms
- **Feature Engineering**: Polynomial features and interaction terms for improved predictions
- **Streamlit Dashboard**: Interactive web app for real-time predictions
- **Model Comparison**: Detailed metrics for all trained models
- **Data Insights**: Statistical analysis and visualizations

## 📊 Model Performance

| Model | Train RMSE | Test RMSE | Test MAE | Test R² |
|-------|-----------|-----------|----------|---------|
| GradientBoosting | 52.74 | 277.68 | 202.60 | 0.5718 |
| **RandomForest** | **114.98** | **260.70** | **189.54** | **0.6226** |
| XGBoost | 61.02 | 271.14 | 197.36 | 0.5918 |

**Best Model**: Random Forest with Test RMSE of **260.70**

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/dmitrytuchin/car-insurance.git
cd car-insurance
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train_model.py
```

This will:
- Load and preprocess the insurance dataset
- Train multiple regression models
- Compare model performance
- Save the best model to `model.pkl`

Expected output:
```
Features: ['vehicle_age_years', 'vehicle_type', 'no_of_kilometers_run', 'number_of_claims', 'region']
  Numerical : ['vehicle_age_years', 'no_of_kilometers_run', 'number_of_claims']
  Categorical: ['vehicle_type', 'region']
  Enhanced features: 10 (added interaction terms)
  Samples   : 1000

Training GradientBoosting ...
Training RandomForest ...
Training XGBoost ...
...
✓ Best model: RandomForest
✓ Test RMSE : 260.7038
```

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 Using the App

### Prediction Tab (🔮)
1. Adjust vehicle parameters in the sidebar:
   - Vehicle Age (years)
   - Vehicle Type
   - Kilometers Run
   - Number of Claims
   - Region
2. Click **Predict Premium** to get an instant prediction
3. View the estimated insurance premium and input summary

### Model Performance Tab (📊)
- View the best model and its metrics
- Compare all trained models
- See model hyperparameters (if optimized)

### Data Insights Tab (📈)
- Dataset overview (records, features, missing values)
- Target distribution
- Premium statistics by vehicle type
- Premium statistics by region
- Sample data preview

## 📁 Project Structure

```
car-insurance/
├── train_model.py                           # Model training script
├── app.py                                   # Streamlit web application
├── challenging_insurance_dataset_regression.csv  # Dataset
├── requirements.txt                         # Python dependencies
├── model.pkl                                # Trained model (generated after training)
├── README.md                                # This file
└── .gitignore                               # Git ignore rules
```

## 🔧 Requirements

- Python 3.8+
- pandas 3.0.2
- scikit-learn 1.8.0
- xgboost 3.2.0
- streamlit 1.56.0
- joblib 1.5.3
- numpy 2.4.4

See `requirements.txt` for full dependency list.

## 📊 Dataset Description

The dataset contains vehicle insurance information with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| vehicle_age_years | Numerical | Age of the vehicle in years |
| vehicle_type | Categorical | Type of vehicle (Truck, Hatchback, SUV, etc.) |
| no_of_kilometers_run | Numerical | Number of kilometers the vehicle has run |
| number_of_claims | Numerical | Number of insurance claims made |
| region | Categorical | Geographic region (Urban, Suburban, Rural) |
| insurance_premium | Numerical | **Target variable** - Insurance premium amount |

## 🔍 Model Details

### Feature Engineering
- **Polynomial Features**: Squared and square root transformations
- **Interaction Terms**: Products of key numerical features
- **Data Imputation**: Median for numerical, mode for categorical features
- **Scaling**: StandardScaler for numerical features
- **Encoding**: OneHotEncoder for categorical features

### Training Approach
- **Train-Test Split**: 80-20 split with random_state=42
- **Cross-Validation**: 5-fold CV for model comparison
- **Hyperparameter Optimization**: Pre-tuned for best performance
- **Evaluation Metrics**: RMSE, MAE, R², Train/Test RMSE

## 🌐 Deploy to Streamlit Cloud

1. Push your code to GitHub (already done!)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select this repository
4. Set the main file path to `app.py`
5. Deploy!

The model will be automatically trained on first run.

## 🐛 Troubleshooting

### Model Not Found Error
If you see "Model not found" error:
1. Ensure you're in the project directory
2. Run `python train_model.py` to train the model
3. Verify `model.pkl` exists in the project directory

### Import Errors
```bash
# Ensure all dependencies are installed:
pip install -r requirements.txt --upgrade
```

### Streamlit Not Found
```bash
# Install directly:
pip install streamlit
```

## 📈 Future Improvements

- [ ] Add ensemble methods and stacking
- [ ] Implement hyperparameter tuning with Optuna
- [ ] Add feature importance visualization
- [ ] Create prediction confidence intervals
- [ ] Add more sophisticated feature engineering
- [ ] Implement model versioning and tracking

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Dmitry Tuchin**
- GitHub: [@dmitrytuchin](https://github.com/dmitrytuchin)
- Email: dmitry.tuchin84@gmail.com

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Made with ❤️ using Machine Learning and Streamlit**
