# Credit Card Fraud Detection App

A machine learning application for detecting fraudulent credit card transactions using Random Forest classification.

## Project Structure

```
credit-card-fraud-app/
├─ app/
│  ├─ __init__.py              # Package initialization
│  ├─ model_utils.py           # Model training and utility functions
│  ├─ plots.py                 # Visualization utilities
│  ├─ main_app.py              # Main Streamlit application
│  └─ assets/
│      └─ cardshield.png       # Logo (add your own)
├─ data/
│  └─ sample_creditcard.csv    # Sample dataset
├─ models/
│  └─ rf_model.pkl             # Trained model (generated after training)
├─ requirements.txt            # Python dependencies
└─ README_RUN.md              # This file
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd credit-card-fraud-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Navigate to the app directory:**
   ```bash
   cd app
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run main_app.py
   ```

3. **Open your browser:**
   The app will automatically open at `http://localhost:8501`

## Features

### 🏠 Home Page
- Overview of the application
- Instructions on how to use the system

### 🚀 Train Model
- Upload a CSV file with credit card transaction data
- Configure training parameters
- Train a Random Forest model
- Model is automatically saved to `models/rf_model.pkl`

### 🔮 Make Predictions
- Load a trained model
- Upload new transaction data for prediction
- Get fraud probability scores
- Download predictions as CSV

### 📊 Model Evaluation
- Evaluate model performance on test data
- View confusion matrix
- Analyze ROC curve and precision-recall curve
- Get detailed classification metrics

## Data Format

Your CSV file should contain:
- Transaction features (numerical columns)
- Target column indicating fraud (0 = Legitimate, 1 = Fraud)

Example columns:
```
Amount, Time, V1, V2, ..., VN, Class
```

## Sample Dataset

A sample credit card dataset can be found in `data/sample_creditcard.csv`. You can use this to test the application.

## Adding Your Logo

1. Place your logo image file (PNG, JPG, etc.) in `app/assets/`
2. Name it `cardshield.png` or update the logo path in `main_app.py`

## Model Details

- **Algorithm:** Random Forest Classifier
- **Features:** Automatically detected from uploaded data
- **Scaling:** StandardScaler for feature normalization
- **Hyperparameters:** 100 trees, balanced class weights

## Dependencies

- **streamlit:** Web framework for ML apps
- **pandas:** Data manipulation
- **numpy:** Numerical computing
- **scikit-learn:** Machine learning library
- **matplotlib & seaborn:** Visualization

## Troubleshooting

### Model not found error
- Train a model first on the "Train Model" page
- Check that `models/rf_model.pkl` exists

### File upload issues
- Ensure CSV file is properly formatted
- Check that the data contains numerical features

### Memory issues with large datasets
- Consider using a sample of your data
- Increase available system memory

## Future Enhancements

- [ ] Add more ML algorithms (XGBoost, LightGBM)
- [ ] Implement model tuning/hyperparameter optimization
- [ ] Add cross-validation reports
- [ ] Support for real-time predictions via API
- [ ] Model versioning and comparison
- [ ] Feature engineering tools

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions, please refer to the Streamlit documentation or scikit-learn documentation.
