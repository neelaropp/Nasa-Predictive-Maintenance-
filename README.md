# NASA Turbofan Engine Degradation Prediction

## Project Overview
This project leverages the FD004 dataset from NASA's Turbofan Engine Degradation Simulation Dataset to develop predictive maintenance models. The objective is to predict Remaining Useful Life (RUL) of turbofan engines under varying operating conditions and fault modes. By analyzing engine sensor data, this project aims to enhance fault detection, improve system reliability, and reduce unplanned maintenance costs in aerospace engineering.

## Research Goals
- Develop predictive models for estimating RUL.
- Implement fault detection algorithms for early warning systems.
- Explore deep learning approaches such as LSTMs, GRUs, and CNN-LSTM hybrids for time-series analysis.
- Assess the impact of multiple operating conditions and two fault modes on predictive performance.
- Evaluate model performance using RMSE, MAE, and R² scores.

## Dataset
The **FD004 dataset** from NASA contains:
- 249 training trajectories and 248 testing trajectories.
- 21 sensor readings capturing different engine health parameters.
- Operating conditions and fault modes affecting degradation behavior.

## Methodology
1. **Data Preprocessing**
   - Handling missing values and outliers.
   - Feature engineering to extract informative trends from sensor data.
   - Normalization and scaling for neural network inputs.
   
2. **Exploratory Data Analysis (EDA)**
   - Visualization of sensor trends and degradation patterns.
   - Distribution analysis of failure modes and operational conditions.
   
3. **Modeling Strategy**
   - Traditional regression models: Linear Regression, Random Forest, XGBoost.
   - Deep learning architectures: LSTMs, GRUs, CNN-LSTMs.
   - Comparative analysis of models on test data.
   
4. **Evaluation Metrics**
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - Coefficient of Determination (R²)
   
5. **Expected Impact**
   - Enhanced predictive maintenance strategies.
   - Reduced operational costs and improved safety in aerospace applications.
   - Insights into the most informative sensor readings for degradation analysis.

## Dependencies
To run the project, install the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

## Running the Code
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd nasa-turbofan-prediction
   ```
2. Load and preprocess the dataset using:
   ```bash
   python data_preprocessing.py
   ```
3. Train the predictive model:
   ```bash
   python train_model.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```
5. Visualize results:
   ```bash
   python visualize_results.py
   ```

## Results
The project will generate:
- Graphs illustrating sensor trends and degradation patterns.
- Performance metrics of different machine learning and deep learning models.
- Insights into the best-performing approach for predictive maintenance.

## Future Work
- Incorporating additional feature selection techniques.
- Experimenting with Transformer-based architectures for improved time-series prediction.
- Deploying the model as a real-time monitoring system for predictive maintenance.

## Author
**Neela Ropp**  
Master's Student in Data Science  

## Acknowledgments
This project is based on the **NASA Turbofan Engine Degradation Simulation Dataset**. Special thanks to NASA for making the dataset publicly available for research and development.
