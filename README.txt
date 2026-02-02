================================================================================
                 HACK-EARTH: ENERGY EFFICIENCY MODEL LAB
                            Interactive ML Platform
                              Version 1.0 (2025)
================================================================================

PROJECT OVERVIEW
================================================================================

Hack-Earth is an interactive machine learning laboratory built with Streamlit 
that enables comprehensive analysis, modeling, and prediction of building energy 
efficiency. The platform uses advanced machine learning techniques to model the 
relationship between building physical characteristics and their heating/cooling 
energy requirements (Based on the pre loaded dataset!).

The application is designed for researchers, engineers, Analysts and sustainability 
professionals to:
  • Load and analyze energy efficiency datasets
  • Train and compare multiple ML models for regression and classification tasks
  • Track computational emissions using CodeCarbon
  • Evaluate model performance with detailed diagnostics
  • Perform "what-if" scenario analysis for predictive testing

PROJECT STRUCTURE
================================================================================

Root Directory: /workspaces/Hack-Earth

Files:
  • app.py                    - Main Streamlit application (516 lines)
  • emissions.csv             - Sample CodeCarbon emissions tracking data
  • emissions.csv*.bak        - Backup files of emissions data
  • LICENSE                   - Apache License 2.0
  • README.txt                - This file

KEY FEATURES & FUNCTIONALITY
================================================================================

1. DATA MANAGEMENT
   ────────────────────────────────────────────────────────────────────────
   
   Dataset Sources:
   • Upload CSV: Import your own energy efficiency dataset
   • Generate Synthetic: Pre-loaded 500,000 realistic building records
     (UCI Energy Efficiency Dataset-inspired schema)
   
   Supported Dataset Schema:
     - RelativeCompactness: Building shape compactness ratio (0.5-1.0)
     - SurfaceArea: Total building surface area in m² (50-500)
     - WallArea: Total wall area in m² (20-300)
     - RoofArea: Total roof area in m² (30-200)
     - OverallHeight: Building height in meters (2.5-10)
     - Orientation: Cardinal direction (North, South, East, West)
     - GlazingAreaDistribution: Window distribution pattern
       Options: Uniform, North-heavy, South-heavy, East-heavy, West-heavy
     - BuildingType: Category of building
       Options: Residential, Commercial, Industrial
     - HeatingLoad: Target variable for regression (kWh)
     - CoolingLoad: Alternative target variable (kWh)
   
   Data Preprocessing:
   • Automatic detection and removal of NaN/Inf values
   • Duplicate row elimination
   • Categorical encoding using one-hot encoding (drop_first=True)
   • Optional feature standardization (StandardScaler)
   • Automatic data subsetting for performance optimization


2. TASK MODES
   ────────────────────────────────────────────────────────────────────────
   
   A. REGRESSION MODE
      Predicts continuous heating/cooling load values
      
      Supported Regressors:
      • LinearRegression - Baseline linear model
      • Ridge - L2-regularized linear regression (alpha=1.0 default)
      • Lasso - L1-regularized linear regression (alpha=1.0 default)
      • RandomForestRegressor - 200 trees ensemble (n_jobs=-1)
      • GradientBoostingRegressor - Sequential tree ensemble
      • XGBRegressor - Gradient boosting with advanced features
        (400 estimators, 0.05 learning rate, hist tree_method)
      • LGBMRegressor - Light gradient boosting (if installed)
      • CatBoostRegressor - Categorical gradient boosting (if installed)
      
      Evaluation Metrics:
      • R² Score: Coefficient of determination (0-1, higher is better)
      • MSE: Mean Squared Error (lower is better)
      • RMSE: Root Mean Squared Error (lower is better)
      • Cross-Validation Score (optional 5-fold CV)
   
   B. CLASSIFICATION MODE
      Bins continuous values into discrete classes for classification
      
      Supported Classifiers:
      • LogisticRegression - Linear probabilistic classifier
      • RandomForestClassifier - 300 tree ensemble (n_jobs=-1)
      • GradientBoostingClassifier - Sequential tree classifier
      • XGBClassifier - Boosted tree classifier with mlogloss metric
      • LGBMClassifier - Light gradient boosting classifier
      • CatBoostClassifier - Categorical gradient boosting classifier
      
      Classification Configuration:
      • Target Binning: User selects base column (default: HeatingLoad)
      • Number of Classes: 3-6 bins configurable
      • Binning Strategy:
        - Quantile (pd.qcut): Creates balanced classes by percentiles
        - Uniform (pd.cut): Uses equal-width ranges
      
      Evaluation Metrics:
      • Accuracy: Proportion of correct predictions
      • F1 Score: Weighted harmonic mean of precision and recall
      • Precision: True positives / (true positives + false positives)
      • Recall: True positives / (true positives + false negatives)
      • Confusion Matrix: Visualization of classification performance
      • Cross-Validation Score (optional 5-fold CV)


3. MODEL TRAINING PIPELINE
   ────────────────────────────────────────────────────────────────────────
   
   Process Flow:
   
   1. Data Loading & Validation
      ├─ Load CSV or generate 500k synthetic samples
      ├─ Remove NaN/Inf values
      ├─ Eliminate duplicates
      └─ Validate dataset integrity
   
   2. Feature Engineering
      ├─ Select features (all columns except target)
      ├─ One-hot encode categorical variables
      ├─ Optionally subsample large datasets (default: 200k rows max)
      └─ Build feature matrix X
   
   3. Target Preparation
      ├─ For Regression: Use raw continuous values
      ├─ For Classification:
      │  ├─ Apply jitter to handle duplicate edges (quantile mode)
      │  └─ Bin values into n_bins classes
      └─ Build target vector y
   
   4. Train-Test Split
      ├─ Stratified split for classification tasks
      ├─ Random split for regression tasks
      ├─ Default test_size: 20% (configurable)
      ├─ Controlled randomization via random_state seed
      └─ Generates X_train, X_test, y_train, y_test
   
   5. Feature Scaling (Optional)
      ├─ StandardScaler applied to numeric features only
      ├─ Fit on training data
      ├─ Transform applied to test data
      └─ Benefits: Linear models, LogisticRegression, SVM
   
   6. Model Training
      ├─ Initialize selected models with optimized hyperparameters
      ├─ Fit each model to training data
      ├─ Record training time (seconds)
      └─ Optional: Track CO2 emissions during training (CodeCarbon)
   
   7. Model Inference
      ├─ Generate predictions on test set
      ├─ Record inference time (seconds)
      └─ Compute performance metrics
   
   8. Optional Cross-Validation
      ├─ 5-fold stratified CV for classification
      ├─ 5-fold CV with R² scoring for regression
      ├─ Compute mean CV score
      └─ Compute std of CV scores (shown in tooltip)
   
   9. Results Aggregation
      ├─ Compile all metrics into results dictionary
      ├─ Sort by primary metric (R² or Accuracy)
      ├─ Identify and highlight best-performing model
      └─ Display comparative results table


4. EMISSIONS TRACKING & SUSTAINABILITY
   ────────────────────────────────────────────────────────────────────────
   
   CodeCarbon Integration:
   • Automatically measures CO₂ emissions during model training
   • Tracks resource consumption (CPU, GPU, RAM power draw)
   • Records location-aware grid emissions data
   
   Metrics Tracked:
   • Duration: Training time in seconds
   • Emissions: CO₂ equivalent in kilograms
   • Emissions Rate: kg CO₂ per second
   • CPU/GPU/RAM Power: Power consumption in watts
   • Energy Consumed: Total energy in kWh
   • Location Data: Country, region, cloud provider (if applicable)
   • Hardware Info: CPU model, GPU count, RAM size, Python version
   
   Stored Data:
   • emissions.csv: Central log of all training runs
   • Columns: 42 including timestamp, project_name, run_id, experiment_id
   
   Example Entry (sample from emissions.csv):
   Timestamp: 2026-02-02T06:08:26
   Emissions: 2.24e-06 kg CO₂
   Emissions Rate: 2.08e-06 kg CO₂/sec
   Duration: 1.076 seconds
   CPU Power: 28.00 W
   GPU Power: 0.0 W
   RAM Power: 10.0 W


5. DIAGNOSTIC VISUALIZATIONS
   ────────────────────────────────────────────────────────────────────────
   
   Regression Diagnostics:
   • Residual Distribution Plot
     - Histogram of (y_true - y_pred)
     - Indicates bias and variance in predictions
     - Ideally centered near zero with normal distribution
   
   • True vs Predicted Scatter Plot
     - Scatter plot: y_true (x-axis) vs y_pred (y-axis)
     - Sample: 10,000 random test points for clarity
     - Ideal: Points cluster along y=x diagonal
     - Deviations indicate systematic prediction errors
   
   Classification Diagnostics:
   • Confusion Matrix Heatmap
     - True label (y-axis) vs Predicted label (x-axis)
     - Diagonal entries show correct predictions
     - Off-diagonal show misclassifications
     - Color intensity: Frequency of predictions
   
   • Class Distribution Bar Plot
     - Histogram of test set class labels
     - Shows class balance/imbalance
     - Important for interpreting classification metrics


6. SCENARIO TESTER: WHAT-IF ANALYSIS (Still under Developmental Phase!)
   ────────────────────────────────────────────────────────────────────────
   
   Interactive prediction interface for hypothetical building scenarios
   
   Input Parameters:
   • RelativeCompactness (slider)
     - Range: Data min-max with 5% padding
     - Default: 0.8
     - Impact: Higher values = more compact = better efficiency
   
   • SurfaceArea (slider)
     - Range: Data min-max with 5% padding
     - Default: 220.0 m²
     - Impact: Larger area = higher heating/cooling demand
   
   • WallArea (slider)
     - Range: Data min-max with 5% padding
     - Default: 130.0 m²
     - Impact: More wall area = more thermal loss potential
   
   • RoofArea (slider)
     - Range: Data min-max with 5% padding
     - Default: 95.0 m²
     - Impact: Larger roof = more solar gain/loss
   
   • OverallHeight (slider)
     - Range: Data min-max with 5% padding
     - Default: 3.2 m
     - Impact: Taller buildings = different pressure zones
   
   • Orientation (dropdown)
     - Options: North, South, East, West
     - Impact: Affects solar heat gain patterns
   
   • GlazingAreaDistribution (dropdown)
     - Options: Uniform, North-heavy, South-heavy, East-heavy, West-heavy
     - Impact: Window orientation affects heating/cooling loads
   
   • BuildingType (dropdown)
     - Options: Residential, Commercial, Industrial
     - Impact: Different usage patterns and efficiency standards
   
   Processing Pipeline:
   1. Create single-row DataFrame with input values
   2. Apply same one-hot encoding as training data
   3. Align columns to match training feature set
   4. Apply StandardScaler (if used during training)
   5. Generate predictions from all trained models
   6. Display comparative predictions in table format


7. MODEL HYPERPARAMETERS
   ────────────────────────────────────────────────────────────────────────
   
   Linear Models:
   • LinearRegression: No hyperparameters (fit_intercept=True default)
   • Ridge: alpha=1.0 (default)
   • Lasso: alpha=1.0 (default)
   • LogisticRegression: max_iter=2000
   
   Tree Ensemble Models:
   • RandomForestRegressor: n_estimators=200, n_jobs=-1, random_state=42
   • RandomForestClassifier: n_estimators=300, n_jobs=-1, random_state=42
   • GradientBoostingRegressor: random_state=42, other defaults
   • GradientBoostingClassifier: random_state=42
   
   Advanced Boosting Models:
   • XGBRegressor:
     - n_estimators=400
     - learning_rate=0.05
     - subsample=0.7 (70% sample ratio per iteration)
     - colsample_bytree=0.8 (80% feature ratio per iteration)
     - tree_method="hist" (Fast GPU-accelerated histogram)
     - random_state=42
   
   • XGBClassifier:
     - Same as XGBRegressor
     - eval_metric="mlogloss" (Multi-class loss)
   
   • LGBMRegressor/LGBMClassifier:
     - Uses LightGBM defaults
     - random_state=42
   
   • CatBoostRegressor/CatBoostClassifier:
     - verbose=0 (No logging during training)
     - random_state=42


TECHNICAL ARCHITECTURE
================================================================================

Framework & Libraries:
┌─────────────────────────────────────────────────────────────────────────┐
│                           Application Stack                             │
├─────────────────────────────────────────────────────────────────────────┤
│ Frontend:        Streamlit (Interactive web UI)                         │
│ Backend:         Python 3.12                                            │
│ Data Processing: pandas, NumPy                                          │
│ ML Algorithms:   scikit-learn, XGBoost, LightGBM, CatBoost             │
│ Evaluation:      sklearn.metrics                                        │
│ Visualization:   matplotlib.pyplot                                      │
│ Sustainability:  CodeCarbon (emissions tracking)                        │
└─────────────────────────────────────────────────────────────────────────┘

Required Dependencies:
  • numpy           - Numerical computing
  • pandas          - Data manipulation and analysis
  • scikit-learn    - Core ML algorithms and metrics
  • streamlit       - Web UI framework
  • matplotlib      - Plotting and visualization

Optional Dependencies:
  • xgboost         - Extreme Gradient Boosting (XGB models)
  • lightgbm        - Light Gradient Boosting (LGBM models)
  • catboost        - Categorical Boosting (CatBoost models)
  • codecarbon      - Carbon emissions tracking


ALGORITHM DETAILS
================================================================================

Regression Models Overview:

1. LINEAR REGRESSION
   • Equation: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
   • Uses: Ordinary Least Squares (OLS) optimization
   • Pros: Interpretable, fast, baseline model
   • Cons: Assumes linear relationship, sensitive to outliers

2. RIDGE REGRESSION (L2 Regularization)
   • Equation: min ||y - Xβ||² + λ||β||²
   • Lambda (α): Controls regularization strength
   • Pros: Handles multicollinearity, prevents overfitting
   • Cons: Shrinks all coefficients (doesn't zero out)

3. LASSO REGRESSION (L1 Regularization)
   • Equation: min ||y - Xβ||² + λ||β||
   • Lambda (α): Controls regularization strength
   • Pros: Feature selection (zeros out unimportant features)
   • Cons: Unstable with collinear features

4. RANDOM FOREST REGRESSOR
   • Ensemble: 200 decision trees (n_estimators=200)
   • Split Strategy: Greedy search for best split
   • Pros: Handles non-linearity, feature interactions, robust
   • Cons: Less interpretable, prone to overfitting without tuning

5. GRADIENT BOOSTING REGRESSOR
   • Sequential: Builds trees to correct previous errors
   • Loss: Squared error (default)
   • Pros: Excellent predictive power, handles interactions
   • Cons: Computationally intensive, hyperparameter-sensitive

6. XGBOOST REGRESSOR
   • Algorithm: Second-order gradient boosting with regularization
   • Tree Method: Histogram-based (fast approximation)
   • Regularization: L1/L2 penalties on leaf weights
   • Learning Rate: 0.05 (slower, more robust learning)
   • Subsample: 70% of rows per iteration (reduces variance)
   • Colsample: 80% of features per iteration
   • Pros: State-of-art performance, handles sparse data well
   • Cons: Complex to tune, slow predictions if many trees

7. LIGHTGBM REGRESSOR
   • Algorithm: Leaf-wise tree building (vs level-wise)
   • Split Criterion: Greedy feature selection
   • Pros: Faster training, memory efficient, handles large datasets
   • Cons: Prone to overfitting with small datasets

8. CATBOOST REGRESSOR
   • Algorithm: Gradient boosting with categorical features
   • Native Handling: Categorical variables without encoding
   • Ordered Boosting: Prevents overfitting (special mechanism)
   • Pros: Native categorical support, strong baseline, fast GPU
   • Cons: Less flexibility than XGBoost

Classification Models Overview:

1. LOGISTIC REGRESSION
   • Equation: P(y=1|x) = 1 / (1 + e^(-β·x))
   • Loss: Binary/Multi-class cross-entropy
   • Pros: Interpretable probabilities, fast, good baseline
   • Cons: Assumes linear decision boundaries

2. RANDOM FOREST CLASSIFIER
   • Ensemble: 300 decision trees (n_estimators=300)
   • Voting: Majority class vote from all trees
   • Pros: Handles non-linear boundaries, robust
   • Cons: Overfitting risk, black-box predictions

3. GRADIENT BOOSTING CLASSIFIER
   • Sequential: Builds trees to correct misclassified samples
   • Loss: Log loss (cross-entropy)
   • Pros: Strong performance, feature importance ranking
   • Cons: Sensitive to hyperparameters, slow training

4. XGBOOST CLASSIFIER
   • Multi-class Loss: mlogloss (multinomial cross-entropy)
   • Tree Method: Histogram-based approximation
   • Optimization: Second-order gradient descent (Newton's method)
   • Pros: Fast convergence, best-in-class performance
   • Cons: Complex tuning, extensive hyperparameter space

5. LIGHTGBM CLASSIFIER
   • Categorical Features: Native support without encoding
   • Leaf Growth: Leaf-wise (lower loss improvement)
   • Pros: Fastest training, memory efficient
   • Cons: Overfitting on small datasets, less stable

6. CATBOOST CLASSIFIER
   • Special Treatment: Categorical features optimized internally
   • Bootstrap: Bayesian bootstrap for regularization
   • Pros: Robust defaults, best for categorical data
   • Cons: Less flexible, slower predictions


DATA SYNTHETIC GENERATION ALGORITHM
================================================================================

Function: generate_synthetic_500k(seed=42, n_rows=500_000)

Purpose: Creates realistic building energy efficiency dataset mimicking UCI dataset

Process:

1. INITIALIZE RANDOM NUMBER GENERATOR
   ├─ Seed: 42 (reproducibility)
   └─ Generator: NumPy 2.0+ default_rng (PCG64 algorithm)

2. GENERATE CONTINUOUS FEATURES
   
   RelativeCompactness (Uniform Distribution)
   ├─ Range: 0.5 to 1.0
   ├─ Simulates: Building shape efficiency (sphere=1.0)
   └─ Rounding: 3 decimal places
   
   SurfaceArea (Uniform Distribution)
   ├─ Range: 50 to 500 m²
   ├─ Simulates: Total external surface (affects heat loss)
   └─ Rounding: 2 decimal places
   
   WallArea (Uniform Distribution)
   ├─ Range: 20 to 300 m²
   ├─ Simulates: Vertical opaque surface
   └─ Rounding: 2 decimal places
   
   RoofArea (Uniform Distribution)
   ├─ Range: 30 to 200 m²
   ├─ Simulates: Horizontal surface (solar/sky radiation)
   └─ Rounding: 2 decimal places
   
   OverallHeight (Uniform Distribution)
   ├─ Range: 2.5 to 10 m
   ├─ Simulates: Number of stories effect
   └─ Rounding: 2 decimal places

3. GENERATE CATEGORICAL FEATURES
   
   Orientation (Categorical)
   ├─ Values: North, South, East, West
   └─ Distribution: Uniform random selection
   
   GlazingAreaDistribution (Categorical)
   ├─ Values: Uniform, North-heavy, South-heavy, East-heavy, West-heavy
   └─ Distribution: Uniform random selection
   
   BuildingType (Categorical)
   ├─ Values: Residential, Commercial, Industrial
   └─ Distribution: Uniform random selection

4. GENERATE HEATING LOAD TARGET (kWh)
   
   Formula Components:
   base = 8 kWh
   
   Compactness Effect:
   ├─ Coefficient: 35
   ├─ Relationship: (1.05 - RelativeCompactness)
   ├─ Logic: Lower compactness = more surface = higher losses
   └─ Impact Range: ±17.5 kWh
   
   Surface Area Effect:
   ├─ Coefficient: 0.015
   ├─ Relationship: (SurfaceArea - 200)
   └─ Impact Range: ±4.5 kWh
   
   Wall Area Effect:
   ├─ Coefficient: 0.01
   ├─ Relationship: (WallArea - 120)
   └─ Impact Range: ±1.8 kWh
   
   Roof Area Effect:
   ├─ Coefficient: 0.02
   ├─ Relationship: (RoofArea - 90)
   └─ Impact Range: ±2.2 kWh
   
   Height Effect:
   ├─ Coefficient: 0.5
   ├─ Relationship: (OverallHeight - 3)
   └─ Impact Range: ±3.5 kWh
   
   Noise:
   ├─ Distribution: Normal(μ=0, σ=4)
   ├─ Represents: Unmeasured factors, measurement error
   └─ Impact: Random variation up to ±12 kWh (95% bounds)
   
   Final: Round to 2 decimals

5. GENERATE COOLING LOAD TARGET (kWh)
   
   Formula Components:
   base = 10 kWh (higher baseline due to HVAC efficiency)
   
   Compactness Effect:
   ├─ Coefficient: 28
   ├─ Relationship: (RelativeCompactness - 0.7)
   ├─ Logic: Compact buildings harder to cool (less mass for thermal storage)
   └─ Impact Range: ±8.4 kWh
   
   Surface Area Effect:
   ├─ Coefficient: 0.012
   └─ Relationship: (SurfaceArea - 200)
   
   Wall Area Effect:
   ├─ Coefficient: 0.008
   └─ Relationship: (WallArea - 120)
   
   Roof Area Effect:
   ├─ Coefficient: 0.015
   └─ Relationship: (RoofArea - 90)
   
   Height Effect:
   ├─ Coefficient: 0.3
   └─ Relationship: (OverallHeight - 3)
   
   Solar Gain Effect (South/West Orientation):
   ├─ South-heavy: +3.0 kWh (maximum solar gain)
   ├─ West-heavy: +1.5 kWh (afternoon sun)
   └─ Other: 0 kWh (North-heavy, North, East, uniform)
   
   Noise:
   ├─ Distribution: Normal(μ=0, σ=4)
   └─ Impact: Random variation
   
   Final: Round to 2 decimals


USER INTERFACE WALKTHROUGH
================================================================================

Sidebar Organization:

┌─ Section 1: Data
│  ├─ Choose source: Upload CSV / Generate synthetic (500k rows)
│  ├─ Display stats: Row count, column count
│  └─ Preview option: Show first 10 rows of data
│
├─ Section 2: Task & Target
│  ├─ Select task: Regression or Classification
│  ├─ Regression target: Choose from numeric columns
│  └─ Classification:
│     ├─ Select base column for binning
│     ├─ Number of bins (3-6)
│     └─ Binning strategy: Quantile vs Uniform
│
├─ Section 3: Split & Sample
│  ├─ Test size (%): 10-50 range (default 20%)
│  ├─ Random seed: Control reproducibility (default 42)
│  └─ Max rows to subsample: Limit training data (default 200k)
│
├─ Section 4: Models & Options
│  ├─ Select models: Multi-select from available models
│  ├─ Scale features: Toggle StandardScaler (default True)
│  ├─ Cross-validation: Toggle 5-fold CV (default False)
│  └─ Track emissions: Toggle CodeCarbon tracking (default True)
│
└─ Section 5: Run
   └─ 🚀 Train & Compare button: Starts full pipeline

Main Content Area:

When no data loaded:
├─ Information message: "Upload a CSV or generate a dataset..."
└─ Application halts until data available

After clicking "Train & Compare":

1. COMPARATIVE RESULTS TABLE
   ├─ Model column: Model name
   ├─ Performance metrics:
   │  ├─ Regression: R2, MSE, RMSE
   │  └─ Classification: Accuracy, F1, Precision, Recall
   ├─ Training & inference time (seconds)
   ├─ CO2 emissions (kg)
   ├─ Cross-validation mean score (if enabled)
   └─ Sorted by primary metric (descending)

2. SUCCESS MESSAGE
   ├─ Text: "Top performer by [METRIC]: [MODEL_NAME]"
   └─ Color: Green checkmark icon

3. DIAGNOSTICS SECTION
   ├─ Regression Mode:
   │  ├─ Left (50%): Residual Distribution Histogram
   │  │  └─ Shows: Distribution of (y_true - y_pred)
   │  └─ Right (50%): True vs Predicted Scatter Plot
   │     ├─ Sample: 10,000 points
   │     └─ Shows: Prediction accuracy visually
   │
   └─ Classification Mode:
      ├─ Left (50%): Confusion Matrix Heatmap
      │  ├─ Shows: True labels vs predictions
      │  └─ Color intensity: Frequency
      └─ Right (50%): Class Distribution Bar Chart
         └─ Shows: Count per class in test set

4. SCENARIO TESTER SECTION
   ├─ 3-column input layout
   │  ├─ Column 1:
   │  │  ├─ RelativeCompactness slider
   │  │  ├─ Orientation dropdown
   │  │  └─ BuildingType dropdown
   │  │
   │  ├─ Column 2:
   │  │  ├─ SurfaceArea slider
   │  │  ├─ GlazingAreaDistribution dropdown
   │  │  └─ WallArea slider
   │  │
   │  └─ Column 3:
   │     ├─ RoofArea slider
   │     └─ OverallHeight slider
   │
   ├─ Prediction Results Table:
   │  ├─ Model name column
   │  ├─ Prediction column (predicted value)
   │  ├─ Optional: Prediction class (classification)
   │  └─ Display all selected models' predictions
   │
   └─ Features: Real-time update on parameter change


PERFORMANCE CHARACTERISTICS
================================================================================

Training Time Estimates (Single Model on 500k rows):

Model                    | CPU Time (s) | Memory (GB) | Notes
────────────────────────┼──────────────┼─────────────┼────────────────────
LinearRegression         | 0.5-1        | 0.2         | Baseline reference
Ridge                    | 0.5-1        | 0.2         | Similar to Linear
Lasso                    | 2-5          | 0.2         | Iterative optimization
LogisticRegression       | 5-10         | 0.3         | Multi-class increases time
RandomForestRegressor    | 10-20        | 2-3         | 200 trees, parallelized
RandomForestClassifier   | 15-25        | 2.5-3.5     | 300 trees, parallelized
GradientBoostingReg      | 15-30        | 1-2         | Sequential tree growth
GradientBoostingCls      | 20-40        | 1-2         | Slower for multi-class
XGBRegressor             | 5-15         | 1-2         | Histogram method faster
XGBClassifier            | 8-20         | 1-2         | Multi-class loss slower
LGBMRegressor            | 3-10         | 0.5-1       | Fastest tree method
LGBMClassifier           | 4-12         | 0.5-1       | Leaf-wise growth efficient
CatBoostRegressor        | 8-15         | 1-1.5       | Categorical handling
CatBoostClassifier       | 10-20        | 1-1.5       | Bootstrap optimization

Inference Time per Sample:
• Linear models: <1 ms (microseconds actually)
• Tree-based models: 0.1-1 ms per sample
• Ensembles scale with number of trees

CO₂ Emissions Estimates (Typical):
• Small model (Linear): 1e-7 to 1e-6 kg CO₂
• Medium model (Random Forest): 1e-6 to 1e-5 kg CO₂
• Large model (XGBoost 500k rows): 1e-5 to 1e-4 kg CO₂


CODE QUALITY & ERROR HANDLING
================================================================================

Safe Library Imports:
• Optional models (XGBoost, LightGBM, CatBoost) fail gracefully
• safe_import() function: Returns None if library missing
• No application crash if optional dependency unavailable
• User selects from available models only

Data Validation:
• NaN/Inf detection and removal: df.replace([np.inf, -np.inf], np.nan)
• Duplicate elimination: df.drop_duplicates()
• Feature-target alignment: Automatic removal of target from features
• Categorical encoding robustness: drop_first=True prevents multicollinearity

Error Recovery:
• Emissions tracker: try/except wrapper allows operation without CodeCarbon
• Model training: Graceful degradation if model unavailable
• Scenario prediction: Column alignment before prediction
• Scaler application: Conditional checks for scaler existence

State Management:
• Streamlit @st.cache_data: Caches synthetic data generation
  └─ Prevents regeneration on reruns (significant speedup)
• Session state: Implicit caching of uploaded files
• Widget values: Persist across application reruns


INSTALLATION & SETUP
================================================================================

Prerequisites:
• Python 3.10+ (tested with 3.12.1)
• pip package manager
• Virtual environment (recommended)

Step 1: Clone Repository
$ git clone https://github.com/RoshanNaidu/Hack-Earth.git
$ cd Hack-Earth

Step 2: Create Virtual Environment (Recommended)
$ python3 -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install Required Dependencies
$ pip install -r requirements.txt

Or manually:
$ pip install numpy pandas scikit-learn streamlit matplotlib

Step 4: Install Optional Dependencies (Recommended)
$ pip install xgboost lightgbm catboost codecarbon

Step 5: Run Application
$ streamlit run app.py

Step 6: Access Web Interface
Default: http://localhost:8501
Browser should open automatically; if not, visit the URL above.


USAGE EXAMPLES
================================================================================

EXAMPLE 1: Quick Baseline Model Comparison
─────────────────────────────────────────

1. Start application: streamlit run app.py
2. In Sidebar:
   ├─ Data: Click "Generate synthetic (500k rows)"
   ├─ Task: Select "Regression"
   ├─ Target: HeatingLoad (default)
   ├─ Test Size: 20%
   ├─ Max rows: 200,000 (faster)
   ├─ Models: Select "LinearRegression", "Ridge", "RandomForestRegressor"
   ├─ Scale features: ☑ (checked)
   ├─ CV: ☐ (unchecked - speed priority)
   └─ Click: 🚀 Train & Compare

Expected Results:
• LinearRegression R²: ~0.85
• Ridge R²: ~0.85
• RandomForestRegressor R²: ~0.95
• Total time: ~30 seconds
• CO₂: ~0.0001 kg


EXAMPLE 2: Comprehensive Model Evaluation with Cross-Validation
────────────────────────────────────────────────────────────────

1. Start application
2. In Sidebar:
   ├─ Data: Upload your custom energy_data.csv
   ├─ Task: Select "Classification"
   ├─ Base column: HeatingLoad
   ├─ Bins: 4 (create 4 efficiency classes)
   ├─ Binning: Quantile (balanced classes)
   ├─ Test Size: 15%
   ├─ Max rows: 500,000 (use all data)
   ├─ Models: Select all available
   ├─ Scale: ☑
   ├─ CV: ☑ (enable 5-fold)
   ├─ Emissions: ☑
   └─ Click: 🚀 Train & Compare

Results Section:
• Detailed comparison of 6+ classifiers
• CV scores show generalization performance
• Confusion matrix reveals misclassification patterns
• Emissions tracking shows computational cost

Time estimate: 2-5 minutes (depending on dataset size)


EXAMPLE 3: What-If Building Optimization Analysis
──────────────────────────────────────────────────

1. Run initial training (any mode)
2. In Scenario Tester section, adjust parameters:

   Building A: Current State
   ├─ RelativeCompactness: 0.75
   ├─ SurfaceArea: 250 m²
   ├─ WallArea: 150 m²
   ├─ RoofArea: 100 m²
   ├─ Height: 5 m
   ├─ Orientation: South
   ├─ Glazing: Uniform
   └─ Type: Commercial

   Building B: Optimized (Reduce heating load)
   ├─ RelativeCompactness: 0.95 (↑ more compact)
   ├─ SurfaceArea: 200 m² (↓ smaller)
   ├─ WallArea: 100 m² (↓ less surface)
   ├─ RoofArea: 80 m² (↓ smaller)
   ├─ Height: 4 m (↓ shorter)
   ├─ Orientation: North (↓ less solar gain)
   ├─ Glazing: North-heavy (↓ minimized)
   └─ Type: Commercial

Interpretation:
• Best model predictions for scenario A vs B
• Quantify efficiency improvements
• Guide architectural design decisions


TROUBLESHOOTING
================================================================================

Issue: "No module named 'xgboost'"
Solution: Models gracefully skip; install optional: pip install xgboost

Issue: "Code Carbon tracker error"
Solution: App continues without emissions tracking; install: pip install codecarbon

Issue: "Memory error on 500k rows"
Solution: Reduce max_rows in sidebar (default 200k) or use smaller dataset

Issue: "Train time very long"
Solution: 
  ├─ Reduce max_rows
  ├─ Disable cross-validation (uncheck CV checkbox)
  └─ Select fewer models

Issue: "Poor model performance"
Diagnosis:
  ├─ Check residual plot for systematic bias
  ├─ Review confusion matrix for specific misclassifications
  ├─ Consider feature engineering
  ├─ Try different binning strategy (quantile vs uniform)
  └─ Increase random_state for reproducibility

Issue: "Predictions seem unrealistic"
Solution:
  ├─ Verify input ranges match training data
  ├─ Check that features scaled correctly
  ├─ Review scenario inputs in Scenario Tester
  └─ Compare against best-performing model


FILE SPECIFICATIONS
================================================================================

app.py (Main Application)
├─ Size: 516 lines
├─ Language: Python 3.10+
├─ Key Sections:
│  ├─ Imports (lines 1-45)
│  ├─ Helper functions (lines 46-200)
│  ├─ Sidebar UI (lines 200-320)
│  ├─ Data preprocessing (lines 320-410)
│  ├─ Model training (lines 410-500)
│  ├─ Results visualization (lines 500-516)
│  └─ Scenario testing (embedded in visualization)
├─ Functions:
│  ├─ safe_import() - Safely load optional libraries
│  ├─ get_emissions_tracker() - Initialize CodeCarbon
│  ├─ generate_synthetic_500k() - Create synthetic dataset
│  ├─ build_model_zoo() - Instantiate all models
│  ├─ evaluate_model() - Train and evaluate single model
│  └─ range_for() - Get slider ranges from data
├─ Decorators:
│  └─ @st.cache_data - Cache expensive operations
└─ Dependencies: 28 external libraries

emissions.csv (Emissions Log)
├─ Format: CSV (comma-separated)
├─ Rows: Variable (appended after each training run)
├─ Columns: 42 fields
├─ Key Columns:
│  ├─ timestamp: ISO 8601 format (UTC)
│  ├─ project_name: "EnergyEfficiencyApp"
│  ├─ run_id: Unique ID (UUID4)
│  ├─ duration: Training time in seconds
│  ├─ emissions: CO₂ in kg
│  ├─ cpu_power: CPU power draw in watts
│  ├─ gpu_power: GPU power draw in watts
│  ├─ ram_power: RAM power draw in watts
│  ├─ energy_consumed: Total energy in kWh
│  ├─ country_name: Location name
│  ├─ cpu_model: Processor model string
│  ├─ python_version: Python version string
│  └─ [36 more fields...]
├─ Example Entry:
│  timestamp: 2026-02-02T06:08:26
│  emissions: 2.2414e-06 kg
│  duration: 1.0757 seconds
└─ Purpose: Track environmental impact, audit model training

*.bak files (Backup Copies)
├─ Format: CSV backup archives
├─ Purpose: Recovery and version history
├─ Note: Can be deleted safely (backups only)


LICENSE INFORMATION
================================================================================

License Type: Apache License 2.0
Full Text: See LICENSE file

Key Points:
• Open-source software
• Permissive free use
• Include license in distributions
• No warranty provided
• User assumes all responsibility
• Attribution appreciated but not required
• Can modify and redistribute


CONTRIBUTING & DEVELOPMENT
================================================================================

Repository: github.com/RoshanNaidu/Hack-Earth
Current Branch: main
Default Branch: main

Future Enhancement Ideas:
• Deploying on Deep Learning Models (Neural Networks)
• Model explanation features (SHAP values)
• Hyperparameter optimization (Optuna/Hyperopt)
• Ensemble stacking/voting mechanisms
• Feature importance visualizations
• Time-series forecasting (if temporal data)
• Database integration (SQLite/PostgreSQL)
• Multi-GPU acceleration
• Mobile-friendly responsive design
• Model persistence (save/load trained models)
• API endpoint exposure
• Real-time deployment capabilities
• Batch prediction interface


DEPLOYMENT CONSIDERATIONS
================================================================================

Streamlit Sharing:
$ streamlit share run app.py
  ├─ Host for free on Streamlit Cloud
  ├─ Requires GitHub connection
  └─ Auto-deployed from repo

Docker Deployment:
$ docker build -t hack-earth .
$ docker run -p 8501:8501 hack-earth
  ├─ Create Dockerfile (example below)
  └─ Container isolation + reproducibility

Dockerfile Template:
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

Environment Variables:
• STREAMLIT_SERVER_PORT: 8501 (default)
• STREAMLIT_SERVER_ADDRESS: localhost (default)
• PYTHONUNBUFFERED: 1 (logs without buffering)


PERFORMANCE OPTIMIZATION TIPS
================================================================================

For Faster Training:

1. SUBSAMPLE DATA
   └─ Set max_rows to 50,000-100,000 for rapid experiments
   └─ Full 500k only needed for final evaluation

2. DISABLE FEATURES
   ├─ Turn off cross-validation for speed
   ├─ Disable emissions tracking
   └─ Select fewer models to compare

3. FEATURE SELECTION
   ├─ Standardize numeric features (faster convergence)
   ├─ One-hot encoding is done automatically
   └─ Consider dropping low-variance features

4. PARALLELIZATION
   └─ Already enabled: n_jobs=-1 on RandomForest/GradientBoosting
   └─ Uses all CPU cores available

5. CACHING
   └─ Synthetic data generation cached automatically
   └─ Subsequent reruns load instantly

6. HARDWARE
   ├─ Use GPU: XGBoost tree_method="gpu_hist" (if NVIDIA GPU)
   ├─ More RAM: Reduce feature dimensionality
   └─ More cores: Parallel models benefit


CONCLUSION & QUICK START
================================================================================

Quick Start (30 seconds):

1. Clone/download: Hack-Earth repository
2. Terminal: cd Hack-Earth
3. Terminal: pip install -r requirements.txt
4. Terminal: streamlit run app.py
5. Browser: Opens http://localhost:8501 automatically
6. Sidebar: Generate synthetic dataset (500k rows)
7. Sidebar: Select 3-4 models
8. Click: 🚀 Train & Compare
9. View: Results, diagnostics, scenarios

Key Takeaways:
✓ End-to-end ML platform in single Python file (516 lines)
✓ Supports 8+ regression models, 6+ classification algorithms
✓ Automatically tracks computational carbon emissions
✓ Interactive "what-if" scenario analysis interface
✓ Comprehensive model evaluation and visualization
✓ Graceful handling of optional dependencies
✓ Production-ready code with error handling
✓ Scalable to 500,000+ rows with subsampling support

For detailed analysis: Review app.py source code with inline comments

Questions or Issues: github.com/RoshanNaidu/Hack-Earth/issues

================================================================================
                              END OF README
================================================================================
Version: 1.0
Last Updated: February 2, 2026
Document Scope: Comprehensive end-to-end project documentation
================================================================================
