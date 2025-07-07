## Data Sources:
# 1. To access the "Key Emission Enterprises in Carbon Emissions Trading" dataset: visit the global CSMAR website, search for "Carbon Emission," select "Database Result," then choose "Carbon Neutrality Research." After that, click on "Carbon Trading" and select "Daily Trading Information of Emission Rights in the Carbon Market."
# 	 The data is available for the years 2013 and 2017. I have generated synthetic data to extend the dataset, covering the period from 2013 to 2022.
# 2. Green Innovation Peer Effect of Chinese Manufacturing Firms Dataset: Wang, Jing; Zhao, Luyao; Zhu, Ruixue (2022), “The dataset of green innovation peer effect of Chinese manufacturing industry”, Mendeley Data, V1, doi: 10.17632/mp7gg98y47.1

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Define file paths
green_innovation_file = 'datasets/Green Innovation Peer Effect of Chinese Manufacturing Firms Dataset.xlsx'
carbon_emissions_file = 'datasets/Carbon_Emissions_Data.xlsx'

# Load the datasets
green_innovation_data = pd.read_excel(green_innovation_file)
carbon_emissions_data = pd.read_excel(carbon_emissions_file)

# Preview the data to understand its structure
print(green_innovation_data.head())
print(carbon_emissions_data.head())

# Data Preprocessing for Carbon Emissions Data
# Convert 'Trading Date' to datetime and extract year-month for fixed effect
carbon_emissions_data['Trading Date'] = pd.to_datetime(
    carbon_emissions_data['Trading Date'])
carbon_emissions_data['Year-Month'] = carbon_emissions_data[
    'Trading Date'].dt.to_period('M')

# Convert the 'Year-Month' column to string
carbon_emissions_data['Year-Month'] = carbon_emissions_data[
    'Year-Month'].astype(str)

# Rename columns to remove spaces for easier reference in the formula
carbon_emissions_data.columns = carbon_emissions_data.columns.str.replace(
    ' ', '_')

# Explicitly rename 'Year-Month' column to 'Year_Month'
carbon_emissions_data.rename(columns={'Year-Month': 'Year_Month'},
                             inplace=True)

# Clean data: remove rows with missing values in critical columns
carbon_emissions_cleaned = carbon_emissions_data.dropna(subset=[
    'Closing_Price', 'Total_Trading_Turnover', 'City_Name', 'Year_Month'
])

# Check the columns to verify the column names
print(carbon_emissions_cleaned.columns)

# Regression Model: Fixed Effects (including City Name and Year-Month)
model = ols(
    "Closing_Price ~ Total_Trading_Turnover + C(City_Name) + C(Year_Month)",
    data=carbon_emissions_cleaned).fit()

# Display the regression results
print(model.summary())

# 2. Event Study: Compliance Period Shocks:
# Load the datasets
green_innovation_data = pd.read_excel(green_innovation_file)
carbon_emissions_data = pd.read_excel(carbon_emissions_file)

# For testing, mock stock data since actual file isn't found
print("Stock data file not found. Using mock stock data for testing.")
stock_data = pd.DataFrame({
    'Firm_ID':
    range(1, 21),  # 20 observations instead of 5
    'Trading_Date':
    pd.date_range(start='2013-01-01', periods=20, freq='M'),
    'Closing_Price': [20 + i * 0.5 for i in range(20)]  # Mock closing prices
})
stock_data['Trading_Date'] = pd.to_datetime(stock_data['Trading_Date'])

# Step 1: Data Preprocessing

# Carbon Data Preprocessing
carbon_emissions_data['Trading Date'] = pd.to_datetime(
    carbon_emissions_data['Trading Date'])

# Create 'Year-Month' column for better period identification
carbon_emissions_data['Year_Month'] = carbon_emissions_data[
    'Trading Date'].dt.to_period('M')
carbon_emissions_data['Year_Month'] = carbon_emissions_data[
    'Year_Month'].astype(str)
carbon_emissions_data.columns = carbon_emissions_data.columns.str.replace(
    ' ', '_')

# Clean data: remove rows with missing values in critical columns
required_columns = [
    'Closing_Price', 'Total_Trading_Turnover', 'City_Name', 'Year_Month'
]
carbon_emissions_cleaned = carbon_emissions_data.dropna(
    subset=required_columns)

# Ensure Firm_ID exists or create it
carbon_emissions_cleaned[
    'Firm_ID'] = carbon_emissions_cleaned.index + 1  # Example strategy

# Stock Data Preprocessing
print("Stock Data Columns:")
print(stock_data.columns)

# Ensure column names match across datasets
stock_data.rename(columns={"Trading_Date": "Trading Date"}, inplace=True)

# Merge stock and carbon data
merged_data = pd.merge(stock_data,
                       carbon_emissions_cleaned,
                       on="Firm_ID",
                       how="inner")

# Merge with Green Innovation data
if 'Firm_ID' not in green_innovation_data.columns:
    green_innovation_data[
        'Firm_ID'] = green_innovation_data.index + 1  # Or use another method

merged_data = pd.merge(merged_data,
                       green_innovation_data[['Firm_ID', 'GRIN']],
                       on='Firm_ID',
                       how='left')

# Create 'Month' column for feature engineering
merged_data['Month'] = merged_data['Trading_Date'].dt.month

# Rename columns for clarity
merged_data.rename(columns={
    "Closing_Price_x": "Closing_Price_stock",
    "Closing_Price_y": "Closing_Price_carbon"
},
                   inplace=True)

# Step 2: Event Study - Compliance Period Shocks
merged_data['Compliance_Period'] = merged_data['Trading_Date'].apply(
    lambda x: 1 if x.month in [11, 12] else 0)

merged_data['Lead'] = merged_data['Compliance_Period'].shift(-1)
merged_data['Lag'] = merged_data['Compliance_Period'].shift(1)

# Step 3: Regression Model - Ensure columns are correctly referenced
model = ols("Closing_Price_stock ~ Lead + Lag + C(City_Name) + C(Year_Month)",
            data=merged_data).fit(cov_type='HC3')
print(model.summary())

# Step 4: Additional Analysis
if 'GRIN' in merged_data.columns:
    model_grin = ols(
        "Closing_Price_stock ~ Lead + Lag + GRIN + C(City_Name) + C(Year_Month)",
        data=merged_data).fit(cov_type='HC3')
    print(model_grin.summary())
else:
    print("GRIN column not found in the merged data.")

# Step 5: Check for Multicollinearity

# Select only numeric columns for VIF calculation
X = merged_data[['Lead', 'Lag', 'GRIN']]

# Check for columns with zero variance
variances = X.var()
zero_variance_columns = variances[variances == 0].index.tolist()

# Drop columns with zero variance
X = X.drop(columns=zero_variance_columns)

# Check if we have sufficient data for VIF calculation
if X.shape[0] > 1:  # Ensure more than one row exists
    # Add constant to the predictors matrix
    X = add_constant(X)

    # Check for missing values or infinities
    X = X.dropna()  # Removes rows with NaN
    X = X[~X.isin([float("inf"), float("-inf")
                   ]).any(axis=1)]  # Removes rows with inf values

    # Ensure there are sufficient data points for VIF (at least 2 rows)
    if X.shape[0] > 1:
        # Compute VIF
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]

        # Display VIF results
        print("\nVariance Inflation Factors (VIF):")
        print(vif)
    else:
        print("Not enough data for VIF calculation. Need more observations.")
else:
    print(
        "Not enough data for VIF calculation. Ensure you have more than 1 data point."
    )

# 3. Robustness Test:
# a. Placebo Test (For Non-Carbon-Intensive Industries)
# Ensure 'Industry' column exists
if 'Industry' not in merged_data.columns:
    print("Industry column is missing. Creating mock industry classification.")
    merged_data['Industry'] = [
        'Retail' if i % 2 == 0 else 'Carbon-Intensive'
        for i in range(len(merged_data))
    ]

# Apply placebo test to non-carbon-intensive industries (e.g., Retail)
non_carbon_data = merged_data[merged_data['Industry'] == 'Retail']

# Add Year column for fixed effects
non_carbon_data['Year'] = non_carbon_data['Trading_Date'].dt.year

# Add Carbon Trading Activity (CarbonTAct) for the placebo test
non_carbon_data['CarbonTAct'] = non_carbon_data['Trading_Date'].apply(
    lambda x: 1 if x.month in [11, 12] else 0)

# Add Lead, Lag, and Fixed Effects
non_carbon_data['Lead'] = non_carbon_data['CarbonTAct'].shift(-1)
non_carbon_data['Lag'] = non_carbon_data['CarbonTAct'].shift(1)

# Running the Placebo Test Model
placebo_model = ols(
    "Closing_Price_stock ~ CarbonTAct + Lead + Lag + C(City_Name) + C(Year) + GRIN",
    data=non_carbon_data).fit(cov_type='HC3')

# Print summary of the placebo regression model
print(placebo_model.summary())

# Check for multicollinearity for the placebo test
X_placebo = non_carbon_data[['CarbonTAct', 'Lead', 'Lag', 'GRIN']]

# Compute VIF for placebo test variables
variances_placebo = X_placebo.var()
zero_variance_columns_placebo = variances_placebo[variances_placebo ==
                                                  0].index.tolist()

# Drop columns with zero variance
X_placebo = X_placebo.drop(columns=zero_variance_columns_placebo)

# Add constant to the predictors matrix
X_placebo = add_constant(X_placebo)

# Check for missing values or infinities
X_placebo = X_placebo.dropna()  # Removes rows with NaN
X_placebo = X_placebo[~X_placebo.isin(
    [float("inf"), float("-inf")]).any(axis=1)]  # Removes rows with inf values

# Compute VIF for placebo model
vif_placebo = pd.DataFrame()
vif_placebo["Variable"] = X_placebo.columns
vif_placebo["VIF"] = [
    variance_inflation_factor(X_placebo.values, i)
    for i in range(X_placebo.shape[1])
]

# Display VIF results for placebo model
print("\nVariance Inflation Factors (VIF) for Placebo Test:")
print(vif_placebo)

# # 3 .Robustness Test:
# b. Machine Learning (XGBoost)

# Create a target variable (HighCarbonTAct)
# Let's classify "high turnover" vs "low turnover" (example threshold)
merged_data['HighCarbonTAct'] = merged_data['Total_Trading_Turnover'].apply(
    lambda x: 1 if x > merged_data['Total_Trading_Turnover'].median() else 0)

# Define Features and Target
features = ['Month', 'City_Name',
            'GRIN']  # Add any additional relevant features
X = merged_data[features]
y = merged_data['HighCarbonTAct']

# Convert categorical columns (e.g., City_Name) to numeric using one-hot encoding
X = pd.get_dummies(X, columns=['City_Name'], drop_first=True)

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

# Train XGBoost Classifier
model = xgb.XGBClassifier(use_label_encoder=False,
                          eval_metric='logloss')  # Using XGBoost Classifier
model.fit(X_train, y_train)

# Make Predictions and Evaluate Model
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance (Optional)
xgb.plot_importance(model)
plt.show()

## Robustness Checks
# c. Alternate Specifications:
# 1. Replace Dependent Variable:

# Assuming merged_data has been already created and contains all necessary columns.
# Step 1: Define 'CarbonTAct' (if not defined already)
merged_data['CarbonTAct'] = merged_data['Trading_Date'].apply(
    lambda x: 1 if x.month in [11, 12] else 0)

# Step 2: Create Highprcit variable (replace it with actual data if necessary)
# For this example, assuming it's a proxy based on the Total Trading Turnover
merged_data['Highprcit'] = merged_data['Total_Trading_Turnover'].apply(
    lambda x: 1 if x > merged_data['Total_Trading_Turnover'].median() else 0)

# Step 3: Run the regression with Highprcit as the dependent variable
# Ensure we have the correct columns: 'Highprcit', 'CarbonTAct', 'Lead', 'Lag', 'City_Name', 'Year_Month'
model_high_price = ols(
    "Highprcit ~ CarbonTAct + Lead + Lag + C(City_Name) + C(Year_Month)",
    data=merged_data).fit(cov_type='HC3')

# Print the regression results with Highprcit as the dependent variable
print("Regression with Highprcit as the Dependent Variable:")
print(model_high_price.summary())

# Step 4: Alternate Specification (Adding more control variables)
# Adding 'GRIN' to check the robustness of results
model_high_price_alt = ols(
    "Highprcit ~ CarbonTAct + Lead + Lag + C(City_Name) + C(Year_Month) + GRIN",
    data=merged_data).fit(cov_type='HC3')

# Print the alternate specification results
print("\nAlternate Specification Model with GRIN included:")
print(model_high_price_alt.summary())

# Step 5: Check for multicollinearity for the alternative specification
X_alt = merged_data[['CarbonTAct', 'Lead', 'Lag', 'GRIN']]

# Check for columns with zero variance
variances_alt = X_alt.var()
zero_variance_columns_alt = variances_alt[variances_alt == 0].index.tolist()

# Drop columns with zero variance
X_alt = X_alt.drop(columns=zero_variance_columns_alt)

# Add constant to the predictors matrix (for regression intercept)
X_alt = add_constant(X_alt)

# Check for missing values or infinities
X_alt = X_alt.dropna()  # Removes rows with NaN
X_alt = X_alt[~X_alt.isin([float("inf"), float("-inf")
                           ]).any(axis=1)]  # Removes rows with inf values

# Compute VIF for alternative model
vif_alt = pd.DataFrame()
vif_alt["Variable"] = X_alt.columns
vif_alt["VIF"] = [
    variance_inflation_factor(X_alt.values, i) for i in range(X_alt.shape[1])
]

# Display VIF results for the alternative model
print("\nVariance Inflation Factors (VIF) for Alternate Specification:")
print(vif_alt)

## Robustness Checks
# c. Alternate Specifications:
# 2. Expand Fixed Effects:

## Step 2: Check for Multicollinearity for the simplified model
# Exclude non-numeric columns from VIF calculation
X_simple = merged_data[['CarbonTAct',
                        'GRIN']]  # Only numeric columns for VIF calculation

# Ensure no zero variance columns
variances_simple = X_simple.var()
zero_variance_columns_simple = variances_simple[variances_simple ==
                                                0].index.tolist()

# Drop columns with zero variance
X_simple = X_simple.drop(columns=zero_variance_columns_simple)

# Add constant to the predictors matrix
X_simple = add_constant(X_simple)

# Check for missing values or infinities
X_simple = X_simple.dropna()  # Removes rows with NaN
X_simple = X_simple[~X_simple.isin(
    [float("inf"), float("-inf")]).any(axis=1)]  # Removes rows with inf values

# Compute VIF for the simplified model
vif_simple = pd.DataFrame()
vif_simple["Variable"] = X_simple.columns
vif_simple["VIF"] = [
    variance_inflation_factor(X_simple.values, i)
    for i in range(X_simple.shape[1])
]

# Display VIF results
print("\nVariance Inflation Factors (VIF) for Simplified Model:")
print(vif_simple)

# If there are high VIF variables, we can remove them
high_vif_columns_simple = vif_simple[vif_simple['VIF'] >
                                     10]['Variable'].tolist()
print(f"\nRemoving variables with high VIF: {high_vif_columns_simple}")

# Drop high VIF columns
X_simple_cleaned = X_simple.drop(columns=high_vif_columns_simple)

# Step 3: Re-run the simplified model with cleaned data
model_simple_cleaned = ols(
    "Closing_Price_stock ~ CarbonTAct + C(City_Name) + C(Year_Month)",  # Same as before, with C() in formula
    data=merged_data).fit(cov_type='HC3')  # Robust standard errors

# Print summary of the cleaned model
print("\nSimplified Model with Cleaned Variables:")
print(model_simple_cleaned.summary())

# 4. Additions:
# GARCH Model for Volatility:

import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# Assuming merged_data is already created and contains all necessary columns.

# Step 1: Calculate daily stock returns (log returns)
merged_data['Log_Returns'] = np.log(
    merged_data['Closing_Price_stock'] /
    merged_data['Closing_Price_stock'].shift(1))

# Step 2: Add CarbonTAct (Carbon Trading Activity) - already present in merged_data
# CarbonTAct represents the log of carbon trading turnover in region c
merged_data['Log_CarbonTAct'] = np.log(
    merged_data['Total_Trading_Turnover'].apply(
        lambda x: x + 1))  # Adding 1 to avoid log(0)

# Drop rows with NaN values
merged_data = merged_data.dropna(subset=['Log_Returns', 'Log_CarbonTAct'])

# Step 3: GARCH Model for Volatility
# GARCH(1,1) model: volatility as a function of lagged squared returns and lagged volatility

# Prepare the data
returns = merged_data['Log_Returns']  # Daily stock returns
carbon_impact = merged_data['Log_CarbonTAct']  # Log of carbon trading turnover

# Define the GARCH model (including carbon trading impact as an exogenous variable)
garch_model = arch_model(
    returns, vol='Garch', p=1, q=1,
    x=carbon_impact)  # GARCH(1,1) with CarbonTAct as an exogenous variable

# Fit the model
garch_fit = garch_model.fit()

# Step 4: Print model parameters to inspect if CarbonTAct is included
print("\nGARCH Model Parameters:")
print(garch_fit.params)

# Step 5: Extract the coefficient for the exogenous variable (CarbonTAct)
params = garch_fit.params

# Check the parameter names
print("\nModel Parameters:", params)

# Step 6: Accessing the exogenous variable coefficient (if available)
# Print out the names of the coefficients and check if the CarbonTAct coefficient is included
for param_name in params.index:
    print(f"Parameter: {param_name}, Value: {params[param_name]}")

# If 'x1' is the name for the exogenous variable, extract it
if 'x1' in params.index:
    delta = params['x1']
    if delta > 0:
        print(f"Carbon trading increases volatility (δ = {delta:.4f})")
    else:
        print(f"Carbon trading decreases volatility (δ = {delta:.4f})")
else:
    print("Carbon trading impact (δ) coefficient not found.")

# Step 7: Plotting the volatility (conditional variance) over time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(garch_fit.conditional_volatility, label='Volatility (σ)')
ax.set_title('Conditional Volatility (GARCH Model)')
ax.set_xlabel('Time')
ax.set_ylabel('Volatility')
ax.legend()
plt.show()

# 4. Additions:
# Portfolio Optimization

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Example data setup (replace this with actual CSMAR data)
# Assuming we have 'returns' dataframe with stock returns of different firms over time
# Returns data for each firm (daily stock returns)
returns_data = pd.DataFrame({
    'Firm1': np.random.normal(0.0005, 0.02, 1000),
    'Firm2': np.random.normal(0.0007, 0.015, 1000),
    'Firm3': np.random.normal(0.0006, 0.01, 1000),
    'Firm4': np.random.normal(0.0004, 0.018, 1000),
    'Firm5': np.random.normal(0.0008, 0.022, 1000)
})

# Simulate Carbon market shock data (1 for compliance shock, 0 otherwise)
shock_data = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])

# Calculate the expected return for each firm (mean return)
expected_returns = returns_data.mean()

# Calculate the covariance matrix of returns
cov_matrix = returns_data.cov()


# Step 1: Define the portfolio optimization function
def portfolio_optimization(returns, cov_matrix, target_return):
    n = len(returns)

    # Portfolio variance function
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # Portfolio return function
    def portfolio_return(weights):
        return np.dot(weights, returns)

    # Constraint: sum of weights = 1
    def constraint(weights):
        return np.sum(weights) - 1

    # Bounds for weights (all weights >= 0)
    bounds = [(0, 1) for _ in range(n)]

    # Initial guess for weights
    initial_weights = np.ones(n) / n

    # Constraints
    cons = ({'type': 'eq', 'fun': constraint})

    # Optimization objective: minimize variance (risk) for a given target return
    def objective(weights):
        # Return and variance of the portfolio
        port_return = portfolio_return(weights)
        port_variance = portfolio_variance(weights)
        # We want to minimize the variance for the given target return
        return port_variance

    # Optimize portfolio weights
    result = minimize(objective,
                      initial_weights,
                      bounds=bounds,
                      constraints=cons)

    # Get optimal weights
    optimal_weights = result.x
    return optimal_weights


# Step 2: Calculate Sharpe Ratio for the optimized portfolio
def calculate_sharpe_ratio(returns,
                           cov_matrix,
                           optimal_weights,
                           risk_free_rate=0.03):
    # Portfolio expected return
    expected_portfolio_return = np.dot(optimal_weights, returns)

    # Portfolio volatility (standard deviation)
    portfolio_volatility = np.sqrt(
        np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

    # Sharpe ratio formula: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
    sharpe_ratio = (expected_portfolio_return -
                    risk_free_rate) / portfolio_volatility
    return sharpe_ratio


# Step 3: Run the optimization for portfolios during compliance shock (1) and non-compliance (0)
# Separate data based on market shocks
compliance_data = returns_data[shock_data == 1]
non_compliance_data = returns_data[shock_data == 0]

# Expected returns and covariance for compliance and non-compliance periods
compliance_expected_returns = compliance_data.mean()
compliance_cov_matrix = compliance_data.cov()

non_compliance_expected_returns = non_compliance_data.mean()
non_compliance_cov_matrix = non_compliance_data.cov()

# Optimize portfolios for both periods
compliance_optimal_weights = portfolio_optimization(
    compliance_expected_returns, compliance_cov_matrix, target_return=0.0005)
non_compliance_optimal_weights = portfolio_optimization(
    non_compliance_expected_returns,
    non_compliance_cov_matrix,
    target_return=0.0005)

# Calculate Sharpe ratios
compliance_sharpe = calculate_sharpe_ratio(compliance_expected_returns,
                                           compliance_cov_matrix,
                                           compliance_optimal_weights)
non_compliance_sharpe = calculate_sharpe_ratio(non_compliance_expected_returns,
                                               non_compliance_cov_matrix,
                                               non_compliance_optimal_weights)

# Step 4: Compare Sharpe ratios
print(f"Sharpe Ratio during compliance shocks: {compliance_sharpe:.4f}")
print(
    f"Sharpe Ratio during non-compliance shocks: {non_compliance_sharpe:.4f}")

# Step 5: Plot Efficient Frontiers (optional)
# You can plot the efficient frontier for both compliance and non-compliance periods
# But for simplicity, we'll assume the results are as follows
compliance_sharpe_ = compliance_sharpe
non_compliance_sharpe_ = non_compliance_sharpe

plt.bar(['Compliance Shock', 'Non-Compliance'],
        [compliance_sharpe_, non_compliance_sharpe_],
        color=['green', 'red'])
plt.title('Comparison of Sharpe Ratios')
plt.ylabel('Sharpe Ratio')
plt.show()
