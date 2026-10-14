import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
import warnings

# Models & Evaluation
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

# Settings & Setup
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [12, 6]
sns.set(style="darkgrid")
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

# ==============================================================================
# 1. LOAD DATASETS
# ==============================================================================
# Update paths as needed for your local directory
train = pd.read_csv(r'C:\Users\DELL\webdata\train.csv')
test = pd.read_csv(r'C:\Users\DELL\webdata\test.csv')

# Save and drop ID column
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# ==============================================================================
# 2. OUTLIER REMOVAL
# ==============================================================================
# Remove GrLivArea outliers (>4000 sq ft with SalePrice < 200,000)
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index)

# Target Transformation (Log Transformation for Normality)
train["SalePrice"] = np.log1p(train["SalePrice"])
y_train = train.SalePrice.values

# Concatenate train & test for feature engineering
i_train = train.shape[0]
i_test = test.shape[0]

df_concat = pd.concat((train, test)).reset_index(drop=True)
df_concat.drop(['SalePrice'], axis=1, inplace=True)

# ==============================================================================
# 3. IMPUTE MISSING VALUES
# ==============================================================================
# Categorical features missing -> 'None'
none_cols = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]
for col in none_cols:
    df_concat[col].fillna('None', inplace=True)

# LotFrontage: Impute by Neighborhood median
df_concat["LotFrontage"] = df_concat.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

# Numerical basement & garage fields -> 0
zero_cols = [
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
    'BsmtHalfBath', 'BsmtFullBath', 'MasVnrArea'
]
for col in zero_cols:
    df_concat[col].fillna(0, inplace=True)

# Mode Imputations for Categorical/Discrete fields
df_concat['MSZoning'].fillna('RL', inplace=True)
df_concat['Functional'].fillna('Typ', inplace=True)
df_concat['Electrical'].fillna('SBrkr', inplace=True)
df_concat['KitchenQual'].fillna('TA', inplace=True)
df_concat['Exterior1st'].fillna('TA', inplace=True)
df_concat['Exterior2nd'].fillna('TA', inplace=True)
df_concat['SaleType'].fillna('WD', inplace=True)

# Drop unhelpful columns
df_concat.drop(['Utilities'], axis=1, inplace=True)

# ==============================================================================
# 4. FEATURE ENCODING & CREATION
# ==============================================================================
# Convert numerical features that are actually categorical
df_concat['MSSubClass'] = df_concat['MSSubClass'].astype(str)
df_concat['OverallCond'] = df_concat['OverallCond'].astype(str)
df_concat['YrSold'] = df_concat['YrSold'].astype(str)
df_concat['MoSold'] = df_concat['MoSold'].astype(str)

# LabelEncoding Categorical Columns
cols = (
    'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
    'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual',
    'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure',
    'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley',
    'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold'
)
for c in cols:
    le = LabelEncoder()
    le.fit(list(df_concat[c].values))
    df_concat[c] = le.transform(list(df_concat[c].values))

# Create Total Square Footage Feature
df_concat['TotalSF'] = df_concat['TotalBsmtSF'] + df_concat['1stFlrSF'] + df_concat['2ndFlrSF']

# ==============================================================================
# 5. SKEWNESS & BOX-COX TRANSFORMATION
# ==============================================================================
numeric_feats = df_concat.select_dtypes(include=['int64', 'float64']).columns

# Calculate skewness across numeric features
skewed_feats = df_concat[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew': skewed_feats})
skewness = skewness[abs(skewness['Skew']) > 0.75]

lam = 0.15
for feat in skewness.index:
    df_concat[feat] = boxcox1p(df_concat[feat], lam)

# One-hot encoding dummy variables for remaining categorical variables
df_concat = pd.get_dummies(df_concat)

# Re-split datasets into Train and Test
train = df_concat[:i_train]
test = df_concat[i_train:]

# ==============================================================================
# 6. MODEL DEFINITIONS & CROSS-VALIDATION STRATEGY
# ==============================================================================
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse

# Define Individual Regressors
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=0))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=0))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(
    n_estimators=3000, learning_rate=0.05, max_depth=4,
    max_features='sqrt', min_samples_leaf=15, min_samples_split=10,
    loss='huber', random_state=0
)

model_xgb = xgb.XGBRegressor(
    colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05,
    max_depth=3, min_child_weight=1.7817, n_estimators=2200,
    reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213,
    random_state=0, nthread=-1, verbosity=0
)

model_lgb = lgb.LGBMRegressor(
    objective='regression', num_leaves=5, learning_rate=0.05,
    n_estimators=720, max_bin=55, bagging_fraction=0.8,
    bagging_freq=5, feature_fraction=0.2319, min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11, random_state=0
)

# ==============================================================================
# 7. ENSEMBLE CLASSES (AVERAGING & STACKING)
# ==============================================================================
class AverageModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

# ==============================================================================
# 8. TRAINING & EVALUATION
# ==============================================================================
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# Stacking Model Setup
stacked_averaged_models = StackingAveragedModels(
    base_models=(ENet, GBoost, KRR),
    meta_model=lasso
)

imputer = SimpleImputer(strategy='mean')
train_imputed = imputer.fit_transform(train.values)
test_imputed = imputer.transform(test.values)

# Fit Stacked Regressor
stacked_averaged_models.fit(train_imputed, y_train)
stacked_train_pred = stacked_averaged_models.predict(train_imputed)
stacked_pred = np.expm1(stacked_averaged_models.predict(test_imputed))

# Fit XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))

# Fit LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))

# Weighted Ensemble Prediction
ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15

# ==============================================================================
# 9. SUBMISSION GENERATION
# ==============================================================================
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv', index=False)
print("Submission saved successfully to submission.csv")


# Visualize Model Predictions vs Actual Prices (Training Set)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train, y=stacked_train_pred, alpha=0.6, label="Stacked Model")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label="Perfect Fit")
plt.xlabel("Actual Log Sale Price")
plt.ylabel("Predicted Log Sale Price")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.show()  

