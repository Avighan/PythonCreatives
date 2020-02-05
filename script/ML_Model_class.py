import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from xgboost import XGBRegressor, XGBClassifier
import shap


class MLmodels:
    def __init__(self, df=None, y_column=None):

        if df is not None and y_column is not None:
            self.dataset = df
            self.y = df[[y_column]]
            self.x = df[[col for col in df.columns if col != y_column]]

    def __train_val_split__(self, **kwargs):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, **kwargs)
        return self.train_x, self.train_y, self.test_x, self.test_y

    def select_model_to_run(self, model_select, **kwargs):
        self.models = {
            'LogisticRegression': LogisticRegression,
            'GaussianNB': GaussianNB,
            'MultinomialNB': MultinomialNB,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'NearestNeighbors': NearestNeighbors,
            'RandomForestClassifier': RandomForestClassifier,
            'AdaBoostClassifier': AdaBoostClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'SGDClassifier': SGDClassifier,
            'SVC': SVC,
            'LinearRegression': LinearRegression,
            'SVR':SVR,
            'Ridge':Ridge,
            'BayesianRidge':BayesianRidge,
            'DecisionTreeRegressor':DecisionTreeRegressor,
            'RandomForestRegressor':RandomForestRegressor,
            'SGDRegressor':SGDRegressor,
            'XGBRegressor': XGBRegressor,
            'XGBClassifier': XGBClassifier
        }
        if len(kwargs) > 0:
            self.model = self.models[model_select](**kwargs)
        else:
            self.model = self.models[model_select]()
        return self.model

    def set_train_test(self, train_x=None, train_y=None, test_x=None, test_y=None):
        if train_x is not None:
            self.train_x = train_x
        if train_y is not None:
            self.train_y = train_y
        if test_x is not None:
            self.test_x = test_x
        if test_y is not None:
            self.train_y = train_y

    def model_fit(self, train_x=None, train_y=None, test_x=None, test_y=None, **kwargs):
        if train_x is not None:
            self.train_x = train_x
        if train_y is not None:
            self.train_y = train_y
        if test_x is not None:
            self.test_x = test_x
        if test_y is not None:
            self.train_y = train_y
        return self.model.fit(self.train_x, self.train_y, **kwargs)

    def model_predict(self):
        return self.model.predict(self.test_x)

    def hyperParameterTuning(self, params, cv, n_jobs=-1, model=None):
        from sklearn.model_selection import GridSearchCV
        if model is None:
            grid_search = GridSearchCV(self.model, param_grid=params, cv=cv, verbose=1, n_jobs=n_jobs)
        else:
            grid_search = GridSearchCV(model, param_grid=params, cv=cv, verbose=1, n_jobs=n_jobs)
        modelsearch = grid_search.fit(self.train_x, self.train_y)
        params = grid_search.best_params_
        return params

    ##SHAP addition

    def shap_kernel_explainer(self,train_x=None,custom_model=None,model_type= None):
        if model_type is not None:
            self.model_type = model_type
        if custom_model is not None:
            self.custom_model = custom_model
        if train_x is not None:
            self.train_x = train_x
        if self.model_type not 'Xgboost':
            rf_shap_values = self.shap.KernelExplainer(self.custom_model.predict_proba, self.train_x, link="logit", **kwargs)
        else:
            rf_shap_values = self.shap.TreeExplainer(self.custom_model).shap_values(self.train_x)
        return rf_shap_values





from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix, \
    roc_curve, log_loss, balanced_accuracy_score, hamming_loss, hinge_loss, jaccard_score, explained_variance_score, \
    max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score


class Metrics:
    def __init__(self, y_test, y_pred, type):
        self.y_test = y_test
        self.y_pred = y_pred
        self.type = type

    def select_metrics(self, metrics_sel):
        if self.type == 'Classification':
            self.class_metrics = {
                'Accuracy': accuracy_score,
                'Recall': recall_score,
                'Precision': precision_score,
                'F1score': f1_score,
                'Confusion': confusion_matrix,
                'AUC': roc_auc_score,
                'ROC': roc_curve,
                'logloss': log_loss,
                'balancedaccuracy': balanced_accuracy_score,
                'hammingloss': hamming_loss,
                'hingeloss': hinge_loss,
                'jaccardscore': jaccard_score
            }
            self.metric = self.class_metrics[metrics_sel]
        else:
            self.regr_metrics = {
                'rmse': mean_squared_error,
                'jaccardscore': jaccard_score,
                'mae': mean_absolute_error,
                'mse': mean_squared_error,
                'medianabserror': median_absolute_error,
                'maxerror': max_error,
                'r2score': r2_score,
                'mean_squared_log_error': mean_squared_log_error,
                'explained_variance_score': explained_variance_score
            }
            self.metric = self.regr_metrics[metrics_sel]
            if metrics_sel == 'rmse':
                np.sqrt(self.metric)
        return self.metric

    def metrics_solve(self, **kwargs):
        return self.metric(self.y_test, self.y_pred, **kwargs)


from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, StandardScaler


class Encoders:
    def __init__(self, df, cat_columns):
        self.encoded_values = {}
        self.df = df
        self.cat_columns = cat_columns
        self.num_columns = [col for col in df.columns if col not in cat_columns]

    def select_encoder(self, encoding_type, **kwargs):
        self.encoding_type = encoding_type
        encoders = {
            'LabelEncoder': LabelEncoder(**kwargs),
            'OneHotEncoder': OneHotEncoder(**kwargs)
        }
        self.encoder = encoders[self.encoding_type]
        return self.encoder

    def fit(self, x):
        fitted_encoded = self.encoder.fit(x)
        return fitted_encoded

    def transform(self, col, x):
        return self.encoded_values[col].transform(x)

    def inverse_transform(self, col, x):
        return self.encoded_values[col].inverse_transform(x)

    def compile_encoding(self):
        encode_df = pd.DataFrame()
        for col in self.df.columns:
            if col in self.cat_columns:
                print(col)
                self.encoded_values[col] = self.fit(self.df[col].astype(str))
                encode_df[col] = self.transform(col, self.df[col].astype(str))
            else:
                encode_df[col] = self.df[col]
        return encode_df

    def get_decoding(self, decode_df):
        for col in self.cat_columns:
            decode_df[col] = self.inverse_transform(col, decode_df[col])
        return decode_df


class scaling:

    def __init__(self, dataframe):
        self.df = dataframe
        self.selectedscalar = None

    def select_scalars(self, scalarSelect, **kwargs):
        self.scalars = {'StandardScaler': StandardScaler(**kwargs),
                        'MaxAbsScaler': MaxAbsScaler(**kwargs),
                        'MinMaxScaler': MinMaxScaler(**kwargs)
                        }
        self.selectedscalar = self.scalars[scalarSelect]
        return self.selectedscalar

    def fit(self, **kwargs):
        return self.selectedscalar.fit(self.df, **kwargs)

    def transform(self, **kwargs):
        return self.selectedscalar.transform(self.df, **kwargs)

    def fit_transform(self, **kwargs):
        return self.selectedscalar.fit_transform(self.df, **kwargs)



#Example code to run -
import pdb
load_data = pd.read_excel('./sample_data.xlsx')
cat_columns = [cat_col for cat_col in load_data.columns if load_data[cat_col].dtype==object]
encode = Encoders(df=load_data,cat_columns = cat_columns)
encode.select_encoder(encoding_type='LabelEncoder')
encoded_df = encode.compile_encoding()
modelobj = MLmodels(df=encoded_df, y_column='careval')
train_x, train_y, test_x, test_y = modelobj.__train_val_split__()
models_to_run = ['LogisticRegression','DecisionTreeClassifier','RandomForestClassifier']
for model in models_to_run:
    modelobj.select_model_to_run(model_select = model)
    fitted_model = modelobj.model_fit()
    y_pred = modelobj.model_predict()
    metric = Metrics(test_y, y_pred, type='Classification')
    metrics_to_check = ['Accuracy','Confusion']
    for metriccheck in metrics_to_check:
        metric.select_metrics(metriccheck)
        print(metriccheck)
        print(metric.metrics_solve())




