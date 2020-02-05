import hyperopt
from hyperopt import tpe
from hpsklearn import HyperoptEstimator, any_classifier,any_preprocessing
import pandas as pd
import numpy as np
import pickle
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
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge,Lasso,ElasticNet
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier
import shap
import pdb
import os

os.environ['OMP_NUM_THREADS'] = "1"

class MLmodels:
    def __init__(self, df=None, y_column=None,hyperparam=False,problem_type='Classification'):
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        if df is not None and y_column is not None:
            self.dataset = df
            self.y = df[[y_column]]
            self.x = df[[col for col in df.columns if col != y_column]]

        self.hyperparam = hyperparam
        self.problem_type = problem_type
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
            'SVR': SVR,
            'Ridge': Ridge,
            'BayesianRidge': BayesianRidge,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'SGDRegressor': SGDRegressor,
            'XGBRegressor': XGBRegressor,
            'XGBClassifier': XGBClassifier,
            'PolynomialFeatures':PolynomialFeatures,
            'Lasso':Lasso,
            'ElasticNet':ElasticNet
        }

    def __train_val_split__(self, **kwargs):
        if hasattr(self, 'x') and hasattr(self, 'y'):
            if self.x is not None and self.y is not None:
                self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, **kwargs)
                return self.train_x, self.train_y, self.test_x, self.test_y

    def select_model_to_run(self, model_select, **kwargs):
        self.model_select = model_select
        if len(kwargs) > 0:
            self.model = self.models[self.model_select](**kwargs)
        else:
            self.model = self.models[self.model_select]()
        return self.model


    def set_model_parameters(self,**kwargs):

        self.model = self.models[self.model_select](**kwargs)
        return self.model

    def set_train_test(self, train_x=None, train_y=None, test_x=None, test_y=None):
        if train_x is not None:
            self.train_x = train_x
        if train_y is not None:
            self.train_y = train_y
        if test_x is not None:
            self.test_x = test_x
        if test_y is not None:
            self.test_y = test_y

    def get_train_test(self):
        return self.train_x, self.train_y, self.test_x, self.test_y

    def model_fit(self, train_x=None, train_y=None, test_x=None, test_y=None, **kwargs):
        if train_x is not None:
            self.train_x = train_x
        if train_y is not None:
            self.train_y = train_y
        if test_x is not None:
            self.test_x = test_x
        if test_y is not None:
            self.train_y = train_y

        if self.train_x is None or self.test_x is None or self.train_y is None or self.test_y is None:
            self.__train_val_split__()
        return self.model.fit(self.train_x, self.train_y, **kwargs)

    def model_predict(self,model=None,X=None):
        if model is None:
            pass
        else:
            self.model = model
        if X is None:
            return self.model.predict(self.test_x)
        else:
            return self.model.predict(X)

    def model_predict_proba(self,model=None,X=None):
        if model is None:
            pass
        else:
            self.model = model
        if X is None:
            return self.model.predict_proba(self.test_x)
        else:
            return self.model.predict_proba(X)

    def hyperParameterTuning(self, params, cv, n_jobs=-1, model=None):
        if self.train_x is None or self.test_x is None or self.train_y is None or self.test_y is None:
            self.__train_val_split__()
        from sklearn.model_selection import GridSearchCV
        if model is None:
            grid_search = GridSearchCV(self.model, param_grid=params, cv=cv, verbose=1, n_jobs=n_jobs)
        else:
            grid_search = GridSearchCV(model, param_grid=params, cv=cv, verbose=1, n_jobs=n_jobs)
        modelsearch = grid_search.fit(self.train_x, self.train_y)
        self.params = grid_search.best_params_
        return self.params

    def compile_modeling(self,onlyexecute=False):
        
        if self.hyperparam == False:
            if self.train_x is None or self.test_x is None or self.train_y is None or self.test_y is None:
                self.__train_val_split__()
            fitted_model = self.model_fit()
        else:
            if self.train_x is None or self.test_x is None or self.train_y is None or self.test_y is None:
                self.__train_val_split__()

            if onlyexecute == False:
                params = self.hyperParameterTuning(params=self.hyperparam,cv=10,n_jobs=-1)
                self.model = self.set_model_parameters(params)
            else:
                pass

            fitted_model = self.model_fit()
        
        return fitted_model

    def shap_kernel_explainer(self, train_x=None, custom_model=None, model_type=None,**kwargs):
        if model_type is not None:
            self.model_type = model_type
        if custom_model is not None:
            self.custom_model = custom_model
        if train_x is not None:
            self.train_x = train_x
        if 'XGB'.upper() not in self.model_type.upper():
            if self.problem_type =='Classification':
                rf_shap_values = shap.KernelExplainer(self.custom_model.predict_proba, self.train_x, link="logit",**kwargs)
            elif self.problem_type == 'Regression':
                rf_shap_values = shap.KernelExplainer(self.custom_model.predict, self.train_x, link="logit",**kwargs)
        else:
            rf_shap_values = shap.TreeExplainer(self.custom_model).shap_values(self.train_x)
        return rf_shap_values
    
    def get_model_keywords(self):
        return self.models
    
    def select_best_model(self,max_evals=100,trial_timeout=120):
        if self.train_x is None or self.test_x is None or self.train_y is None or self.test_y is None:
            self.__train_val_split__()

        estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                                  preprocessing={},
                                  algo=tpe.suggest,
                                  max_evals=max_evals,
                                  trial_timeout=trial_timeout)

        estim.fit(self.train_x.values,self.train_y.values)
        print(estim.score(self.test_x, self.test_y.values))
        return estim