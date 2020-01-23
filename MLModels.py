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
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import pdb

class MLmodels:
    def __init__(self, df=None, y_column=None,hyperparam=False):
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        if df is not None and y_column is not None:
            self.dataset = df
            self.y = df[[y_column]]
            self.x = df[[col for col in df.columns if col != y_column]]

        self.hyperparam = hyperparam

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
            'XGBRegressor':XGBRegressor,
            'XGBClassifier':XGBClassifier
        }
        if len(kwargs) > 0:
            self.model = self.models[model_select](**kwargs)
        else:
            self.model = self.models[model_select]()
        return self.model


    def set_model_parameters(self,**kwargs):
        self.model = self.model(**kwargs)
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
        if self.train_x is None or self.test_x is None or self.train_y is None or self.test_y is None:
            self.__train_val_split__()
        return self.model.fit(self.train_x, self.train_y, **kwargs)

    def model_predict(self,X=None):
        if X is None:
            return self.model.predict(self.test_x)
        else:
            return self.model.predict(X)

    def model_predict_proba(self,X=None):
        if X is None:
            return self.model.predict_proba(self.test_x)
        else:
            return self.model.predict_proba(X)

    def hyperParameterTuning(self, params, cv, n_jobs=-1, model=None):
        if self.train_x is None or self.test_x is None or self.train_y is None or self.test_y is None:
            self.__train_val_split__()
        from sklearn.model_selection import GridSearchCV
        pdb.set_trace()
        if model is None:
            grid_search = GridSearchCV(self.model, param_grid=params, cv=cv, verbose=1, n_jobs=n_jobs)
        else:
            grid_search = GridSearchCV(model, param_grid=params, cv=cv, verbose=1, n_jobs=n_jobs)
        modelsearch = grid_search.fit(self.train_x, self.train_y)
        self.params = grid_search.best_params_
        return self.params

    def compile_modeling(self):
        if self.hyperparam == False:
            if self.train_x is None or self.test_x is None or self.train_y is None or self.test_y is None:
                self.__train_val_split__()
            model = self.model_fit()
        else:
            if self.train_x is None or self.test_x is None or self.train_y is None or self.test_y is None:
                self.__train_val_split__()

            params = self.hyperParameterTuning(params=self.hyperparam,cv=10,n_jobs=-1)
            self.model = self.set_model_parameters(params)
            model = self.model_fit()
        return model
