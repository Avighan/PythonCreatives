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
        self.params = grid_search.best_params_
        return self.params
