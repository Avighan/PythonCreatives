import pdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, StandardScaler, RobustScaler, Binarizer, FunctionTransformer, Normalizer, OrdinalEncoder, PowerTransformer, QuantileTransformer, LabelBinarizer, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest
import re



class FeatureSelection:

    def __init__(self):
        featureSelect = None
        self.selection_option = {
            'SelectKBest': SelectKBest,
            'RFE': RFE,
            'PCA': PCA,
            'LDA': LatentDirichletAllocation
        }

    def feature_selection_option(self,selectMethod):        
        return self.selection_option[selectMethod]

    def set_params(self,**kwargs):
        pass




class Preprocessing:

    get_outlier_data = pd.DataFrame()

    def __init__(self,data,**kwargs):
        self.outlier_detection_methods = {'default':self.default_outlier_method,
                                     'LocalOutlierFactor': LocalOutlierFactor}
        if 'select_method' not in kwargs.keys or kwargs['select_method'] is None:
            self.selected_method = 'default'
        else:
            self.selected_method = kwargs['select_method']
        self.data=data
        self.outlier_to_run = self.outlier_detection_methods[self.selected_method]

    def run_outier(self,series_data,**kwargs):
        outlier_detector = self.outlier_to_run(**kwargs)
        outlier_mask = outlier_detector.fit_predict(series_data)
        return {'outliermask':outlier_mask}


    def default_outlier_method(self,series_data,**kwargs):
        outlier_k = kwargs['k']
        q25, q75 = np.percentile(series_data, 25), np.percentile(series_data, 75)
        iqr = q75 - q25

        cut_off = iqr * outlier_k
        lower, upper = q25 - cut_off, q75 + cut_off

        return {'outliermask':[True if x < lower or x > upper else False for x in series_data]}

    def compile(self,**kwargs):
        for col in self.data.columns:
            self.get_outlier_data[col+'_'+'outlier'] = self.outlier_to_run(**kwargs)['outliermask']

        return self.get_outlier_data


class Imputers:

    def __init__(self,dataframe,select_imputer=None):
        self.df = dataframe
        if select_imputer is not None:
            self.select_imputer = select_imputer
        else:
            self.select_imputer = None
        self.imputers = {
            'SimpleImputer': SimpleImputer
        }

    def select_imputers(self,imputerSelect,**kwargs):
        
        self.imputer = self.imputers[imputerSelect](**kwargs)
        return self.imputer

    def fit(self, **kwargs):
        return self.imputer.fit(self.df, **kwargs)

    def transform(self, **kwargs):
        return self.imputer.transform(self.df, **kwargs)

    def fit_transform(self, **kwargs):
        return self.imputer.fit_transform(self.df, **kwargs)




class Encoders:
    def __init__(self, **kwargs):
        self.encoded_values = {}
        self.df = kwargs['df']
        self.encoders = {
            'LabelEncoder': LabelEncoder,
            'OneHotEncoder': OneHotEncoder,
            'OrdinalEncoder': OrdinalEncoder,
            'Binarizer': Binarizer,
            'LabelBinarizer': LabelBinarizer,
            'MultiLabelBinarizer': MultiLabelBinarizer
        }

        if 'cat_columns' not in kwargs.keys():
            self.cat_columns = [cat_col for cat_col in self.df.columns if self.df[cat_col].dtype==object]
            self.num_columns = [col for col in self.df.columns if col not in self.cat_columns]
        else:
            self.cat_columns = [cat_col for cat_col in kwargs['cat_columns'] if self.df[cat_col].dtype == object]
            self.num_columns = [col for col in self.df.columns if col not in kwargs['cat_columns']]

        if 'column_wise_encoding_dict' in kwargs.keys():
            self.column_wise_encoding_dict = kwargs['column_wise_encoding_dict']
        else:
            self.column_wise_encoding_dict = None
        if 'encoding_type' in kwargs.keys():
            self.encoding_type = kwargs['encoding_type']
            self.encoder = self.select_encoder()
        else:
            self.encoding_type = 'OneHotEncoder'
        self.y = kwargs['y']



    def select_encoder(self, encode_type =None, **kwargs):
        if encode_type is not None:
            self.encoding_type = encode_type
        else:
            pass
        self.encoder = self.encoders[self.encoding_type](**kwargs)
        return self.encoder


    def set_encoder_parameters(self,**kwargs):
        self.encoder = self.encoder(**kwargs)
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

        if self.column_wise_encoding_dict is None:

            if self.encoding_type is None:
                print('Encoder not defined!!!')
            for col in self.df.columns:
                if col in self.cat_columns:
                    if self.encoding_type != 'OneHotEncoder':
                        self.encoded_values[col] = self.fit(self.df[col].astype(str))
                        encode_df[col] = self.transform(col, self.df[col].astype(str))
                    else:
                        if col != self.y:
                            one_hot = pd.get_dummies(self.df[col])
                            one_hot.columns = [col + '_' +  str(s) for s in one_hot.columns]
                            encode_df = pd.concat([encode_df,one_hot],axis=1)
                        elif col == self.y:
                            sent_encoder = self.encoding_type
                            encoder_default_y = self.select_encoder('LabelEncoder')
                            self.encoded_values[col] = self.fit(self.df[col].astype(str))
                            encode_df[col] = self.transform(col, self.df[col].astype(str))
                            encoder_default_y = self.select_encoder(sent_encoder)
                else:
                    encode_df[col] = self.df[col]
        else:
            for encoder,columns in self.column_wise_encoding_dict.items():
                if encoder != 'OneHotEncoder':
                    for column in columns:
                        self.encoder = self.select_encoder(encoder)
                        self.encoded_values[column] = self.fit(self.df[column].astype(str))
                        encode_df[column] = self.transform(column, self.df[column].astype(str))
                else:
                    for column in columns:
                        if column != self.y:
                            one_hot = pd.get_dummies(self.df[column])
                            one_hot.columns = [column + '_' + str(s) for s in one_hot.columns]
                            encode_df = pd.concat([encode_df, one_hot], axis=1)
                        elif column == self.y:
                            sent_encoder = self.encoding_type
                            encoder_default_y = self.select_encoder('LabelEncoder')
                            self.encoded_values[column] = self.fit(self.df[column].astype(str))
                            encode_df[column] = self.transform(column, self.df[column].astype(str))
                            encoder_default_y = self.select_encoder(sent_encoder)

        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        encode_df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                      encode_df.columns.values]

        return encode_df


    def get_decoding(self, decode_df):
        for col in self.cat_columns:
            decode_df[col] = self.inverse_transform(col, decode_df[col])
        return decode_df


    def get_encoded_df(self):
        return self.encoder
    
    def get_encoder_list(self):
        return self.encoders




class scaling:

    def __init__(self, **kwargs):
        self.scalar_values = {}
        self.df = kwargs['df']
        if 'cat_columns' not in kwargs.keys():
            self.cat_columns = [cat_col for cat_col in self.df.columns if self.df[cat_col].dtype==object]
            self.num_columns = [col for col in self.df.columns if col not in self.cat_columns]
        else:
            self.cat_columns = [cat_col for cat_col in kwargs['cat_columns'] if self.df[cat_col].dtype == object]
            self.num_columns = [col for col in self.df.columns if col not in kwargs['cat_columns']]

        if 'column_wise_scalar_dict' in kwargs.keys():
            self.column_wise_scalar_dict = kwargs['column_wise_scalar_dict']
        else:
            self.column_wise_scalar_dict = None
        if 'scalar_type' in kwargs.keys():
            self.scalar_type = kwargs['scalar_type']
        else:
            self.scalar_type = 'StandardScaler'
        self.y = kwargs['y']
        self.scalars = {'StandardScaler': StandardScaler,
                        'MaxAbsScaler': MaxAbsScaler,
                        'MinMaxScaler': MinMaxScaler,
                        'RobustScaler': RobustScaler,
                        'Normalizer': Normalizer,
                        'PowerTransformer': PowerTransformer,
                        'QuantileTransformer': QuantileTransformer
                        }


    def select_scalar(self, scalar_type=None, **kwargs):
        if scalar_type is None:
            self.scalar_type = 'StandardScaler'
        else:
            self.scalar_type = scalar_type
        
        self.scalar = self.scalars[self.scalar_type](**kwargs)
        return self.scalar


    def set_scalar_parameters(self,**kwargs):
        self.scalar = self.scalar(**kwargs)
        return self.scalar

    def fit(self, x):

        fitted_scalar = self.scalar.fit(x)

        return fitted_scalar

    def transform(self, col, x):
        return self.scalar_values[col].transform(x)

    def inverse_transform(self, col, x):
        return self.scalar_values[col].inverse_transform(x)

    def fit_transform(self,col, x):
        return self.scalar_values[col].fit_transform(x)


    def compile_scalar(self):
        #scalar_df = pd.DataFrame()
        scalar_df = self.df[:]
        if self.column_wise_scalar_dict is None:
            for col in self.df.columns:
                if col in self.num_columns:
                    if self.y != col:
                        self.scalar = self.select_scalar(self.scalar_type)
                        self.scalar_values[col] = self.fit(self.df[[col]])
                        scalar_df[[col]] = self.transform(col, self.df[[col]])
                    else:
                        pass
                else:
                    scalar_df[col] = self.df[col]
        else:
            for scalar,columns in self.column_wise_encoding_dict.items():
                for col in columns:
                    self.scalar = self.select_scalar(self.scalar_type)
                    self.scalar_values[col] = self.fit(self.df[[col]])
                    scalar_df[[col]] = self.transform(col, self.df[[col]])
        return scalar_df


    def get_scalar_list(self):
        return self.scalars