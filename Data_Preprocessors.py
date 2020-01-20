import pdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, StandardScaler, RobustScaler, Binarizer, FunctionTransformer, Normalizer, OrdinalEncoder, PowerTransformer, QuantileTransformer, LabelBinarizer, MultiLabelBinarizer




class Encoders:
    def __init__(self, **kwargs):
        self.encoded_values = {}
        self.df = kwargs['df']
        if 'cat_columns' in kwargs.keys():
            self.cat_columns = [cat_col for cat_col in self.df.columns if self.df[cat_col].dtype==object]
            self.num_columns = [col for col in self.df.columns if col not in self.cat_columns]

        if 'column_wise_encoding_dict' in kwargs.keys():
            self.column_wise_encoding_dict = kwargs['column_wise_encoding_dict']
        else:
            self.column_wise_encoding_dict = None
        if 'encoding_type' in kwargs.keys():
            self.encoding_type = kwargs['encoding_type']
        else:
            self.encoding_type = None



    def select_encoder(self, encoding_type, **kwargs):
        self.encoding_type = encoding_type
        encoders = {
            'LabelEncoder': LabelEncoder,
            'OneHotEncoder': OneHotEncoder,
            'OrdinalEncoder': OrdinalEncoder,
            'Binarizer': Binarizer,
            'LabelBinarizer': LabelBinarizer,
            'MultiLabelBinarizer': MultiLabelBinarizer
        }
        self.encoder = encoders[self.encoding_type](**kwargs)
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
                        one_hot = pd.get_dummies(self.df[col])
                        one_hot.columns = [col + '_' +  str(s) for s in one_hot.columns]
                        encode_df = pd.concat([encode_df,one_hot],axis=1)
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
                        one_hot = pd.get_dummies(self.df[column])
                        one_hot.columns = [column + '_' + str(s) for s in one_hot.columns]
                        encode_df = pd.concat([encode_df, one_hot], axis=1)

        return encode_df

    def get_decoding(self, decode_df):
        for col in self.cat_columns:
            decode_df[col] = self.inverse_transform(col, decode_df[col])
        return decode_df


    def get_encoded_df(self):
        return self.encoder

class scaling:

    def __init__(self, dataframe):
        self.df = dataframe
        self.selectedscalar = None

    def select_scalars(self, scalarSelect, **kwargs):
        self.scalars = {'StandardScaler': StandardScaler,
                        'MaxAbsScaler': MaxAbsScaler,
                        'MinMaxScaler': MinMaxScaler,
                        'RobustScaler': RobustScaler,
                        'Normalizer': Normalizer,
                        'PowerTransformer': PowerTransformer,
                        'QuantileTransformer': QuantileTransformer
                        }

        self.selectedscalar = self.scalars[scalarSelect](**kwargs)
        return self.selectedscalar

    def set_scalar_parameter(self,**kwargs):
        self.selectedscalar = self.selectedscalar(**kwargs)
        return self.selectedscalar


    def fit(self, **kwargs):
        return self.selectedscalar.fit(self.df, **kwargs)

    def transform(self, **kwargs):
        return self.selectedscalar.transform(self.df, **kwargs)

    def fit_transform(self, **kwargs):
        return self.selectedscalar.fit_transform(self.df, **kwargs)


