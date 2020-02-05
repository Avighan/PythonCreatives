import pandas as pd
import sys
import os
import shutil
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.abspath(''), '.'))
from script import MLModels as mlmod
from script import Data_Preprocessors as dp
from script import ML_Metrics as metrics
import pickle
from script import ML_Template_Run as mltr
from script import Utilities as util
import pdb


class ML_func_temp1:
    def __init__(self):
        pass
    
    def set_data(self,**kwargs):
        self.input_data = {}
        self.input_data['data_file'] = kwargs['data_files'] #entered as list
        self.input_data['file_type'] = kwargs['file_type']
        self.input_data['y'] = kwargs['y']
        self.input_data['model_type'] = kwargs['model_type']
        self.input_data['model_to_run'] = kwargs['model_to_run']
        self.input_data['hyperparameters'] = kwargs['hyperparameters'] #parameter dictionary
        self.input_data['metrics_to_check'] = kwargs['metrics_to_check'] # as list of metrics
        self.input_data['outputPath'] = kwargs['outputPath']
        print(self.input_data['model_to_run'])
    
    
    def load_data(self,parallel=True,n_jobs=-1):
        files = self.input_data['data_file']
        unireader = util.universal_reader(files, self.input_data['file_type'])
        if parallel == True:
            df = unireader.compile_all_files_into_df_parallel(n_jobs=n_jobs,header=0)
        else:
            df = unireader.compile_all_files_into_df(header=0)
        return df


    def handle_na(self,df,type='mean'):
        df = df.fillna(df.mean())

        return df

    def encode_data(self,df,encoding_type=None):
        if encoding_type is not None:
            encoder = dp.Encoders(df=df, y=self.input_data['y'], encoding_type=encoding_type)
        else:
            encoder = dp.Encoders(df=df, y=self.input_data['y'])
        encoded_df = encoder.compile_encoding()
        return encoded_df
    
    def scale_data(self,df):
        scalar = dp.scaling(df=df, y=self.input_data['y'])
        scalar.select_scalar()
        scalar_df = scalar.compile_scalar()
        return scalar_df
    
    def create_train_val_split(self,df):
        
        ml = mlmod.MLmodels(df=df, y_column=self.input_data['y'], problem_type=self.input_data['model_type'])
        train_x, train_y, test_x, test_y = ml.__train_val_split__()
        return {'train_x':train_x,
                'train_y':train_y,
                'test_x':test_x,
                'test_y':test_y}
    
    def hypertune_parameters(self,**kwargs):
        if len(kwargs)>0:
            if self.input_data['hyperparameters'] is not None:
                df=kwargs['df']
                cv = 10 #kwargs['cv']
                n_jobs = -1 #n_jobs['n_jobs']
                ml = mlmod.MLmodels(y_column=self.input_data['y'], problem_type=self.input_data['model_type'])
                ml.set_train_test(train_x=df['train_x'], train_y=df['train_y'], test_x=df['test_x'], test_y=df['test_y'])
                ml.select_model_to_run(model_select=self.input_data['model_to_run'])
                params = ml.hyperParameterTuning(params=self.input_data['hyperparameters'], cv=cv, n_jobs=n_jobs)
            else:
                params = None
        else:
            params = None


        return params
    
    def ml_training(self,df,params=None):

        ml = mlmod.MLmodels(y_column=self.input_data['y'], problem_type=self.input_data['model_type'])
        train_xdf = df['train_x']
        test_xdf = df['test_x'][train_xdf.columns]
        ml.set_train_test(train_x=train_xdf, train_y=df['train_y'], test_x=test_xdf, test_y=df['test_y'])
        ml.select_model_to_run(model_select=self.input_data['model_to_run'])
        
        if params is not None:
            ml.set_model_parameters(**params)
        else:
            pass
        fitted_model = ml.compile_modeling(onlyexecute=True)
        return fitted_model
    
    def calculate_metrics(self,df,model):
        ml = mlmod.MLmodels(y_column=self.input_data['y'], problem_type=self.input_data['model_type'])
        y_pred = ml.model_predict(model=model,X=df['test_x'])
        y_test = df['test_y']
        metricclass = metrics.Metrics(y_test,y_pred, self.input_data['model_type'])
        metricOutput = {}
        for metric_req in self.input_data['metrics_to_check']:
            metricclass.select_metrics(metric_req)
            metricOutput[metric_req] = metricclass.metrics_solve()
    
        return metricOutput
    
    def run_shap(self,df,custom_model):
        shap_analyze = mlmod.MLmodels(y_column=self.input_data['y'], problem_type=self.input_data['model_type'])
        shap_output = shap_analyze.shap_kernel_explainer(train_x=df['train_x'],custom_model=custom_model, model_type=self.input_data['model_type'])
        return shap_output


    def get_best_model(self,df,max_evals=100,trial_timeout=120):
        ml = mlmod.MLmodels(y_column=self.input_data['y'], problem_type=self.input_data['model_type'])
        ml.set_train_test(df['train_x'],df['train_y'],df['test_x'],df['test_y'])
        estimator = ml.select_best_model(max_evals,trial_timeout)
        return estimator.best_model(),estimator.score(self.test_x.values, self.test_y.values)
    
    def get_statistics(self,df,col=None):
        if col is None:
            return df.describe()
        else:
            return df[col].describe()
        
        