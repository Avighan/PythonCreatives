import pandas as pd
import sys
import os
import shutil
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.abspath(''), '.'))
from script import MLModels as mlmod
from script import Data_Preprocessors as dp
from script import ML_Metrics as metrics
import pickle
from script import ML_Template_Run as mltr
from script import Utilities as util
import pdb




def set_data( **kwargs):
    
    input_data = {}
    input_data['data_file'] = kwargs['data_files']  # entered as list
    folderpath = kwargs['folderpath']
    input_data['file_type'] = kwargs['file_type']
    input_data['y'] = kwargs['y']
    input_data['model_type'] = kwargs['model_type']
    input_data['model_to_run'] = kwargs['model_to_run']
    input_data['hyperparameters'] = kwargs['hyperparameters']  # parameter dictionary
    input_data['metrics_to_check'] = kwargs['metrics_to_check']  # as list of metrics
    input_data['outputPath'] = kwargs['outputPath']
    pickle.dump(input_data, open(folderpath + set_data.__name__, "wb"))

def remove_folder(path):
    # check if folder exists
    if os.path.exists(path):
        # remove if exists
        shutil.rmtree(path)

def create_run_folder(folderpath):
    if os.path.exists(folderpath):
        remove_folder(folderpath)
        os.makedirs(folderpath)
    else:
        os.makedirs(folderpath)



def load_data(folderpath, parallel=True, n_jobs=-1):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    files = input_data['data_file']
    unireader = util.universal_reader(files, input_data['file_type'])
    if parallel == True:
        df = unireader.compile_all_files_into_df_parallel(n_jobs=n_jobs, header=0)
    else:
        df = unireader.compile_all_files_into_df(header=0)
    pickle.dump(df, open(folderpath + load_data.__name__, "wb"))
    return df

def handle_na(folderpath, file, type='mean'):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    df = pickle.load(open(folderpath + file, "rb"))
    df = df.fillna(df.mean())
    pickle.dump(df, open(folderpath + handle_na.__name__, "wb"))
    return df

def encode_data(folderpath, file, encoding_type=None):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    df = pickle.load(open(folderpath + file, "rb"))
    if encoding_type is not None:
        encoder = dp.Encoders(df=df, y=input_data['y'], encoding_type=encoding_type)
    else:
        encoder = dp.Encoders(df=df, y=input_data['y'])
    encoded_df = encoder.compile_encoding()
    pickle.dump(encoded_df, open(folderpath + encode_data.__name__, "wb"))
    return encoded_df

def scale_data(folderpath, file):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    df = pickle.load(open(folderpath + file, "rb"))
    scalar = dp.scaling(df=df, y=input_data['y'])
    scalar.select_scalar()
    scalar_df = scalar.compile_scalar()
    pickle.dump(scalar_df, open(folderpath + scale_data.__name__, "wb"))
    return scalar_df

def create_train_val_split(folderpath, file):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    df = pickle.load(open(folderpath + file, "rb"))
    ml = mlmod.MLmodels(df=df, y_column=input_data['y'], problem_type=input_data['model_type'])
    train_x, train_y, test_x, test_y = ml.__train_val_split__()

    pickle.dump({'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y}, open(folderpath + create_train_val_split.__name__, "wb"))

    return {'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y}

def hypertune_parameters( **kwargs):
    folderpath = kwargs['folderpath']
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    if len(kwargs) > 0:
        if input_data['hyperparameters'] is not None:
            df = pickle.load(open(folderpath + kwargs['file'], "rb"))
            cv = 10  # kwargs['cv']
            n_jobs = -1  # n_jobs['n_jobs']
            ml = mlmod.MLmodels(y_column=input_data['y'], problem_type=input_data['model_type'])
            ml.set_train_test(train_x=df['train_x'], train_y=df['train_y'], test_x=df['test_x'],
                              test_y=df['test_y'])
            ml.select_model_to_run(model_select=input_data['model_to_run'])
            params = ml.hyperParameterTuning(params=input_data['hyperparameters'], cv=cv, n_jobs=n_jobs)
        else:
            params = None
    else:
        params = None
    pickle.dump(params, open(folderpath + hypertune_parameters.__name__, "wb"))
    return params

def ml_training(folderpath, file, params=None):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    df = pickle.load(open(folderpath + file, "rb"))
    params = pickle.load(open(folderpath + params, "rb"))
    ml = mlmod.MLmodels(y_column=input_data['y'], problem_type=input_data['model_type'])
    train_xdf = df['train_x']
    test_xdf = df['test_x'][train_xdf.columns]
    ml.set_train_test(train_x=train_xdf, train_y=df['train_y'], test_x=test_xdf, test_y=df['test_y'])
    ml.select_model_to_run(model_select=input_data['model_to_run'])

    if params is not None:
        ml.set_model_parameters(**params)
    else:
        pass
    fitted_model = ml.compile_modeling(onlyexecute=True)
    pickle.dump(fitted_model, open(folderpath + ml_training.__name__, "wb"))
    return fitted_model

def calculate_metrics(folderpath, file, model):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    df = pickle.load(open(folderpath + file, "rb"))
    model = pickle.load(open(folderpath + model, "rb"))
    ml = mlmod.MLmodels(y_column=input_data['y'], problem_type=input_data['model_type'])
    y_pred = ml.model_predict(model=model, X=df['test_x'])
    y_test = df['test_y']
    metricclass = metrics.Metrics(y_test, y_pred, input_data['model_type'])
    metricOutput = {}
    for metric_req in input_data['metrics_to_check']:
        metricclass.select_metrics(metric_req)
        metricOutput[metric_req] = metricclass.metrics_solve()

    pickle.dump(metricOutput, open(folderpath + calculate_metrics.__name__, "wb"))
    return metricOutput

def run_shap(folderpath, file, custom_model):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    df = pickle.load(open(folderpath + file, "rb"))
    shap_analyze = mlmod.MLmodels(y_column=input_data['y'], problem_type=input_data['model_type'])
    shap_output = shap_analyze.shap_kernel_explainer(train_x=df['train_x'], custom_model=custom_model,
                                                     model_type=input_data['model_type'])
    pickle.dump(shap_output, open(folderpath + run_shap.__name__, "wb"))
    return shap_output

def get_best_model(folderpath, file, max_evals=100, trial_timeout=120):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    df = pickle.load(open(folderpath + file, "rb"))
    ml = mlmod.MLmodels(y_column=input_data['y'], problem_type=input_data['model_type'])
    ml.set_train_test(df['train_x'], df['train_y'], df['test_x'], df['test_y'])
    estimator = ml.select_best_model(max_evals, trial_timeout)
    pickle.dump(estimator, open(folderpath + get_best_model.__name__, "wb"))
    return estimator.best_model(), estimator.score(df['test_x'].values, df['test_y'].values)

def get_statistics(folderpath, file, col=None):
    input_data = pickle.load(open(folderpath + set_data.__name__, "rb"))
    df = pickle.load(open(folderpath + file, "rb"))
    if col is None:
        return df.describe()
    else:
        return df[col].describe()
    pickle.dump(df, open(folderpath + get_statistics.__name__, "wb"))
