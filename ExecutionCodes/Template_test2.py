import pandas as pd
import Data_Preprocessors as dp
import MLModels as mlmod
import TemplateCreator as temp_create
import TemplateExecutor as execute

import sys
import pdb

def load_data(file):
    df = pd.read_excel(file)
    return df


def encode_data(df,encoding_type,y):
    encoder = dp.Encoders(df=df,y=y)
    encoder.select_encoder(encoding_type=encoding_type)
    encoded_df = encoder.compile_encoding()
    return encoded_df

def ml_run(df,y,model_to_run,params=None):
    ml = mlmod.MLmodels(df = df,y_column= y )
    ml.select_model_to_run(model_select = model_to_run)
    print(params)
    if params is not None:
        ml.set_model_parameters(**params)
    else:
        pass
    model = ml.compile_modeling()
    return model


def ml_hyperparameter_tuning(df,y,model_to_run,params,cv=10,n_jobs=-1):
    ml = mlmod.MLmodels(df=df, y_column=y)
    ml.select_model_to_run(model_select=model_to_run)
    params = ml.hyperParameterTuning( params = params, cv=cv, n_jobs=n_jobs)
    return params

def scale_data(df,y_column,scalar_type='StandardScaler'):
    scalar = dp.scaling(df=df,y=y_column)
    scalar.select_scalar(scalar_type=scalar_type)
    scalar_df = scalar.compile_scalar()
    return scalar_df



data_file = 'sample_data.xlsx'
y = 'careval'
encoding_type='LabelEncoder'
model_to_run = 'XGBClassifier'
params = {'penalty': ['l1', 'l2'], 'C': [1.0, 2.0, 5.0]}

create = temp_create.TemplateCreator(templateName = 'MLTemplate1')
n1 = create.create_functionNode(name = 'LoadData',function = load_data,parameters =  {'file':data_file})
n2 = create.create_functionNode('EncodeData',encode_data,{'df':n1,'encoding_type':encoding_type,'y':y})
n3 = create.create_functionNode('ScaleData',scale_data,{'df':n2,'y_column':y,'scalar_type':'StandardScaler'})
n4 = create.create_functionNode('Tuning',ml_hyperparameter_tuning,{'df':n3,'y':y,'model_to_run':model_to_run,'params':params,'cv':10,'n_jobs':-1})
n5 = create.create_functionNode('MLRun',ml_run,{'df':n3,'y':y,'model_to_run':model_to_run,'params':n4})
print(create.create_connectionSequence(node_exe_sequence=[n1,n2,n3,n4,n5]))

template_flow = create.create_connectionSequence(node_exe_sequence=[n1,n2,n4,n5])

exe = execute.TemplateExecute(template_flow)
print(exe.execute_template())
print(exe.get_executionResults())