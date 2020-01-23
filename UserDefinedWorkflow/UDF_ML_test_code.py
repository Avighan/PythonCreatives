import pandas as pd
import ML_Template_Run as mltr
import Data_Preprocessors as dp
import MLModels as mlmod
import sys
import pdb

def load_data(file):
    df = pd.read_excel(file)
    return df


def encode_data(df,encoding_type):
    encoder = dp.Encoders(df=df)
    encoder.select_encoder(encoding_type=encoding_type)
    encoded_df = encoder.compile_encoding()
    return encoded_df

def ml_run(df,y,model_to_run,params=False):
    ml = mlmod.MLmodels(df = df,y_column= y )

    ml.select_model_to_run(model_select = model_to_run)
    if params == True:
        ml.set_model_parameters(params)
    else:
        pass
    model = ml.compile_modeling()
    return model


def ml_hyperparameter_tuning(df,y,model_to_run,params,cv=10,n_jobs=-1):

    ml = mlmod.MLmodels(df=df, y_column=y)

    ml.select_model_to_run(model_select=model_to_run)

    params = ml.hyperParameterTuning( params = params, cv=cv, n_jobs=n_jobs)
    return params


mlflow_creator = mltr.UDF_Flow_Creator()
n1 = mlflow_creator.create_function_node('load_data',load_data,{'file':'sample_data.xlsx'})
n2 = mlflow_creator.create_function_node('encode_data',encode_data,{'encoding_type':'LabelEncoder'})
n3 = mlflow_creator.create_function_node('ml_run',ml_run,{'y':'careval','model_to_run':'XGBClassifier'})
n4 = mlflow_creator.create_function_node('ml_hyper',ml_hyperparameter_tuning,{'y':'careval','model_to_run':'XGBClassifier','params':{'penalty': ['l1', 'l2'], 'C': [1.0, 2.0, 5.0]}})

cn1 = mlflow_creator.create_connections(n1.functionName,n2.functionName,input_output_params={'df': n1.functionName})
cn2 = mlflow_creator.create_connections(n1.functionName,n4.functionName,input_output_params={'df':cn1.functionName})
cn3 = mlflow_creator.create_connections(cn1.functionName,n3.functionName,input_output_params={'df': cn1.functionName,'params':cn2.functionName})

#Execute the model
test_exe = mltr.UDF_Flow_executor(mlflow_creator.get_flow_dict())
executed_flow = test_exe.execute_flow()
print(executed_flow)
print(test_exe.get_step_results())


