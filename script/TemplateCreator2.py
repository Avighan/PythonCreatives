import TemplateCreator as mltr
import Template1_Functions as tempfunc


data_files = ['C:/MyFolder/Data Science Team Dashboards/Generic ML model/dags/datasets/sample_data.xlsx']
file_type = 'xlsx'
target_column = 'careval'

hyperparameters = {'penalty': ['l1', 'l2'], 'C': [1.0, 2.0, 5.0]}
model_to_run = 'XGBClassifier'
metrics_to_check = ['Accuracy', 'Confusion']
shap = True
outputPath = './TestFolder'

encoding = True
scaling = True
shap = True




def_parames = tempfunc.TemplateFunctions(data_files = data_files,
file_type = file_type,
target_column = target_column,
hyperparameters = hyperparameters,
model_to_run = model_to_run,
metrics_to_check = metrics_to_check,
outputPath = outputPath)


mlflow_creator = mltr.UDF_Flow_Creator()
n1 = mlflow_creator.create_function_node('LoadData',def_parames.load_data,{'parallel':False})
n2 = mlflow_creator.create_function_node('Encodedata',def_parames.encode_data,{})
n3 = mlflow_creator.create_function_node('ScaleData',def_parames.scale_data,{})
n4 = mlflow_creator.create_function_node('TrainValSplit',def_parames.create_train_val_split,{})
n5 = mlflow_creator.create_function_node('Tuning',def_parames.hypertune_parameters,{})
n6 = mlflow_creator.create_function_node('ModelTrain',def_parames.ml_training,{})
n7 = mlflow_creator.create_function_node('Metrics',def_parames.calculate_metrics,{})

output = None
if encoding == True:
    if scaling == True:
        cn1 = mlflow_creator.create_connections(n1.functionName, n2.functionName,
                                                   input_output_params={'df': n1.functionName})
        output = mlflow_creator.create_connections(cn1.functionName, n3.functionName,
                                                   input_output_params={'df': cn1.functionName})
    else:
        output = mlflow_creator.create_connections(n1.functionName, n2.functionName,
                                                input_output_params={'df': n1.functionName})

else:
    if scaling == True:
        output = mlflow_creator.create_connections(n1.functionName,n3.functionName,input_output_params={'df':n1.functionName})
    else:
        output = mlflow_creator.create_connections(n1.functionName,n4.functionName,input_output_params={'df':n1.functionName})


cn3 = mlflow_creator.create_connections(output.functionName,n4.functionName,input_output_params={'df':output.functionName})
cn4 = mlflow_creator.create_connections(cn3.functionName,n5.functionName,input_output_params={'df':cn3.functionName})
cn5 = mlflow_creator.create_connections(cn4.functionName,n6.functionName,input_output_params={'df':cn3.functionName,'params':cn4.functionName})
cn6 = mlflow_creator.create_connections(cn5.functionName,n7.functionName,input_output_params={'df':cn3.functionName,'model':cn5.functionName})

pipeline_flow = mlflow_creator.get_flow_dict()
print(pipeline_flow)

mlflow_creator.save_flow(pipeline_flow,'Template1.pkl')

#executor = mltr.UDF_Flow_executor(mlflow_creator.get_flow_dict())
#print(executor.execute_flow())