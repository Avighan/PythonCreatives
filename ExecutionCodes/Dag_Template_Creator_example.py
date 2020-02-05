import sys
import os
import pdb
#sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.abspath(''), '.'))
from script import TemplateCreator as temp_create
from script  import Template1Functions_for_Airflow as tfunc

create = temp_create.TemplateCreator(templateName = 'AutoML_Test1')



n0 = create.create_functionNode('SetData',tfunc.set_data,{'folderpath':'./Example1/',
                                                          'data_files':['../datasets/boston-housing.csv'],
                                                          'file_type':'csv',
                                                          'y':'MEDV',
                                                          'model_type':'Regression',
                                                          'model_to_run':'XGBRegressor',
                                                          'hyperparameters':{'penalty': ['l1', 'l2'], 'C': [1.0, 2.0, 5.0]},
                                                          'metrics_to_check':['r2score'],
                                                          'outputPath':'./'})





n1 = create.create_functionNode('LoadData',tfunc.load_data,{'folderpath':'./Example1/','parallel':False})
n2 = create.create_functionNode('handleNA',tfunc.handle_na,{'folderpath':'./Example1/','file':tfunc.load_data.__name__,'type':'mean'})
n3 = create.create_functionNode('Encodedata',tfunc.encode_data,{'folderpath':'./Example1/','file':tfunc.handle_na.__name__})
n4 = create.create_functionNode('ScaleData',tfunc.scale_data,{'folderpath':'./Example1/','file':tfunc.encode_data.__name__})
n5 = create.create_functionNode('TrainValSplit',tfunc.create_train_val_split,{'folderpath':'./Example1/','file':tfunc.scale_data.__name__})
n6 = create.create_functionNode('Tuning',tfunc.hypertune_parameters,{'folderpath':'./Example1/','file':tfunc.create_train_val_split.__name__,'cv':10,'n_jobs':2})
n7 = create.create_functionNode('ModelTrain',tfunc.ml_training,{'folderpath':'./Example1/','file':tfunc.create_train_val_split.__name__,'params':tfunc.hypertune_parameters.__name__})
n8 = create.create_functionNode('Metrics',tfunc.calculate_metrics,{'folderpath':'./Example1/','file':tfunc.create_train_val_split.__name__,'model':tfunc.ml_training.__name__})
n9 = create.create_functionNode('Statistics',tfunc.get_statistics,{'folderpath':'./Example1/','file':tfunc.load_data.__name__})
n10 = create.create_functionNode('CreateFolder',tfunc.create_run_folder,{'folderpath':'./Example1/'})
template_flow = create.create_connectionSequence(node_exe_sequence=[n10,n0,n1,[n2,n9],n3,n4,n5,n6,n7,n8])

print(template_flow)
create.save_flow('Template_Airflow.pkl')
