import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.abspath(''), '.'))
from script import TemplateCreator as temp_create
from script  import Template1Functions as tf

create = temp_create.TemplateCreator(templateName = 'AutoML_Test1')

tfunc = tf.ML_func_temp1()

n0 = create.create_functionNode('SetData',tfunc.set_data,{'data_files':['../datasets/boston-housing.csv'],
                                                                 'file_type':'csv',
                                                                 'y':'MEDV',
                                                                 'model_type':'Regression',
                                                                 'model_to_run':'XGBRegressor',
                                                                 'hyperparameters':{'penalty': ['l1', 'l2'], 'C': [1.0, 2.0, 5.0]},
                                                                 'metrics_to_check':['r2score'],
                                                                 'outputPath':'./'})





n1 = create.create_functionNode('LoadData',tfunc.load_data,{'parallel':False})
n2 = create.create_functionNode('handleNA',tfunc.handle_na,{'df':n1,'type':'mean'})
n3 = create.create_functionNode('Encodedata',tfunc.encode_data,{'df':n2})
n4 = create.create_functionNode('ScaleData',tfunc.scale_data,{'df':n3})
n5 = create.create_functionNode('TrainValSplit',tfunc.create_train_val_split,{'df':n4})
n6 = create.create_functionNode('Tuning',tfunc.hypertune_parameters,{'df':n5,'cv':10,'n_jobs':2})
n7 = create.create_functionNode('ModelTrain',tfunc.ml_training,{'df':n5,'params':n6})
n8 = create.create_functionNode('Metrics',tfunc.calculate_metrics,{'df':n5,'model':n7})
n9 = create.create_functionNode('Statistics',tfunc.get_statistics,{'df':n1})
#n10 = create.create_functionNode('GetBestModel',tfunc.get_best_model,{'df':n5,'max_evals':100,'trial_timeout':120})
#n8 = create.create_functionNode('Shap',tfunc.run_shap,{'df':n4,'custom_model':n6})

template_flow = create.create_connectionSequence(node_exe_sequence=[n0,n1,[n2,n9],n3,n4,n5,n6,n7,n8])

print(template_flow)
create.save_flow('Template_bestModel.pkl')
