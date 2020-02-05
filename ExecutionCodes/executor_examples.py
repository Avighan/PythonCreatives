import sys
import os
 
sys.path.append(os.path.join(os.path.abspath('.'), ''))
sys.path.append(os.path.join(os.path.abspath('..'), ''))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from script import TemplateExecutor as te
import pdb
from datetime import datetime, timedelta
from script import TemplateCreator as temp_create
from script  import Template1Functions_for_Airflow as tfunc
from script import Dag_Creation_check_file as dagcreate

if __name__ ==  '__main__':

    execute = te.TemplateExecute()
    flow = execute.load_from_file('Template_Airflow.pkl')


    execute = te.TemplateExecute()
    flow = execute.load_from_file('Template_Airflow.pkl')
    params = {'SetData':{'folderpath':'./Example2/',
                         'data_files':['../datasets/diabetes.csv'],
                         'file_type':'csv',
                         'y':'Outcome',
                         'model_type':'Classification',
                         'model_to_run':'XGBClassifier',
                         'hyperparameters':{'penalty': ['l1', 'l2'], 'C': [1.0, 2.0, 5.0]},
                         'metrics_to_check':['Accuracy', 'Confusion'],
                         'outputPath':'./'}}


    flow = execute.update_node(params)
    execute.update_parameters('folderpath', './Example2/')

    dagCheck = dagcreate.Dagtaskexecute(flow, None)
    dagCheck.create_dag()
    pdb.set_trace()
    execute.execute_template(flow)

    params = {'SetData': {'folderpath': './Example3/',
                          'data_files': ['../datasets/breastcancer.csv'],
                          'file_type': 'csv',
                          'y': 'Class',
                          'model_type': 'Classification',
                          'model_to_run': 'XGBClassifier',
                          'hyperparameters': {'penalty': ['l1', 'l2'], 'C': [1.0, 2.0, 5.0]},
                          'metrics_to_check': ['Accuracy', 'Confusion'],
                          'outputPath': './'}}

    flow = execute.update_node(params)
    execute.update_parameters('folderpath', './Example3/')
    execute.execute_template(flow)

