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
from script import Template1Functions_for_Airflow as tfunc

from script import Airflow_dag_creation as dagcreate
from airflow.operators import PythonOperator, BashOperator, DummyOperator
from airflow.models import DAG

seven_days_ago = datetime.combine(
    datetime.today() - timedelta(7), datetime.min.time())

args = {'owner': 'DLTeam', 'start_date': seven_days_ago, }

execute = te.TemplateExecute()
flow = execute.load_from_file('Template_Airflow.pkl')
params = {'SetData': {'folderpath': './Example2/',
                      'data_files': ['./datasets/diabetes.csv'],
                      'file_type': 'csv',
                      'y': 'Outcome',
                      'model_type': 'Classification',
                      'model_to_run': 'XGBClassifier',
                      'hyperparameters': {'penalty': ['l1', 'l2'], 'C': [1.0, 2.0, 5.0]},
                      'metrics_to_check': ['Accuracy', 'Confusion'],
                      'outputPath': './'}}

flow = execute.update_node(params)
execute.update_parameters('folderpath', './Example2/')
# execute.execute_template(flow)

dag = DAG(dag_id='AutoML2', default_args=args, schedule_interval=None)

dagimplement = dagcreate.Dagtaskexecute(template_flow=flow, dag=dag)
dagimplement.create_dag()
