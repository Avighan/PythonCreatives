import sys
import os
sys.path.append(os.path.join(os.path.abspath(''), '.'))
#sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
#sys.path.append(os.path.join(os.path.dirname(__file__)))
from script import TemplateExecutor as te
import pdb


if __name__ ==  '__main__':
    execute = te.TemplateExecute()
    flow = execute.load_from_file('Template_bestModel.pkl')


    params = {'SetData':{'data_files':['../datasets/diabetes.csv'],
                         'file_type':'csv',
                         'y':'Outcome',
                         'model_type':'Classification',
                         'model_to_run':'XGBClassifier',
                         'hyperparameters':{'penalty': ['l1', 'l2'], 'C': [1.0, 2.0, 5.0]},
                         'metrics_to_check':['Accuracy', 'Confusion'],
                         'outputPath':'./'}}

    flow = execute.update_node(params)
    execute.execute_template()
    print(execute.get_node_steps_name())
    print(execute.get_executionResults('Metrics'))
#print(execute.get_executionResults())


"""
Things to add -
1. Functions to see keyword for available models, metrics, encoders, scalers etc.
2. Update the connection flow based on execution step.
3. Create a class to create airflow dag based on the current flow diagram
4. Add or remove reference node
5. Run from intermediate steps 
6. Run from node a to b
7. Check and handle exceptions
8. Update autocreate instance for template 1 functions

"""
