import sys
import os
from airflow.operators import PythonOperator,BashOperator,DummyOperator
from airflow.models import DAG
sys.path.append(os.path.join(os.path.abspath('.'), ''))
sys.path.append(os.path.join(os.path.abspath('..'), ''))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__)))
import TemplateCreator


class Dagtaskexecute:
    def __init__(self,template_flow,dag):
        self.dag = dag
        self.dag_tasks = {}
        self.step_count = 0
        self.template_flow = template_flow

    def create_pythonoperator_task(self,task_id, function, params):
        if params is not None:
            task = PythonOperator(task_id=task_id, python_callable=function, op_kwargs=params, dag=self.dag)
        else:
            task = PythonOperator(task_id=task_id, python_callable=function, dag=self.dag)
        return task
    

    def set_conect_based_on_flow(self):
        count_pos = 0
        dag_flow = {}
        for seqno, connection in self.template_flow.items():
            if isinstance(connection.Node, TemplateCreator.FunctionNode):
                dag_flow[connection.Node.Name] = connection.Node
            elif type(connection.Node) is list:
                for node in connection.Node:
                    if isinstance(node, TemplateCreator.FunctionNode):
                        dag_flow[node.Name] = node
            else:
                dag_flow[connection.Node.Name] = connection.Node
        
        for stepname, functiondetails in dag_flow.items():
            task_id = stepname
            function = functiondetails.Function
            params = functiondetails.Parameters
            if count_pos == 0:
                if len(params) > 0:
                    self.dag_tasks[task_id] = self.create_pythonoperator_task(task_id, function, params)
                    prev_dag = self.dag_tasks[task_id]
                else:
                    self.dag_tasks[task_id] = self.create_pythonoperator_task(task_id, function, None)
                    prev_dag = self.dag_tasks[task_id]
            else:
                if len(params) > 0:
                    self.dag_tasks[task_id] = self.create_pythonoperator_task(task_id, function, params)
                    prev_dag.set_downstream(self.dag_tasks[task_id])
                    prev_dag = self.dag_tasks[task_id]
                else:
                    self.dag_tasks[task_id] = self.create_pythonoperator_task(task_id, function, None)
                    prev_dag = self.dag_tasks[task_id]
            count_pos += 1

    def create_dag(self):
        self.set_conect_based_on_flow()

