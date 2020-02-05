from airflow.operators import PythonOperator,BashOperator,DummyOperator
from airflow.models import DAG

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
        for index, connection in self.templatetoRun.items():
            task_id = connection.Node.Name
            function = connection.Node.Function
            params = connection.Node.Parameters
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