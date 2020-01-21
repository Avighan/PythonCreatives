import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import getitem
import pdb

class Function:

    def __init__(self,functionName,functionParameters):
        self.functionName = functionName
        self.functionParameters = functionParameters
        self.parent = None


class UDF_Flow_Creator(Function):

    exe_count=0
    def __init__(self):
        self.flow_dict = {}


    def create_function_node(self,functionName,inputParametes,functionRank=None):
        function = Function(functionName,inputParametes)
        self.exe_count = self.exe_count + 1
        function.rank = self.exe_count if functionRank is None else functionRank
        self.flow_dict[functionName]=function
        return function


    def delete_function_node(self,functionName):
        del self.flow_dict[functionName]

    def create_connections(self,parentFunction,childFunction):
        tempfunction = self.flow_dict[childFunction]
        tempfunction.parent = parentFunction
        self.flow_dict[childFunction]=tempfunction

    def get_flow_dict(self):
        return self.flow_dict



class UDF_Flow_executor:


    def __init__(self,flow_structure):
        self.flow_structure = flow_structure
        self.execution_dict = {}
        execution_flow = {}


    def get_flow_structure(self):
        return self.flow_structure

    def get_execution_status(self):
        return self.execution_dict

    def prepare_flow(self,steps_execute=None):
        if steps_execute is None:
            for steps in self.flow_structure:
                temp_fn_str = {}
                for variables,values in vars(self.flow_structure[steps]).items():
                    temp_fn_str[variables]=values
                self.execution_dict[steps] = temp_fn_str
            self.execution_dict = OrderedDict(sorted(self.execution_dict.items(),
                                     key=lambda x: getitem(x[1], 'rank')))

        return self.execution_dict



    def execute_flow(self,steps_execute=None):
        self.step_result={}
        execution_flow = self.prepare_flow(steps_execute)
        final_key = None
        for key,value in execution_flow.items():
            final_key = key
            if value['parent'] is None:
                params = value['functionParameters']
                self.step_result[key] = value['functionName'](**params)
            else:
                value_from_parent = value['functionParameters']['parent']
                del value['functionParameters']['parent']
                value['functionParameters'][value_from_parent] = self.step_result[value['parent']]
                params = value['functionParameters']
                self.step_result[key] = value['functionName'](**params)
        return self.step_result[final_key]


    def get_step_results(self,step_function):
        return self.step_result[step_function]

