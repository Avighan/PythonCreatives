import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import getitem
import pickle as pkl
import pdb

class Function:

    def __init__(self,functionName,function,functionParameters):
        self.functionName = functionName
        self.function = function
        self.functionParameters = functionParameters
        self.parent = None


    def get_function(self):
        return Function

class UDF_Flow_Creator(Function):

    exe_count=0
    connector_rank = 0
    def __init__(self):
        self.nodes = {}
        self.flow_dict = {}


    def create_function_node(self,functionName,function,inputParameters):
        function = Function(functionName,function,inputParameters)
        self.nodes[functionName]=function
        return function


    def delete_function_node(self,functionName):
        del self.nodes[functionName]


    def create_start_connection(self,parentFunction):
        if len(self.flow_dict) == 0:
            self.exe_count = self.exe_count + 1
            rank = self.exe_count
            parentFunction = self.nodes[parentFunction]
            startfunction = Function(parentFunction.functionName, parentFunction.function,
                                         parentFunction.functionParameters)
            startfunction.parent = None
            startfunction.rank = rank
            self.flow_dict[parentFunction.functionName] = startfunction
        else:
            pass






    def create_connections(self, parentFunction, childFunction,input_output_params=None,functionRank=None):
        self.create_start_connection(parentFunction)
        tempfunction = self.nodes[childFunction]
        self.connector_rank = self.connector_rank + 1
        connectedfunction = Function('AutoML_' +  childFunction,tempfunction.function,tempfunction.functionParameters)
        connectedfunction.parent = parentFunction

        self.flow_dict['AutoML_' +  childFunction] = connectedfunction
        connections = {'connections':{}}

        if input_output_params is None:
            connections = None
        else:
            for functionInput, joinedparent in input_output_params.items():
                if joinedparent in self.flow_dict.keys():
                    pass
                else:

                    self.exe_count = self.exe_count + 1
                    rank = self.exe_count if functionRank is None else functionRank
                    self.flow_dict[joinedparent] = self.nodes[joinedparent]
                    self.flow_dict[joinedparent].rank = rank


                connections['connections'].update({functionInput:joinedparent})
        self.exe_count = self.exe_count + 1
        rank = self.exe_count if functionRank is None else functionRank
        connectedfunction.rank = rank
        if connections is not None:
            self.flow_dict['AutoML_' +  childFunction].functionParameters = {**self.flow_dict['AutoML_' +  childFunction].functionParameters, **connections}
        else:
            pass

        return self.flow_dict['AutoML_' +  childFunction]

    def get_flow_dict(self):
        return self.flow_dict

    def get_function_nodes(self):
        return self.nodes


    def save_flow(self,obj,filename):
        pkl.dump(obj,filename)



    def prepare_flow(self):
        self.execution_dict = {}

        for steps in self.flow_dict:

            temp_fn_str = {}
            for variables,values in vars(self.flow_dict[steps]).items():
                temp_fn_str[variables]=values
            self.execution_dict[steps] = temp_fn_str
        self.execution_dict = OrderedDict(sorted(self.execution_dict.items(),
                                 key=lambda x: getitem(x[1], 'rank')))
        return self.execution_dict


class UDF_Flow_executor:



    def __init__(self,flow_structure=None):
        self.flow_structure = flow_structure
        self.execution_dict = {}
        execution_flow = {}


    def load_from_file(self,filename):
        self.flow_structure = pkl.load(filename)

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
            params={}

            if 'connections' not in value['functionParameters'].keys():
                params = value['functionParameters']
            else:
                for paramName, paramvalue in value['functionParameters'].items():

                    if paramName != 'connections':
                        params = {**params, **{paramName:paramvalue}}
                    else:
                        for inconparamname, inconparamvalue in value['functionParameters'][paramName].items():
                            params = {**params, **{inconparamname:self.step_result[inconparamvalue]}}

            print(key)
            self.step_result[key] = value['function'](**params)
        return self.step_result[final_key]


    def run_nodes(self,node,add_to_step=False):
        if add_to_step == False:
            function = node.function
            param  = node.functionParameters
            output = function(**param)
            return output
        else:
            function = node.function
            param = node.functionParameters
            output = function(**param)
            self.step_result[node.functionName] = output



    def get_step_results(self,step_function=None):
        if step_function is not None:
            return self.step_result[step_function]
        else:
            return self.step_result



