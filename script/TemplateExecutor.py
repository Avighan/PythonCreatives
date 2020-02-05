from collections import OrderedDict
from operator import getitem
import pickle as pkl
from script import TemplateCreator
import multiprocessing
from multiprocessing import Pool

import pdb
class TemplateExecute:

    def __init__(self,templatetoRun=None,print_step_results=False):
        self.templatetoRun = templatetoRun
        self.executionResults = {}
        self.print_step_results=print_step_results
        
     
    def get_template(self):
        return self.templatetoRun
        
    def get_executed_template(self):
        return self.template_executing
    
    
    def execute_nodes(self,index,Node,executionResults):
        if index == 0:
            Params = Node.Parameters
        else:
            #pdb.set_trace()
            for paramkey, value in Node.Parameters.items():
                if isinstance(value, TemplateCreator.FunctionNode):
                    Node.Parameters[paramkey] = executionResults[value.nodeOutput]

            Params = Node.Parameters

        if Node.Parameters is None:
            executionResults[Node.nodeOutput] = Node.Function()
        else:
            executionResults[Node.nodeOutput] = Node.Function(**Params)
        return executionResults[Node.nodeOutput]
        
    def execute_template(self,flow=None):
        if flow is None:
            self.template_executing = self.templatetoRun
        else:
            self.template_executing = flow
            
        for index,connection in self.templatetoRun.items():
            if connection.runparallel == True:
                count = 0
                processes = multiprocessing.cpu_count()
                div = len(connection.Node) // processes
                rem = len(connection.Node) % processes
                
                if rem != 0:
                    no_of_loop = div + 1
                else:
                    no_of_loop = div

                pool = Pool(processes=processes)
                start = count * processes if count != 0 else 0 * processes
                end = (count + 1) * processes if count != 0 else 1 * processes
                nodes = connection.Node[start:end]

                for node in nodes:
                    print(node.Name)
                    results = pool.apply_async(
                        self.execute_nodes,
                        args=(index, node,self.executionResults),
                    )

                    self.executionResults[node.nodeOutput] = results.get()
                    if self.print_step_results == True:
                        print(self.executionResults[node.nodeOutput])
                    """
                    self.execute_nodes(index,node)
                    """
                pool.close()
                pool.join()
            else:
                self.executionResults[connection.Node.nodeOutput] =  self.execute_nodes(index,connection.Node,self.executionResults)
            if self.print_step_results == True:
                print(self.executionResults[connection.Node.nodeOutput])
        return 0
    
        
    def update_node(self,new_updated_node,old_node=None,modify_base=True):
        if old_node is not None:
            pass
        else:
            for index, connection in self.templatetoRun.items():

                if connection.runparallel == False:
                    for updatenodeName,Updatenodeparam in new_updated_node.items():
                        if connection.Node.Name == updatenodeName:
                            connection.Node.Parameters = Updatenodeparam
                else:
                    node_iterator = 0
                    for nodes in connection.Node:

                        for updatenodeName, Updatenodeparam in new_updated_node.items():
                            if connection.Node[node_iterator].Name == updatenodeName:

                                connection.Node[node_iterator].Parameters = Updatenodeparam
                        node_iterator+=1
                    
        if modify_base == True:
            self.templatetoRun = self.templatetoRun
            pkl.dump(self.templatetoRun, open(self.filename, "wb"))
        else:
            pass
        return self.templatetoRun

    
    def update_parameters(self,parameterName,parametervalue,modify_base=False):
        for index, connection in self.templatetoRun.items():
            if isinstance(connection.Node, TemplateCreator.FunctionNode):
                connection.Node.Parameters[parameterName]=parametervalue
            elif type(connection.Node) is list:
                for node in connection.Node:
                    if isinstance(node, TemplateCreator.FunctionNode):
                        node.Parameters[parameterName] = parametervalue
        
        if modify_base == True:
            self.templatetoRun = self.templatetoRun
            pkl.dump(self.templatetoRun, open(self.filename, "wb"))
        else:
            pass
        return self.templatetoRun
                        
    def load_from_file(self,filename):
        self.filename = filename
        self.templatetoRun = pkl.load(open(filename, "rb"))
        return self.templatetoRun

    def get_executionResults(self,stepName=None):
        if stepName is None:
            return self.executionResults
        else:
            return self.executionResults[stepName]

    
    def get_node_steps_name(self):
        node_name = []
        for index,connection in self.templatetoRun.items():
            if connection.runparallel == True:
                for node in connection.Node:
                    node_name.append(node.Name)

            else:
                node_name.append(connection.Node.Name)
            
        return node_name