from collections import OrderedDict
from operator import getitem
import pickle as pkl
import pdb


class FunctionNode:

    def __init__(self,name,function,parameters):
        self.Name = name
        self.Function = function
        self.Parameters = parameters
        self.nodeOutput = name

class Connection:
    def __init__(self,name,Node,ExeRank,runparallel=False):
        self.Name = name
        self.Node = Node
        self.executionSequence = ExeRank
        self.runparallel = runparallel

class TemplateCreator:

    def __init__(self,templateName):
        self.templateName = templateName
        self.ConnectionHolder = {}
        self.NodeFunctionHolder = {}
        self.Flow = {}

    def create_functionNode(self,name, function, parameters):
        Func = FunctionNode(name,function,parameters)
        self.NodeFunctionHolder[Func.Name] = Func
        return Func

    def create_connectionSequence(self,node_exe_sequence):        
        for i,node in enumerate(node_exe_sequence):
            if type(node) is list:
                measure_parallel_count = 0
                self.ConnectionHolder[i] = Connection('Seq'+str(i),node,i+1,True)
            else:
                self.ConnectionHolder[i] = Connection('Seq'+str(i),node,i+1,False)
        return self.ConnectionHolder


    def save_flow(self,file):
        pkl.dump(self.ConnectionHolder, open(file, "wb"))


    def load_from_file(self,filename):
        self.filename = filename
        self.templatetoRun = pkl.load(open(filename, "rb"))
        return self.templatetoRun

    def update_node(self, new_updated_node, old_node=None, modify_base=True):
        if old_node is not None:
            pass
        else:
            for index, connection in self.templatetoRun.items():

                if connection.runparallel == False:
                    for updatenodeName, Updatenodeparam in new_updated_node.items():
                        if connection.Node.Name == updatenodeName:
                            connection.Node.Parameters = Updatenodeparam
                else:
                    node_iterator = 0
                    for nodes in connection.Node:

                        for updatenodeName, Updatenodeparam in new_updated_node.items():
                            if connection.Node[node_iterator].Name == updatenodeName:
                                connection.Node[node_iterator].Parameters = Updatenodeparam
                        node_iterator += 1

        if modify_base == True:
            self.templatetoRun = self.templatetoRun
            pkl.dump(self.templatetoRun, open(self.filename, "wb"))
        else:
            pass
        return self.templatetoRun
