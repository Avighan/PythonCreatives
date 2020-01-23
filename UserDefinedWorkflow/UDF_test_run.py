import pandas as pd
import ML_Template_Run as mltr
import sys

#Preparing the Functions
filename='sample_data.xlsx'
def load_data(file):
    load_data = pd.read_excel(file)
    return load_data

def show_header(df,n=10):
    return df.head(n)




#Defining the flow
test1 = mltr.UDF_Flow_Creator()
n1 = test1.create_function_node('load',load_data,{'file':filename})
n2 = test1.create_function_node('head',show_header,{'n':10})

cn1 = test1.create_connections(n1.functionName,n2.functionName,input_output_params={'df': n1.functionName})
test_exe = mltr.UDF_Flow_executor(test1.get_flow_dict())


print(test_exe.run_nodes(n1))
sys.exit()

executed_flow = test_exe.execute_flow()
#Extrating data from step
print(test_exe.get_step_results(cn1.functionName))

