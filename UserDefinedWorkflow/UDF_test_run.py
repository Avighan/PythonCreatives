import pandas as pd
import ML_Template_Run as mltr

#Preparing the Functions
filename='sample_data.xlsx'
def load_data(file):
    load_data = pd.read_excel(file)
    return load_data

def show_header(df,n=10):
    return df.head(n)



#Defining the flow
test1 = mltr.UDF_Flow_Creator()
func1 = test1.create_function_node(load_data,{'file':filename})
func2 = test1.create_function_node(show_header,{'parent':'df','n':10})
test1.create_connections(load_data,show_header)
test_exe = mltr.UDF_Flow_executor(test1.get_flow_dict())
executed_flow = test_exe.execute_flow()

#Extrating data from step
print(test_exe.get_step_results(show_header))

