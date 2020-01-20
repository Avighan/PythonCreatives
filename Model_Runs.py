import pickle
import pandas as pd
import Data_Preprocessors as DP
import MLModels as mlrun
import ML_Metrics as metrics



class ModelExecution:

    def __init__(self):
        pass

    def save_object(self,file,obj):
        pickle.dump(obj, open(file, "wb"))

    def load_object(self,file,obj):
        self.obj = pickle.load(open(file, "rb"))
        return self.obj

    def load_data_analysis(self,data):
        if 'data' in data.keys():
            return data['data']
        else:
            if 'train' in data.keys() and 'test' in data.keys():
                df = pd.concat(data['train'], data['test'])
                return df



    def run_generic_model(self,**kwargs):
        # enter as dictionary, if test and train available mention as 'test':test_df,'train':train_df,
        # if nothing mentioned enter as df, specify the split as 'split':0.2 (if null default is taken as 0.3)
        print(kwargs.keys())
        output_columnName = kwargs['output_columnName']

        encoder_types = kwargs['encoders'] #Enter as dictionary with associated columns
        model_to_run = kwargs['models'] #Enter it as list of models and associated metrics example 'XgBoost':['Confusion','Accuracy']'
        process_maintainance = {}
        if 'model_type' in kwargs.keys():
            model_type = kwargs['model_type']
        else:
            exit()

        if 'data' in kwargs.keys():
            data = kwargs['data']
            data = self.load_data_analysis(data)
            process_maintainance['data'] = data
            encoded_df = data.copy(deep=True)
            if 'encoders' in kwargs.keys():

                if len(encoder_types.keys()) > 1:
                    enc = DP.Encoders(df=data,column_wise_encoding_dict =  encoder_types)
                    encoded_df = enc.compile_encoding()
                else:
                    enc = DP.Encoders(df=data, cat_columns=encoder_types.values())
                    encoded_df = enc.compile_encoding()


            models = {}
            metrics_save = {}
            metrics_value = {}
            if 'models' in kwargs.keys():
                for model,model_parameters in model_to_run.items():
                    models[model] = mlrun.MLmodels(encoded_df,output_columnName)
                    print (model_parameters)
                    train_x, test_x, train_y, test_y = models[model].__train_val_split__()
                    running_model = models[model].select_model_to_run(model_select=model)
                    if 'hyperparameter' in model_parameters.keys():
                        cv = model_parameters['cv'] if 'cv' in model_parameters.keys() else 10
                        params = models[model].hyperParameterTuning(model_parameters['hyperparameter'],cv= cv)
                        process_maintainance['best_params']=params
                    else:
                        params = None

                    if params is None:
                        running_model = models[model].select_model_to_run(model_select=model)
                    else:
                        running_model = models[model].select_model_to_run(model_select=model,parmas=params)

                    models[model].model_fit()
                    fitted_model = models[model].model_fit()
                    y_pred = models[model].model_predict()
                    if 'metrics' in model_parameters.keys():
                        for metric in model_parameters['metrics']:
                            metrics_save[model+'_'+ metric] = metrics.Metrics(test_y, y_pred, type=model_type)
                            metrics_save[model + '_' + metric].select_metrics(metric)
                            metrics_value[model + '_' + metric] = metrics_save[model + '_' + metric].metrics_solve()
                    else:
                        exit()
            else:
                exit()

        else:
            exit()

        process_maintainance['models'] = models
        process_maintainance['metrics'] = metrics_save
        process_maintainance['metric_value'] = metrics_value
        return process_maintainance




load_data = pd.read_excel('./sample_data.xlsx')
cat_columns = [cat_col for cat_col in load_data.columns if load_data[cat_col].dtype==object]
label_encoder_columns= [cat_col for cat_col in load_data.columns if load_data[cat_col].dtype==object  and cat_col != 'buying']
exe_model = ModelExecution()
print(exe_model.run_generic_model(data = {'data':load_data},
                  output_columnName ='careval',
                  encoders={'LabelEncoder':label_encoder_columns,'OneHotEncoder':['buying']},
                  models = {'XGBClassifier':{
                      'metrics':['Accuracy','Confusion'],'hyperparameter' : {'penalty':['l1', 'l2'], 'C': [1.0,2.0,5.0]},'cv':5
                  },

                        'LogisticRegression':
                            {'metrics':['Accuracy']}
                  },
                  model_type='Classification'))