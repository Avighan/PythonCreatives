import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool


class universal_reader:


    ExtensionMapper = {
        'csv':',',
        'parquet':None,
        'tab':'\t',
        'xlsx':None,
        'xls':None
    }

    def __init__(self,files,file_type="csv",**kwargs):
        self.files = files
        self.file_type = file_type


    def compile_all_files_into_df(self,fileList=None,nrows=None, header=None,**kwargs):
        all_data = pd.DataFrame()
        if fileList is None:
            fileList = self.files

        delimiter = self.ExtensionMapper[self.file_type]
        count = 0

        for file in fileList:

            if os.path.getsize(file) > 0:
                if count == 0:
                    temp_data = self.read_file_parallel(file,nrows,header,**kwargs)
                    all_data = temp_data
                else:

                    all_data = pd.concat([all_data, self.read_file_parallel(file,nrows,header,**kwargs)],
                                         axis=0)
                count += 1
                print(count, len(all_data))

        return all_data


    def read_file_parallel(self,file, nrows=None, header=None,**kwargs):
        delimiter = self.ExtensionMapper[self.file_type]
        if self.ExtensionMapper[self.file_type] is not None:
            if os.path.getsize(file) > 0:
                temp_data = pd.read_csv(file, delimiter=delimiter, nrows=nrows, header=header)
            else:
                temp_data = pd.DataFrame()
        elif self.file_type == 'parquet':
            if os.path.getsize(file) > 0:
                temp_data = pd.read_parquet(file)
            else:
                temp_data = pd.DataFrame()
        elif self.file_type == 'xlsx':
            if os.path.getsize(file) > 0:
                temp_data = pd.read_excel(file)
            else:
                temp_data = pd.DataFrame()

        return temp_data


    def compile_all_files_into_df_parallel(self,fileList=None,n_jobs=-1, nrows=None, header=None,columns=None):
        count = 0
        if fileList is None:
            fileList = self.files

        if n_jobs == -1:
            processes = multiprocessing.cpu_count()
        else:
            processes = n_jobs
        div = len(fileList) // processes
        rem = len(fileList) % processes
        if rem != 0:
            no_of_loop = div + 1
        else:
            no_of_loop = div


        df_final = pd.DataFrame()
        while no_of_loop > count:
            pool = Pool(processes=processes)
            start = count * processes if count != 0 else 0 * processes
            end = (count + 1) * processes if count != 0 else 1 * processes
            file_load = fileList[start:end]

            for file in file_load:
                print(file)
                results = pool.apply_async(
                    self.read_file_parallel,
                    args=(file, nrows, header),
                )
                if columns == None:
                    df_final = pd.concat(
                        [df_final, pd.DataFrame(results.get())], axis=0,
                    )
                else:
                    temp_data = pd.DataFrame(results.get())
                    temp_data = temp_data[columns]
                    df_final = pd.concat(
                        [df_final, temp_data], axis=0,
                    )

            pool.close()
            pool.join()
            count += 1

        return df_final



