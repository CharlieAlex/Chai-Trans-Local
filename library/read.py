from translate.storage.tmx import tmxfile
import os
import pandas as pd
from typing import Self
from t0_config import rawdata_path, workdata_path

class ReadMachine:
    def __init__(self:Self, params) -> None:
        self.file_name = params['file_name']
        self.from_lang = params['from_lang']
        self.to_lang = params['to_lang']
        self.rawdata = rawdata_path
        self.workdata = workdata_path

    def tmx2df(self:Self) -> pd.DataFrame:
        os.chdir(self.rawdata)
        with open(self.file_name + '.tmx', 'rb') as fin:
            tmx_file = tmxfile(fin, self.from_lang, self.to_lang)
        data = []
        for node in tmx_file.unit_iter():
            data.append([node.source, node.target])
        df = pd.DataFrame(data, columns=[self.from_lang, self.to_lang])
        return df

    def xlsx2df(self:Self) -> pd.DataFrame:
        os.chdir(self.rawdata)
        return pd.read_excel(self.file_name + '.xlsx', header=0,  names=['zh', 'en'])

    def csv2df(self:Self) -> pd.DataFrame:
        os.chdir(self.rawdata)
        return pd.read_csv(self.file_name + '.csv', header=0,  names=['zh', 'en'])

    def repeat_df(self:Self, df:pd.DataFrame, times:int) -> pd.DataFrame:
        return pd.concat([df]*times).reset_index(drop=True)

    def df2json(self:Self, df:pd.DataFrame) -> pd.DataFrame:
        os.chdir(self.workdata)
        df.to_json(self.file_name +'.json', orient='records', lines=True)
        print('資料已儲存成json檔案')
        return df