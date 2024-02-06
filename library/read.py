from translate.storage.tmx import tmxfile
import os
import pandas as pd
from typing import Self
from t0_config import rawdata_path, workdata_path


class ReadMachine:
    def __init__(self:Self, params) -> None:
        self.directory = params['directory']
        self.file_name = params['file_name']
        self.from_lang = params['from_lang']
        self.to_lang = params['to_lang']
        self.rawdata = f'{rawdata_path}/{self.directory}/'
        self.workdata = f'{workdata_path}/{self.directory}/'

    def tmx2df(self:Self) -> pd.DataFrame:
        os.chdir(self.rawdata)
        with open(self.file_name, 'rb') as fin:
            tmx_file = tmxfile(fin, self.from_lang, self.to_lang)
        data = []
        for node in tmx_file.unit_iter():
            data.append([node.source, node.target])
        df = pd.DataFrame(data, columns=[self.from_lang, self.to_lang])
        return df

    def xlsx2df(self:Self) -> pd.DataFrame:
        os.chdir(self.rawdata)
        return pd.read_excel(self.file_name, header=0,  names=['zh', 'en'])

    def csv2df(self:Self) -> pd.DataFrame:
        os.chdir(self.rawdata)
        return pd.read_csv(self.file_name, header=0,  names=['zh', 'en'])

    def repeat_df(self:Self, df:pd.DataFrame, times:int) -> pd.DataFrame:
        return pd.concat([df]*times).reset_index(drop=True)

    def df2json(self:Self, df:pd.DataFrame) -> None:
        os.chdir(self.workdata)
        new_name = self.file_name.split('.')[0]+'.json'
        df.to_json(new_name, orient='records', lines=True)
        return df