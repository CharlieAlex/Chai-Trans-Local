# 路徑
Project_path = '/Users/alexlo/Desktop/Project/Chai_Trans'
rawdata_path = f'{Project_path}/data/rawdata'
workdata_path = f'{Project_path}/data/workdata'
doc_path = f'{Project_path}/doc'
output_dir = f"{Project_path}/marian-finetuned-kde4-zh-to-en-local"

# 關鍵變數
file_name = '[for MT] 132957'
from_lang = 'zh'
to_lang = 'en'

target_range = range(0, 30)
commit_text = "char_5000_6000"
model_checkpoint = "charliealex123/marian-finetuned-kde4-zh-to-en"
model_name = "marian-finetuned-kde4-zh-to-en"