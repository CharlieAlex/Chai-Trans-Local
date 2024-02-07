import re

def is_english_start(text):
    """
    檢查一個字符串是否符合指定條件：
    1. 為空
    2. 英文字開頭
    3. 跳脫字元後英文字開頭，e.g. \tThis is an apple.
    """
    pattern = r'^(\s*|\\[^\s]*)[A-Za-z]'

    # 如果字符串以英文字母開頭，返回 True
    if re.match(pattern, text):
        return True

    # 如果字符串為空，返回 True
    if not text.strip():
        return True

    return False

def split_string_into_chunks(text, max_length=500):
    """
    將輸入的字串分成不超過指定長度的片段
    """
    chunks = []

    # 將輸入字串按照最大長度分成多個片段
    while len(text) > max_length:
        chunk = text[:max_length]
        chunks.append(chunk)
        text = text[max_length:]

    # 將最後剩餘的部分添加到片段中
    if len(text) > 0:
        chunks.append(text)

    return chunks

def translation_filter(raw_text_list:[str], target_range:range) -> [str]:
    try:
        target_text_list = [raw_text_list[i] for i in target_range]
        print(f'即將翻譯第 {target_range[0]} 句至第 {target_range[-1]} 句')
    except:
        target_text_list = raw_text_list
        print(f'即將翻譯整篇文章共 {len(raw_text_list)} 句')

    print('從以下開始翻譯:', target_text_list[0])

    return target_text_list
