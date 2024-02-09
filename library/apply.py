import pandas as pd
from functools import partial
from tqdm import tqdm, trange

def skip_train(text):
    skip_sentence = [
        'Technical Field',
        'Description of Related Art',
        'SUMMARY',
        'BRIEF DESCRIPTION OF THE DRAWINGS',
        'DESCRIPTION OF THE EMBODIMENTS',
        'WHAT IS CLAIMED IS:',
        'ABSTRACT',
    ]

    # 如果字符串以英文字母開頭，返回 True
    if text.strip() in skip_sentence:
        return True

    # 如果字符串為空，返回 True
    if not text.strip():
        return True

    return False

def translation_filter(raw_text_list:list[str], target_range:range) -> list[str]:
    try:
        target_text_list = [raw_text_list[i] for i in target_range]
        print(f'即將翻譯第 {target_range[0]} 句至第 {target_range[-1]} 句')
    except:
        target_text_list = raw_text_list
        print(f'即將翻譯整篇文章共 {len(raw_text_list)} 句')

    print('從以下開始翻譯:', target_text_list[0])

    return target_text_list

def flatten_list(nested_list:list)->list:
    """
    Flatten a nested list into a flat list.

    Args:
    - nested_list: Nested list.

    Returns:
    - Flattened list.
    """
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def split_func(text_list:list[str], sep:str, max_length:int, cut_half:bool=False)->list[str]:
    """
    Split the list of texts based on the specified delimiter and ensure that the length of each resulting substring
    does not exceed the specified maximum length.

    Args:
    - text_list: List containing the texts to be split.
    - sep: Delimiter used for splitting the texts.
    - max_length: Maximum length of the resulting substrings.
    - cut_half: Whether to attempt cutting in half when exceeding the maximum length. Default is False.

    Returns:
    - List of split texts.
    """
    text_list = text_list.copy()
    for index, text in enumerate(text_list):
        if len(text) > max_length:
            split_list = text.split(sep)
            if cut_half:
                num_sep = int(len(split_list)/2)
                split_list = [sep.join(split_list[:num_sep])] + [sep.join(split_list[num_sep:])]
            text_list[index] = [i+sep for i in split_list[:-1]] + [split_list[-1]]
    return flatten_list(text_list)

def count_characters(strings:str)->int:
    """
    Calculate the number of characters in the longest string in a list of strings.

    Args:
    - strings: List containing strings.

    Returns:
    - Number of characters in the longest string in the list.
    """
    return max([len(s) for s in strings])

def count_max_characters(split_sentence:pd.Series)->int:
    """
    Calculate the maximum number of characters in the split sentences in a DataFrame column.

    Args:
    - split_sentence: DataFrame column containing split sentences.

    Returns:
    - Maximum number of characters in the split sentences of the DataFrame column.
    """
    return max(split_sentence.apply(lambda x: count_characters(x)))


split_string_by_dot = partial(split_func, sep='。', max_length=0)
split_string_by_semicolon = partial(split_func, sep='；', max_length=100)
split_string_by_comma = partial(split_func, sep='，', max_length=100, cut_half=True)