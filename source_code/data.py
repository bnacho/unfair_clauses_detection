# 라이브러리 임포트
import pandas as pd 
import os
import json
import xml.etree.ElementTree as ET

def load_train():
    '''
    학습 데이터 불러오기
    json 형식
    파일이 나눠져 있으므로 반복문을 통해 로드
    '''
    folder_path = "data/Training"
    file_names = os.listdir(folder_path) # 폴더 안에 파일들을 불러온다.

    train = []

    for file_name in file_names:
        if file_name.endswith(".json"): # 끝이 .json은 파일들만 걸러낸다.
            with open(f"{folder_path}/{file_name}", "r", encoding = "utf-8") as f:
                data = json.load(f)
                train.append(data)
            f.close()

    df = pd.DataFrame(train)

    return df

def load_test():
    '''
    테스트 데이터 불러오기
    json 형식
    파일이 나눠져 있으므로 반복문을 통해 로드
    '''
    folder_path = "data/Validation"
    file_names = os.listdir(folder_path)
    test = []

    for file_name in file_names:
        if file_name.endswith(".json"):
            with open(f"{folder_path}/{file_name}", "r", encoding = "utf-8") as f:
                data = json.load(f)
                test.append(data)
            f.close()
    df = pd.DataFrame(test)

    return df

def load_all():
    train = load_train()
    test = load_test()

    return train, test


