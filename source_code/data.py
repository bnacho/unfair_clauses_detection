import pandas as pd 
import os
import json
import xml.etree.ElementTree as ET

def load_train():
    folder_path = "data/Training"
    file_names = os.listdir(folder_path)

    train = []

    for file_name in file_names:
        if file_name.endswith(".json"):
            with open(f"{folder_path}/{file_name}", "r", encoding = "utf-8") as f:
                data = json.load(f)
                train.append(data)
            f.close()

    df = pd.DataFrame(train)

    return df

def load_test():
    folder_path = "data/Validation"
    file_names = os.listdir(folder_path)
    test = []

    for file_name in file_names:
        if file_name.endswith(".xml"):
            try:
                tree = ET.parse(f"{folder_path}/{file_name}")
                root = tree.getroot()

                for cn in root.iter("cn"):
                    test.append(cn.text)
            except:
                pass
        
    df = pd.DataFrame(test, columns = ["clauseArticle"])

    return df

def load_all():
    train = load_train()
    test = load_test()

    return train, test


