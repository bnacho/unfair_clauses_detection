import pandas as pd 
import os
import json
import xml.etree.ElementTree as ET


def load_data():
    train_folder_path = "data/Training"
    test_folder_path = "data/Validation"

    train_file_names = os.listdir(train_folder_path)
    test_file_names = os.listdir(test_folder_path)

    train_data = []
    test_data = []

    for file_name in train_file_names:
        if file_name.endswith(".json"):
            with open(f"{train_folder_path}/{file_name}", 'r', encoding = 'utf-8') as f:
                data = json.load(f)
                train_data.append(data)
            f.close()
    
    for file_name in test_file_names:
        if file_name.endswith(".xml"):
            with open(f"{test_folder_path}/{file_name}", 'r', encoding = 'utf-8') as f:
                try:
                    root = ET.fromstring(f.read())
                    data = root.find("cn")
                    test_data.append(data)
                except ET.ParseError:
                    print(f"Error parsing {file_name}")
            f.close()

    train = pd.DataFrame(train_data)
    test = pd.DataFrame(test_data)

    return train, test

train, test = load_data()
print(train)
print("="*50)
print(test)


