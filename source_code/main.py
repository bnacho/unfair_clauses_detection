# 라이브러리 임포트
import numpy as np
import pandas as pd 

# 로컬 라이브러리 임포트
from data import load_train, load_test, load_all

# 데이터 불러오기
train, test = load_all()

print(train.head())
print("="*50)
print(test.head())