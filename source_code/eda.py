# 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from konlpy.tag import Okt
from collections import Counter

# 로컬 라이브러리 임포트
from data import load_train, load_test, load_all

def simple_preprocess():
    # 데이터 불러오기
    train, test = load_all()
    # 학습 데이터의 약관 내용, 불공정약관 여부만 사용 & \n 등을 제거
    train = train[["clauseArticle", "dvAntageous"]]
    train["clauseArticle"] = train["clauseArticle"].str.join("") # 리스트를 문자열로 변환
    # \n, \r, \t를 제거
    train["clauseArticle"] = train["clauseArticle"].str.replace("\n", "")
    train["clauseArticle"] = train["clauseArticle"].str.replace("\r", "")
    train["clauseArticle"] = train["clauseArticle"].str.replace("\t", "")
    # 불공정약관 여부를 int로 변환
    train["dvAntageous"] = train["dvAntageous"].astype(int)

    # 테스트 데이터의 약관 내용만 사용 & \n 등을 제거
    test = test[["clauseArticle", "dvAntageous"]]
    test["clauseArticle"] = test["clauseArticle"].str.join("") # 리스트를 문자열로 변환
    # \n, \r, \t를 제거
    test["clauseArticle"] = test["clauseArticle"].str.replace("\n", "")
    test["clauseArticle"] = test["clauseArticle"].str.replace("\r", "")
    test["clauseArticle"] = test["clauseArticle"].str.replace("\t", "")

    # 불공정약관 여부를 int로 변환
    train["dvAntageous"] = train["dvAntageous"].astype(int)
    test["dvAntageous"] = test["dvAntageous"].astype(int)

    # 기본적인 정보 확인
    print(train.info()) # 1 : 유리, 2 : 불리
    print(test.info())

    # 라벨링 데이터 변경 -> 불리 : 0, 유리 : 1
    train["dvAntageous"] = train["dvAntageous"].astype(int).replace(2, 0)
    test["dvAntageous"] = test["dvAntageous"].astype(int).replace(2, 0)

    print("=" * 50)
    print("간단한 전처리 후 학습 데이터")
    print("=" * 50)
    print(train.head())
    print("=" * 50)
    print("간단한 전처리 후 테스트 데이터")
    print("=" * 50)
    print(test.head())

    return train, test

# 클래스별 데이터 비율 시각화
def class_distribution():
    train, test = simple_preprocess()

    temp_data = pd.concat([
        train[['dvAntageous']].assign(Dataset = "train"),
        test[['dvAntageous']].assign(Dataset = "test")
    ])
    sns.countplot(x = 'dvAntageous', hue = 'Dataset', data = temp_data)
    plt.title("클래스별 데이터 분포")
    plt.xlabel("독소조항 여부(0 : 독소조항, 1 : 정상조항)")
    plt.ylabel("개수")
    plt.legend(loc = "upper left")
    plt.show()
    
# 문장 길이 시각화
def sentence_length():
    train, test = simple_preprocess()

    # temp_data = pd.concat([
    #     train[['clauseArticle']].assign(Dataset = "train"),
    #     test[['clauseArticle']].assign(Dataset = "test")
    # ])

    train["sentence_length"] = train.apply(lambda x : len(x["clauseArticle"]), axis = 1)
    test["sentence_length"] = test.apply(lambda x : len(x["clauseArticle"]), axis = 1)
    
    sns.histplot(train["sentence_length"], kde = True)
    sns.histplot(test["sentence_length"], kde = True)
    plt.title("문장 길이 분포")
    plt.xlabel("문장 길이")
    plt.ylabel("개수")
    plt.show()

# 단어 빈도 시각화
def word_frequency():
    train, test = simple_preprocess()

    okt = Okt()
    train_tokenized = []
    test_tokenized = []
        
    for i in range(len(train)):
        train_tokenized.append(okt.morphs(train["clauseArticle"][i]))

    for i in range(len(test)):
        test_tokenized.append(okt.morphs(test["clauseArticle"][i]))

    train_counter = Counter(train_tokenized)
    test_counter = Counter(test_tokenized)

    train_counter.most_common(10)
    test_counter.most_common(10)

    print("=" * 50)
    print("학습 데이터의 단어 빈도")
    print("=" * 50)
    print(train_counter.most_common(10))
    print("=" * 50)
    print("테스트 데이터의 단어 빈도")
    print("=" * 50)
    print(test_counter.most_common(10))

word_frequency()
        

        

        




