# 라이브러리 임포트
from konlpy.tag import Okt
from kiwipiepy import Kiwi
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from collections import Counter

# 로컬 라이브러리 임포트
from data import load_all
from eda import simple_preprocess

def okt_tokenizer():
    """
    Okt 모델을 사용하여 토큰화
    """
    train, test = simple_preprocess()
    okt = Okt()
    train_tokenized = []
    test_tokenized = []

    # 명사만 추출하여 저장
    for i in range(len(train)):
        train_tokenized.append([word for word in okt.nouns(train["clauseArticle"][i]) if len(word) > 1])
    print(len(set([word for sublist in train_tokenized for word in sublist])))

    for i in range(len(test)):
        test_tokenized.append([word for word in okt.nouns(test["clauseArticle"][i]) if len(word) > 1])
    print(len(set([word for sublist in test_tokenized for word in sublist])))

    # 학습 데이터 토큰화 시각화
    plt.figure(figsize = (10, 6))
    sns.countplot(y = [word for sublist in train_tokenized for word in sublist], 
    order = [word for word, count in Counter([word for sublist in train_tokenized for word in sublist]).most_common(20)])
    plt.title("학습 데이터 토큰화 시각화(Okt)")
    plt.show()

    # 테스트 데이터 토큰화 시각화
    plt.figure(figsize = (10, 6))
    sns.countplot(y = [word for sublist in test_tokenized for word in sublist], 
    order = [word for word, count in Counter([word for sublist in test_tokenized for word in sublist]).most_common(20)])
    plt.title("테스트 데이터 토큰화 시각화(Okt)")
    plt.show()

    return train_tokenized, test_tokenized

def kiwi_tokenizer():
    """
    Kiwi 모델을 사용하여 토큰화
    """
    train, test = simple_preprocess()
    kiwi = Kiwi()
    train_tokenized = []
    test_tokenized = []

    for i in range(len(train)):
        train_tokenized.append([word.form for word in kiwi.tokenize(train["clauseArticle"][i]) if len(word.form) > 1 and word.tag == "NNG"])
    print(len(set([word for sublist in train_tokenized for word in sublist])))

    for i in range(len(test)):
        test_tokenized.append([word.form for word in kiwi.tokenize(test["clauseArticle"][i]) if len(word.form) > 1 and word.tag == "NNG"])
    print(len(set([word for sublist in test_tokenized for word in sublist])))

    # 학습 데이터 토큰화 시각화
    plt.figure(figsize = (10, 6))
    sns.countplot(y = [word for sublist in train_tokenized for word in sublist], 
    order = [word for word, count in Counter([word for sublist in train_tokenized for word in sublist]).most_common(20)])
    plt.title("학습 데이터 토큰화 시각화(Kiwi)")
    plt.show()

    # 테스트 데이터 토큰화 시각화
    plt.figure(figsize = (10, 6))
    sns.countplot(y = [word for sublist in test_tokenized for word in sublist], 
    order = [word for word, count in Counter([word for sublist in test_tokenized for word in sublist]).most_common(20)])
    plt.title("테스트 데이터 토큰화 시각화(Kiwi)")
    plt.show()

    return train_tokenized, test_tokenized

if __name__ == "__main__":
    train_okt, test_okt = okt_tokenizer()
    train_kiwi, test_kiwi = kiwi_tokenizer()

# Okt 고유명사 수
# 학습 : 3959
# 테스트 : 1696

# Kiwi 고유명사 수
# 학습 : 3206
# 테스트 : 1472

# 고유명사 수가 더 적은 Kiwi를 사용하는 것이 더 좋을 것 같음

