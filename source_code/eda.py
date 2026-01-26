# 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from konlpy.tag import Okt
from collections import defaultdict, Counter
from nltk.util import ngrams

# 로컬 라이브러리 임포트
from data import load_train, load_test, load_all
    
# 데이터 전처리
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
    train["clauseArticle"] = train["clauseArticle"].apply(lambda x : re.sub(r'[①-⑮]', ' ', x))
    # 불공정약관 여부를 int로 변환
    train["dvAntageous"] = train["dvAntageous"].astype(int)

    # 테스트 데이터의 약관 내용만 사용 & \n 등을 제거
    test = test[["clauseArticle", "dvAntageous"]]
    test["clauseArticle"] = test["clauseArticle"].str.join("") # 리스트를 문자열로 변환
    # \n, \r, \t를 제거
    test["clauseArticle"] = test["clauseArticle"].str.replace("\n", "")
    test["clauseArticle"] = test["clauseArticle"].str.replace("\r", "")
    test["clauseArticle"] = test["clauseArticle"].str.replace("\t", "")
    test["clauseArticle"] = test["clauseArticle"].apply(lambda x : re.sub(r'[①-⑮]', ' ', x))
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

def synonym():   
    '''
    Okt 모델을 사용하여 학습 데이터의 명사 토큰화를 수행,
    K-Means 클러스터링을 사용하여 대표적인 단어를 찾음
    Word2Vec 모델을 사용하여 유의어를 찾으려 했으나 정확도가 많이 떨어짐

    LLM을 활용하는 것이 더 정확할 것 같음
    '''
    train, test = simple_preprocess()
    okt = Okt()
    cluster_groups = defaultdict(list) # defaultdict : 기본값을 가지는 딕셔너리
    train_tokenized = []
    test_tokenized = []

    # 학습 데이터 명사 토큰화
    for i in range(len(train)):
        train_tokenized.append([word for word in okt.nouns(train["clauseArticle"][i]) if len(word) > 1])

    for i in range(len(test)):
        test_tokenized.append([word for word in okt.nouns(test["clauseArticle"][i]) if len(word) > 1])
    # Word2Vec 모델 학습 -> 벡터화
    model = Word2Vec(sentences = train_tokenized, vector_size = 100, 
    window = 5, min_count = 5, workers = 4)

    # Word2Vec 모델의 벡터화된 단어들을 가져오기
    word_vectors = model.wv.vectors
    word_list = model.wv.index_to_key

    # K-Means 클러스터링
    kmeans = KMeans(n_clusters = 100, random_state = 42)
    clusters = kmeans.fit_predict(word_vectors)

    # 클러스터링 결과를 단어와 매핑
    word_clusters = dict(zip(word_list, clusters))

    # 모든 단어를 포함한 리스트 생성, 단어 빈도수 계산
    all_words = [word for sublist in train_tokenized for word in sublist]
    word_counts = Counter(all_words)

    # 단어 빈도수와 클러스터링된 id를 함께 저장
    for word, cluster_id in word_clusters.items():
        if word in word_counts:
            word_counts[word] = (word_counts[word], cluster_id) # {단어 : (단어 빈도수, 클러스터링된 id)}

    # 클러스터링된 id를 기준으로 단어들을 그룹화
    for word, cluster_id in word_clusters.items():
        count = word_counts[word]
        cluster_groups[cluster_id].append((word, count))

    # 클러스터별 단어 빈도수 정렬
    for cluster_id in cluster_groups:
        cluster_groups[cluster_id].sort(key = lambda x : x[1], reverse = True)

    # 클러스터별 대표 단어 저장
    boss_words = []

    for cluster_id, words in cluster_groups.items():
        boss_word = words[0][0]
        boss_words.append(boss_word)
        # 명사가 아니면 제거
        for word in boss_words:
            pos_tag = okt.pos(word)
            if pos_tag != "Noun":
                boss_words.remove(word)
        print(f"Cluster {cluster_id}: {boss_word}")

    # # 유의어 대치
    # boss_words = ["회원", "금액", "계약", "책임", "상품", "사업자", "서비스"]
    # candidates = []
    # synonyms = defaultdict(list)

    # for boss_word in boss_words:
    #     candidates = model.wv.most_similar(boss_word, topn = 10)
    #     synonyms[boss_word] = [candidate for candidate in candidates if candidate[1] >= 0.8]
    
    # print(synonyms)
   
    # Gemini 3.0 pro를 사용하여 유의어를 찾음(대표 단어는 K-Means 클러스터링으로 찾음)
    synonym_map = {
    # 1. 인적 주체: '누가'
    '회원': '이용자', '고객': '이용자', '사용자': '이용자', '가입자': '이용자',
    
    # 2. 금전 객체: '얼마를' (비용 대신 '금액'으로 통일)
    '비용': '금액', '보험금': '금액', '보험료': '금액', '환급금': '금액', '계약금': '금액', '수수료': '금액',
    
    # 3. 금전 행위: '준다는 동작' (지급은 그대로 유지하되 유의어만 흡수)
    '납부': '지급', '지불': '지급', '입금': '지급', '송금': '지급',
    
    # 4. 계약 관련: '약속'과 '맺음'을 분리
    '약정': '계약', '협약': '계약', '합의': '계약',
    '성립': '체결', 
    
    # 5. 물품 관련
    '물품': '상품', '물건': '상품', '화물': '상품',
    
    # 6. 통지 관련
    '고지': '통지', '안내': '통지', '알림': '통지',
    
    # 7. 책임 관련 (이행은 행위이므로 유지)
    '의무': '책임', '과실': '책임'
    }

    # 유의어 대치 -> 토크나이징된 리스트로 반환
    train_final = [[synonym_map.get(word, word) for word in sentence] for sentence in train_tokenized]
    test_final = [[synonym_map.get(word, word) for word in sentence] for sentence in test_tokenized]

    return train_final, test_final

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

    okt = Okt() # 형태소 분석기

    target_tags = ['Noun', 'Verb', 'Adjective'] # 명사, 동사, 형용사

    # 불용어
    stopwords = [
        '하다', '되다', '있다', '수', '및', '등', '제', '항', '호', '경우', 
        '이', '그', '저', '것', '바', '위', '대하여', '관련', '내용', '의', 
        '을', '를', '가', '이', '은', '는', '에', '와', '과', '도', '갑', '을',
        '각'
    ]

    # 토큰화
    train_tokenized = []
    test_tokenized = []
        
    for i in range(len(train)):
        words = okt.pos(train["clauseArticle"][i], stem = True)
        for w, tag in words:
            if (tag in target_tags) and (w not in stopwords):
                train_tokenized.append(w)

    for i in range(len(test)):
        words = okt.pos(test["clauseArticle"][i], stem = True)
        for w, tag in words:
            if (tag in target_tags) and (w not in stopwords):
                test_tokenized.append(w)

    train_counter = Counter(train_tokenized)
    test_counter = Counter(test_tokenized)

    print("=" * 50)
    print("학습 데이터의 단어 빈도")
    print("=" * 50)
    print(train_counter.most_common(10))
    print("=" * 50)
    print("테스트 데이터의 단어 빈도")
    print("=" * 50)
    print(test_counter.most_common(10))

    # 단어 빈도 시각화
    train_word, train_frequency  = zip(*train_counter.most_common(10))
    test_word, test_frequency = zip(*test_counter.most_common(10))

    sns.barplot(x = train_word, y = train_frequency)
    sns.barplot(x = test_word, y = test_frequency)
    plt.title("학습 데이터의 단어 빈도")
    plt.xlabel("단어")
    plt.ylabel("빈도")
    plt.show()

def ngram_frequency():
    train_final, test_final = synonym()
    
    train_bigram = []
    train_trigram = []
    test_bigram = []
    test_trigram = []

    for sentence in train_final:
        train_bigram.extend(ngrams(sentence, 2))
        train_trigram.extend(ngrams(sentence, 3))

    for sentence in test_final:
        test_bigram.extend(ngrams(sentence, 2))
        test_trigram.extend(ngrams(sentence, 3))
    
    train_bigram_counter = Counter(train_bigram)
    train_trigram_counter = Counter(train_trigram)
    test_bigram_counter = Counter(test_bigram)
    test_trigram_counter = Counter(test_trigram)

    print("=" * 50)
    print("학습 데이터의 bigram")
    print("=" * 50)
    print(train_bigram_counter.most_common(10))
    print("=" * 50)
    print("학습 데이터의 trigram")
    print("=" * 50)
    print(train_trigram_counter.most_common(10))

    # ngram 시각화
    train_bigram, train_bigram_frequency  = zip(*train_bigram_counter.most_common(10))
    test_bigram, test_bigram_frequency = zip(*test_bigram_counter.most_common(10))

    train_trigram, train_trigram_frequency = zip(*train_trigram_counter.most_common(10))
    test_trigram, test_trigram_frequency = zip(*test_trigram_counter.most_common(10))

    joined_train_bigram = [' '.join(bigram) for bigram in train_bigram]
    joined_test_bigram = [' '.join(bigram) for bigram in test_bigram]
    
    joined_train_trigram = [' '.join(trigram) for trigram in train_trigram]
    joined_test_trigram = [' '.join(trigram) for trigram in test_trigram]

    sns.barplot(x = joined_train_bigram, y = train_bigram_frequency)
    sns.barplot(x = joined_test_bigram, y = test_bigram_frequency)
    plt.title("학습 데이터의 bigram")
    plt.xlabel("bigram")
    plt.ylabel("빈도")
    plt.show()

    sns.barplot(x = joined_train_trigram, y = train_trigram_frequency)
    sns.barplot(x = joined_test_trigram, y = test_trigram_frequency)
    plt.title("학습 데이터의 trigram")
    plt.xlabel("trigram")
    plt.ylabel("빈도")
    plt.show()

# 4, 5-gram도 의미 있는지 테스트
# TF-IDF 시각화
# 단어 빈도수 체크(너무 적게 나오는 단어 따로 처리)


    
        

        

        




