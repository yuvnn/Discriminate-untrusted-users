import os
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import nltk
from tqdm import tqdm

# punkt 다운로드 (처음에만 필요)
nltk.data.path.append("/home/st_group_1/anaconda3/envs/duu/nltk_data")
nltk.download('punkt', download_dir="/home/st_group_1/anaconda3/envs/duu/nltk_data")

# punkt 데이터가 설치되었는지 확인
try:
    nltk.data.find('tokenizers/punkt')
    print("punkt 데이터가 정상적으로 설치되었습니다.")
except LookupError:
    print("punkt 데이터가 설치되지 않았습니다.")

# a 폴더 안의 모든 feather 파일 목록 가져오기
folder_path = './rating_groupby_category&userID_feather'  # a 폴더 경로
feather_files = [f for f in os.listdir(folder_path) if f.endswith('.feather')]

# Doc2Vec 모델 학습 및 벡터화 함수
def train_doc2vec_model(df):
    tagged_corpus_list = []

    # 문서별로 토큰화 진행
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['content of review']
        tag = row['UserID']
        # 영어 문장을 단어 단위로 토큰화
        tokenized_text = word_tokenize(text.lower())  # 소문자로 변환하여 일관성 유지
        tagged_corpus_list.append(TaggedDocument(tags=[tag], words=tokenized_text))

    # Doc2Vec 모델 학습
    model = Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=8, window=8)
    model.build_vocab(tagged_corpus_list)
    model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=20)
    
    return model, tagged_corpus_list

# 각 Feather 파일에 대해 처리
for file_name in feather_files:
    print(f"처리 중: {file_name}")
    
    # Feather 파일 읽기
    df = pd.read_feather(os.path.join(folder_path, file_name))
    
    # 결측값 처리 및 데이터 준비
    df['content of review'] = df['content of review'].fillna("").astype(str)
    
    # Doc2Vec 모델 학습
    model, tagged_corpus_list = train_doc2vec_model(df)
    
    # 모델 저장
    model.save(f'dart.doc2vec')

    # 각 문서의 벡터 추출 및 'ID'와 함께 저장
    docvecs = [model.dv[doc_id] for doc_id in df['UserID']]
    
    # 벡터를 문자열로 변환 (각 벡터를 공백으로 구분하여 하나의 문자열로 저장)
    df['embedding'] = [' '.join(map(str, vec)) for vec in docvecs]
    
    # 필요한 컬럼만 추출
    df_vectors = df[['UserID', 'embedding']]
    
    # CSV로 저장
    output_file = f'./URE_output/{os.path.splitext(file_name)[0]}_URE.feather'
    df_vectors.to_feather(output_file)
    print(f"벡터 저장 완료: ",output_file)
