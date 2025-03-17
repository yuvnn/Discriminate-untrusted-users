import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# 처리할 폴더 경로
folder_path = "../dataset/rating_groupby_category_feather"
output_folder = "../dataset/rating_for_UI_feather"
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 feather 파일 처리
for file in os.listdir(folder_path):
    if file.endswith(".feather"):
        file_path = os.path.join(folder_path, file)
        output_file = os.path.join(output_folder, file)
        
        # Feather 파일 읽기
        df = pd.read_feather(file_path)
        
        # NaN을 빈 문자열로 변환
        df['content of review'] = df['content of review'].fillna("")
        
        # 문장을 토큰화하고 review_idx를 태그로 지정
        tagged_data = [TaggedDocument(words=word_tokenize(row['content of review']), tags=[str(row['index'])]) 
                       for _, row in df.iterrows()]
        
        # Doc2Vec 모델 학습
        model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=1, workers=4, epochs=20)
        
        # 각 리뷰의 벡터를 가져오기
        df['embedding'] = df['index'].apply(lambda idx: model.dv[str(idx)])
        
        # UserID 처리
        df['UserID'] = df['UserID'].astype(str)
        df['UserID'] = df['UserID'].apply(lambda x: x.split('.')[0])
        
        # 날짜 변환 (DD.MM.YYYY → timestamp)
        df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y')
        df['time'] = df['time'].astype('int64') // 10**9  # 초 단위로 변환
        
        # 필요한 컬럼만 저장
        final_df = df[['UserID', 'time', 'embedding']]
        final_df.to_feather(output_file)
        
        print(f"Processed: {file}")
