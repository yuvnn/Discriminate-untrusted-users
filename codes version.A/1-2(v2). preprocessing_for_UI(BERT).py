import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# 처리할 폴더 경로
input_folder = '../dataset/rating_groupby_category_feather'
output_folder = '../dataset/rating_for_UI(BERT)+norm_feather'
os.makedirs(output_folder, exist_ok=True)

# 사전학습된 모델 로드
model = SentenceTransformer('all-MiniLM-L12-v2')

# 폴더 내 모든 feather 파일 처리
for file in os.listdir(input_folder):
    if file.endswith('.feather'):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        
        # Feather 파일 읽기
        df = pd.read_feather(input_path)
        
        # NaN을 빈 문자열로 변환
        df['content of review'] = df['content of review'].fillna("")
        
        # 문장들을 리스트로 준비
        sentences = df['content of review'].tolist()
        
        # 문장 임베딩 생성
        embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)
        
        # 임베딩 결과를 데이터프레임에 추가
        df['embedding'] = list(embeddings)
        
        # UserID 처리
        df['UserID'] = df['UserID'].astype(str)
        df['UserID'] = df['UserID'].apply(lambda x: x.split('.')[0])
        
        # 날짜 변환 (DD.MM.YYYY → timestamp)
        df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y')
        df['time'] = df['time'].astype('int64') // 10**9  # 초 단위
        
        # 필요한 컬럼만 저장
        final_df = df[['UserID', 'time', 'embedding']]
        final_df.to_feather(output_path)
        
        print(f"Processed: {file}")
