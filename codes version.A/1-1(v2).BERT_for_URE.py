import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

# 1. 입력 폴더, 출력 폴더
input_folder = '../dataset/rating_groupby_category&userID_feather'
output_folder = '../output/URE(BERT)_output'
os.makedirs(output_folder, exist_ok=True)

# 2. SentenceTransformer 모델 로드
model = SentenceTransformer('all-MiniLM-L12-v2')

# 3. 입력 폴더의 feather 파일들 모두 처리
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.feather'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 3-1. 파일 읽기
        df = pd.read_feather(input_path)

        # 3-2. 문장 리스트 추출
        sentences = df['content of review'].tolist()

        # 3-3. 임베딩 생성
        embeddings = model.encode(sentences)

        # 3-4. 새로운 데이터프레임 만들기
        result_df = pd.DataFrame({
            'UserID': df['UserID'],
            'UR_BERT_Embedding': embeddings.tolist()
        })

        # 3-5. feather로 저장
        result_df.to_feather(output_path)

print("✅ 모든 feather 파일 처리 완료!")
