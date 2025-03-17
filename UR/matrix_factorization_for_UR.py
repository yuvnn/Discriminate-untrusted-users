import os
import pandas as pd
from scipy.sparse import coo_matrix
import implicit

# R_size에 따른 num_factors 초기화 함수
def set_num_factors(R_size):
    factor_map = {
        (1000, 10000): 2,
        (10000, 100000): 4,
        (100000, 1000000): 8,
        (1000000, float('inf')): 8
    }
    for (lower, upper), factors in factor_map.items():
        if lower <= R_size < upper:
            return factors
    return 1  # 기본값

# ALS 모델 학습 함수
def train_als(R_sparse, num_factors=16, reg=0.1, iterations=20):
    model = implicit.als.AlternatingLeastSquares(
        factors=num_factors, 
        regularization=reg, 
        iterations=iterations
    )
    model.fit(R_sparse)
    return model

# 데이터 처리 및 ALS 모델 실행 함수
def process_feather_file(file_path, output_dir):
    # 1. 데이터 로드
    df = pd.read_feather(file_path)

    # NaN 값이 있는지 확인하고 처리 (NaN이 있는 행 제거)
    df = df.dropna(subset=["UserID", "products", "rating"])  # 필요한 컬럼에서 NaN 제거

    # 2. 사용자 & 아이템 리스트 생성
    users = sorted(df["UserID"].unique())
    items = sorted(df["products"].unique())

    # 3. 사용자-아이템 인덱스 매핑
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {p: j for j, p in enumerate(items)}

    # 4. 사용자-아이템 희소 행렬 생성
    row = df["UserID"].map(user_idx)
    col = df["products"].map(item_idx)
    data = df["rating"]  # 평점 데이터
    R_sparse = coo_matrix((data, (row, col)), shape=(len(users), len(items)))

    # R_size에 따른 num_factors 계산
    R_size = len(users) * len(items)
    num_factors = set_num_factors(R_size)

    # 5. ALS 모델 학습
    als_model = train_als(R_sparse, num_factors=num_factors)

    # 6. 사용자 벡터 추출
    user_factors_array = als_model.user_factors.to_numpy()
    user_factors_list = user_factors_array.tolist()

    # 7. 사용자 벡터 DataFrame으로 변환
    user_vectors = pd.DataFrame({
        "UserID": users, 
        "Vector": user_factors_list 
    })

    # 8. 결과를 output 폴더에 저장
    file_name = os.path.basename(file_path).split('.')[0]
    output_file = os.path.join(output_dir, "{}_UR.feather".format(file_name))
    user_vectors.to_feather(output_file)

# 메인 함수
def main(input_dir, output_dir):
    # input_dir에서 모든 Feather 파일을 순회
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".feather"):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing {file_path}...")
            process_feather_file(file_path, output_dir)
    print("Processing completed.")

# 실행
input_dir = "../dataset/rating_groupby_category_feather"  # 입력 파일이 있는 디렉토리
output_dir = "../output/UR_output"  # 결과를 저장할 출력 디렉토리

# 출력 디렉토리 생성 (없으면 생성)
os.makedirs(output_dir, exist_ok=True)

# 메인 함수 실행
main(input_dir, output_dir)
