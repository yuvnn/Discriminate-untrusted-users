## 전체 작업 플로우

![image](https://github.com/user-attachments/assets/4d82eaa7-5aaa-4900-b2a2-20ef97c1e333)

**URE (정적 선호도) - 기본적인 선호를 반영**

- 리뷰데이터 유저당 카테고리 별로 벡터 임베딩

**UI (시계열 데이터로 얻은 사용자의 동적 선호도) - 최근 변화한 선호를 반영**

- 리뷰데이터 임베딩 lstm입력 (시간대에 따라)

**UR**

- rating matrix 전처리
    
    ❗rating에 대해 비어 있는 데이터 다수 존재하므로
    
    - (**`matrix facorization`** 사용하여 결측값 처리) **`doc2vec`**

**F**

- 전체 결과 합치기
- 전체 결과 저차원 임베딩
