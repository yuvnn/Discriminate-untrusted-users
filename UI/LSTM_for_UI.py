import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os


input_folder = "../dataset/rating_for_UI_feather"
output_folder = '../output/UI_output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# LSTM 모델 정의
class LSTMPreferenceModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(LSTMPreferenceModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)  # LSTM 출력 → 선호 벡터 예측

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# 선호 벡터 예측 함수
def predict_future_preference(model, past_embeddings):
    model.eval()
    with torch.no_grad():
        seq_tensor = torch.tensor(past_embeddings, dtype=torch.float32).unsqueeze(0)
        predicted_future = model(seq_tensor)
    return predicted_future.squeeze(0).numpy()

# a 폴더 내 모든 feather 파일에 대해 작업 수행
for filename in os.listdir(input_folder):
    if filename.endswith('.feather'):
        # Feather 파일 읽기
        file_path = os.path.join(input_folder, filename)
        df = pd.read_feather(file_path)

        # 리스트 형태의 embedding을 실제 리스트로 변환
        df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

        # UserID와 time을 기준으로 정렬
        df = df.sort_values(by=['UserID', 'time'])

        user_sequences = []
        user_targets = []

        for user_id, group in df.groupby('UserID'):
            sequence = np.stack(group['embedding'].values)  # embedding 벡터 리스트 → 배열 변환
            if len(sequence) > 1:  # 최소한 2개 이상의 데이터가 있어야 시퀀스를 만들 수 있음
                user_sequences.append(sequence[:-1])  # 입력 시퀀스
                user_targets.append(sequence[1:])  # 타겟 (다음 시간의 선호 벡터)

        # 하이퍼파라미터 설정
        embedding_dim = 100  # 벡터 차원
        hidden_dim = 128  # LSTM 은닉층 차원
        num_layers = 2  # LSTM 레이어 개수
        learning_rate = 0.001
        num_epochs = 10

        # 모델 초기화
        model = LSTMPreferenceModel(embedding_dim, hidden_dim, num_layers)
        criterion = nn.MSELoss()  # 예측된 벡터와 실제 벡터 간 차이를 최소화
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 학습 루프
        for epoch in range(num_epochs):
            total_loss = 0
            for seq, target in zip(user_sequences, user_targets):
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, 시퀀스 길이, 100)
                target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)  # (1, 시퀀스 길이, 100)

                optimizer.zero_grad()
                output = model(seq_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # 각 유저에 대한 선호 벡터 예측
        user_predictions = {}
        for user_id, user_sequence in zip(df['UserID'].unique(), user_sequences):
            past_embeddings = user_sequence[-5:]  # 최근 5개 embedding 사용
            predicted_preference = predict_future_preference(model, past_embeddings)
            user_predictions[user_id] = predicted_preference[-1]  # 마지막 예측 벡터 (미래 선호 벡터)

        # 유저별 예측된 선호 벡터를 DataFrame으로 변환
        user_predictions_df = pd.DataFrame(
            [(user_id, predicted_pref) for user_id, predicted_pref in user_predictions.items()],
            columns=['UserID', 'Predicted Preference']
        )

        # 예측 결과를 b 폴더에 Feather 형식으로 저장
        filename = filename.split('.')[0]
        output_file_path = os.path.join(output_folder, f"{filename}_UI.feather")
        user_predictions_df.to_feather(output_file_path)

        print(f"Predicted preferences saved to {output_file_path}.")
