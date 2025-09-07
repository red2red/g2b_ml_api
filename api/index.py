from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# FastAPI 앱 생성
app = FastAPI()

# --- 모델 로딩 ---
# 앱이 시작될 때 모델을 한 번만 불러옵니다.
model_path = '../bid_winner_model.pkl' # main.py 기준 상대 경로
model_data = joblib.load(model_path)
model = model_data['model']
model_columns = model_data['columns']

# --- 입력 데이터 형식 정의 ---
# API로 받을 데이터의 형식을 미리 정의합니다.
class BidFeatures(BaseModel):
    prtcptCnum: int
    bssamt: float
    ntceInsttNm: str
    dminsttNm: str
    month: int
    weekday: int
    bidprcAmt: float
    # ... 모델 학습에 사용된 다른 피처들도 여기에 추가 ...

# --- API 엔드포인트 생성 ---
@app.get("/")
def read_root():
    return {"message": "나라장터 낙찰 확률 예측 API"}

@app.post("/predict")
def predict_winner(features: BidFeatures):
    # 1. 입력받은 데이터를 DataFrame으로 변환
    input_df = pd.DataFrame([features.dict()])
    
    # 2. 학습 때와 동일한 전처리 수행
    # 카테고리 타입 변환
    categorical_cols = ['ntceInsttNm', 'dminsttNm']
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype('category')

    # 학습 당시의 컬럼 순서와 동일하게 맞춰주기
    input_aligned = input_df.reindex(columns=model_columns, fill_value=0)

    # 3. 낙찰 확률 예측
    probability = model.predict_proba(input_aligned)[:, 1]
    win_probability = probability[0]

    # 4. 결과 반환
    return {"win_probability": win_probability}