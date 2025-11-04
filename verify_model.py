import torch
import os
from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import Dataset
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import json

# --- 1. 설정 및 경로 ---
# 최종 학습 결과 모델이 저장된 경로 (train_model.py와 동일해야 합니다)
MODEL_PATH = './kobart_model_output/final_model'

# 데이터베이스 연결 정보 (docker run 시 설정했던 값)
DB_CONFIG = {
    'DB_NAME': 'llm_schema_db',
    'DB_USER': 'llm_user',
    'DB_PASS': '1a2a3a4a',  # 설정한 비밀번호로 변경
    'DB_HOST': '127.0.0.1', # VM 내부에서 docker container로 접근 (PostgreSQL 포트)
    'DB_PORT': '5432'
}

# 최종 학습 데이터셋 파일
DATA_PATH = './final_training_data.json'

# --- 2. DB 연결 및 실행 함수 ---
def execute_sql(sql_query):
    """PostgreSQL 데이터베이스에 접속하여 SQL 쿼리를 실행합니다."""
    conn = None
    try:
        # SQLAlchemy 엔진을 사용하여 연결 (Pandas read_sql 사용 목적)
        engine = create_engine(
            f"postgresql+psycopg2://{DB_CONFIG['DB_USER']}:{DB_CONFIG['DB_PASS']}@{DB_CONFIG['DB_HOST']}:{DB_CONFIG['DB_PORT']}/{DB_CONFIG['DB_NAME']}"
        )
        
        # SQL 쿼리를 실행하고 결과를 DataFrame으로 받음
        df_result = pd.read_sql(sql_query, engine)
        
        return df_result.to_string(index=False, header=True)
        
    except Exception as e:
        return f"❌ SQL 실행 오류: {e}"
    finally:
        if conn:
            conn.close()

# --- 3. 모델 추론 함수 ---
def generate_sql(model, tokenizer, question, schema_encoding):
    """질문과 스키마를 입력으로 받아 SQL 쿼리를 생성합니다."""
    # 모델 입력 형식: 질문 [SEP] 스키마
    input_text = f"{question} [SEP] {schema_encoding}"
    
    # 토큰화
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True,
        padding="max_length"
    ).to(model.device)

    # SQL 생성 (추론)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=1024,
            num_beams=4,
            early_stopping=True
        )

    # 토큰을 SQL 텍스트로 디코딩
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_sql

# --- 4. 메인 검증 로직 ---
if __name__ == '__main__':
    print("=========================================")
    print("🤖 Text to SQL 모델 검증 시작 🤖")
    print("=========================================")
    
    # 0. GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 모델 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # AutoModelForSeq2SeqLM 대신 BartForConditionalGeneration 사용
        model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        exit()

    # 2. 데이터셋에서 테스트 질문 및 스키마 로드
    try:
        # final_training_data.json 파일에서 전체 데이터를 로드
        df = pd.read_json(DATA_PATH, lines=True)
        schema_encoding = df['SCHEMA_ENCODING'].iloc[0] # 모든 행의 스키마는 동일
        print(f"✅ Data and Schema loaded. Total {len(df)} test cases.")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}. 경로를 확인하세요.")
        exit()

    # 3. 테스트 케이스 정의 (상위 5개 및 특정 케이스)
    test_cases = [
        ("총 직원수가 몇명이야?"), # 단순 COUNT
        ("개발팀(org_code : OC001) 소속 직원의 이름과 이메일을 알려줘."), # JOIN 및 WHERE 조건
        ("홍길동(mem_id : MEM00019)의 직책은 뭐야?"), # 코드 테이블 JOIN
        ("지난 번에 반려된 결재 문서를 찾아줘."), # 결재 상태 코드 조건
        ("두 개 이상의 부서에 소속된 직원이 있는지 찾아줘."), # Subquery 또는 HAVING
    ]

    print("\n=========================================")
    print("📊 SQL 생성 및 DB 실행 결과")
    print("=========================================")

    for i, question in enumerate(test_cases):
        print(f"\n--- TEST CASE {i+1} ---")
        print(f"Q: {question}")
        
        # 3.1 SQL 생성 (추론)
        generated_sql = generate_sql(model, tokenizer, question, schema_encoding)
        print(f"A: [Generated SQL]\n {generated_sql}")
        
        # 3.2 DB 실행
        if generated_sql.upper().startswith("SELECT"):
            result_df = execute_sql(generated_sql)
            print(f"R: [DB Result]\n{result_df}")
        else:
            print("R: [DB Result] 유효하지 않은 SQL 형식입니다.")