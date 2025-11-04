import pandas as pd
import re
from datasets import Dataset
import os
import json

# --- 설정 변수 ---
CSV_FILE_PATH = 'customer_dataset.csv'
DDL_FILE_PATH = 'schema.sql' # DDL 파일 경로 명시
SEP_TOKEN = " [SEP] " # 모델 입력 시퀀스 구분 토큰

# --- 1. DDL 파싱 함수 (스키마 구조 인코딩) ---
def get_schema_encoding(ddl_file_path):
    # 이 부분은 기존 코드를 함수로 통합했습니다.
    # DDL 파일에서 테이블명, 컬럼, PK 정보를 추출하여 하나의 텍스트로 만듭니다.
    
    import re
    import os

    with open(ddl_file_path, 'r', encoding='utf-8') as f:
        ddl_content = f.read()

    schema_metadata = {}

    table_pattern = re.compile(r'CREATE TABLE\s+(\w+)\s+\((.*?)\);', re.DOTALL | re.IGNORECASE)

    for match in table_pattern.finditer(ddl_content):
        table_name = match.group(1).lower()
        columns_block = match.group(2).strip()
        
        schema_metadata[table_name] = {'columns': [], 'primary_keys': []}
        
        for line in columns_block.split(','):
            line = line.strip()
            if not line:
                continue
                
            col_match = re.match(r'(\w+)\s+', line, re.IGNORECASE)
            if col_match:
                column_name = col_match.group(1).lower()
                
                if 'PRIMARY KEY' in line.upper():
                    pk_match = re.search(r'PRIMARY KEY\s*\((.*?)\)', line, re.IGNORECASE)
                    if pk_match:
                        pk_cols = [c.strip().lower() for c in pk_match.group(1).split(',')]
                        schema_metadata[table_name]['primary_keys'].extend(pk_cols)
                    else:
                        schema_metadata[table_name]['primary_keys'].append(column_name)

                # FK/CONSTRAINT 관련 라인은 컬럼 목록에 추가하지 않습니다.
                if not any(keyword in line.upper() for keyword in ['PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'REFERENCES']):
                    schema_metadata[table_name]['columns'].append(column_name)

    schema_encoding_texts = []
    for table, data in schema_metadata.items():
        cols = ", ".join(data['columns'])
        pk = ", ".join(data['primary_keys'])
        
        text = f"| {table} : {cols} (PK: {pk}) |"
        schema_encoding_texts.append(text)

    final_schema_encoding = " ".join(schema_encoding_texts)
    return final_schema_encoding

# --- 2. 데이터 전처리 (최소 가공 원칙 준수) ---
def get_processed_dataframe(csv_file_path, schema_encoding):
    
    df = pd.read_csv(csv_file_path, encoding='utf-8')

    # 1. 유효성 필터링 (최소 가공)
    df_valid = df[df['STATUS'] == '검토완료'].copy()
    df_valid = df_valid.drop(columns=['STATUS'])

    # 2. TABLES 컬럼 추출 로직 (기존 로직 유지)
    response_col_index = df_valid.columns.get_loc('RESPONSE')
    all_table_cols = df_valid.columns[response_col_index + 1:].tolist()

    def clean_and_combine_tables(row):
        table_values = row[all_table_cols].tolist()
        cleaned_list = [str(t).strip() for t in table_values if pd.notna(t) and str(t).strip() != '']
        return cleaned_list
        
    df_valid['TABLES'] = df_valid.apply(clean_and_combine_tables, axis=1)
    df_valid = df_valid.drop(columns=all_table_cols)

    # 3. SQL 정규화 및 플레이스홀더 처리
    def normalize_sql(sql):
        sql = re.sub(r'\s+', ' ', sql).strip()
        # {lang} 플레이스홀더만 'ko'로 대체 (SQL 문법 유효성 확보)
        sql = sql.replace('{lang}', "'ko'") 
        return sql

    df_final = df_valid.rename(columns={'QUESTION': 'natural_language', 'RESPONSE': 'SQL'})
    df_final['SQL'] = df_final['SQL'].apply(normalize_sql)
    
    # --- 4. 모델 입력 시퀀스 구성 (핵심 수정) ---
    
    # TABLES 리스트를 쉼표로 구분된 문자열로 변환 (모델 힌트용)
    df_final['TABLE_HINT'] = df_final['TABLES'].apply(lambda x: ", ".join(x))
    
    # 모든 행에 동일한 전체 스키마 인코딩 추가
    df_final['SCHEMA_ENCODING'] = schema_encoding

    # 최종 모델 입력 형식 (질문 + TABLES 힌트 + 전체 스키마)
    df_final['MODEL_INPUT'] = df_final['natural_language'] + \
                              SEP_TOKEN + "TABLES: " + df_final['TABLE_HINT'] + \
                              SEP_TOKEN + df_final['SCHEMA_ENCODING']
    
    return df_final.drop(columns=['TABLES', 'TABLE_HINT'])


# --- 3. 메인 실행 로직 ---
if __name__ == "__main__":
    # 1. 스키마 인코딩 생성
    schema_text = get_schema_encoding(DDL_FILE_PATH)
    
    # 2. 데이터프레임 전처리 및 통합 (TABLES 힌트 포함)
    df_final = get_processed_dataframe(CSV_FILE_PATH, schema_text)
    
    # 3. 최종 학습 파일로 저장
    df_final.to_json('final_training_data.json', orient='records', lines=True)
    
    print("✅ 최종 학습 데이터 파일 'final_training_data.json' 생성이 완료되었습니다.")