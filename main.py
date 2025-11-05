import os
import re
import torch
import uvicorn # pip install "fastapi[all]"
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import csv         # <-- (추가 2)
import json        # <-- (추가 3)
import io          # <-- (추가 4)
import shutil # <-- (추가 1) 로컬 어댑터 폴더 삭제용
import threading

from google.cloud import storage # <-- GCS 라이브러리 임포트
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

# --- ⚠️ 중요 설정 ---
TRAIN_MAX_SEQ_LENGTH = 1024 
SCHEMA_FILE_PATH = "./schema.sql"
BASE_MODEL_NAME = "paust/pko-t5-base"
ADAPTER_DIR = "./sql-lora-adapter" # ⬅️ GCS에서 다운로드한 어댑터가 덮어쓸 로컬 경로
JSON_OUTPUT_FILE = "./final_training_data.json" 

GCS_BUCKET_NAME = "text2sql-pipeline-bucket" # ⬅️ 본인의 버킷 이름
GCS_ADAPTER_PREFIX = "adapters/" # ⬅️ GCS의 어댑터들이 저장된 상위 폴더

# 기타 설정
DTYPE = torch.float16 
SQL_PREFIX = "SQL 쿼리 생성: "

# --- Pydantic 모델 (입력 JSON 형식 정의) ---
class SQLRequest(BaseModel):
    prompt: str
    tables: List[str]

# (신규 추가 2) 배포 요청용 Pydantic 모델
class AdapterDeployRequest(BaseModel):
    adapter_name: str # 예: "adapter-20251105-073830"

# --- FastAPI 앱 인스턴스 생성 ---
app = FastAPI()
model_cache = {}

# (신규 추가 3) 
# VRAM에 접근하는 작업(추론, 핫스왑)이 동시에 일어나지 않도록 막는 잠금장치
model_lock = threading.Lock()

# --- 헬퍼 함수 (verify_model.py에서 가져옴) ---
def read_file_content(filepath):
    """파일 내용을 읽어 문자열로 반환합니다."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"오류: {filepath} 파일을 찾을 수 없습니다.")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise IOError(f"오류: {filepath} 파일 읽기 실패 - {e}")

def extract_schemas(full_sql, table_names):
    """
    full_sql 문자열을 파싱하여, table_names 목록에 있는
    테이블의 'CREATE TABLE ...;' 구문만 추출합니다.
    """
    extracted = []
    for table_name in table_names:
        pattern = rf"(CREATE TABLE\s+{re.escape(table_name)}\s*\(.*?;)"
        match = re.search(pattern, full_sql, re.IGNORECASE | re.DOTALL)
        
        if match:
            extracted.append(match.group(1).strip())
        else:
            print(f"--- 경고: '{table_name}' 테이블 스키마를 찾지 못했습니다.")
            
    return "\n\n".join(extracted)


# --- (신규 추가 1) prepare_training_data.py에서 가져온 스키마 파서 ---
def parse_schema(schema_content):
    """
    .sql 파일 내용을 파싱하여 테이블 이름과 CREATE TABLE 구문을 매핑하는 딕셔너리를 반환합니다.
    """
    schema_dict = {}
    # 'CREATE TABLE'로 시작해서 세미콜론(;)으로 끝나는 모든 블록을 찾습니다.
    statements = re.findall(r'(CREATE TABLE.*?;)', schema_content, re.DOTALL | re.IGNORECASE)
    
    if not statements:
        print(f"--- 🚨 경고: '{SCHEMA_FILE_PATH}'에서 'CREATE TABLE ... ;' 패턴을 찾을 수 없습니다.")
        
    for statement in statements:
        statement = statement.strip()
        # 테이블 이름 추출
        match = re.search(
            r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?`?\"?(\w+)`?\"?', 
            statement, 
            re.IGNORECASE
        )
        if match:
            table_name = match.group(1)
            schema_dict[table_name] = statement
            
    return schema_dict

def list_gcs_adapters(bucket_name, prefix):
    """GCS에서 'adapters/' 폴더 안의 하위 폴더 목록을 가져옵니다. (디버깅 모드)"""
    print(f"\n--- [Debug] list_gcs_adapters: 버킷 '{bucket_name}', 접두사 '{prefix}' 조회 시작 ---")
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # (수정) 'delimiter'를 제거하고 모든 blob을 다 가져옵니다.
        blobs = bucket.list_blobs(prefix=prefix)
        
        found_files = []
        found_folders = set() # set을 사용해 중복 폴더 이름 제거

        # blobs 리스트를 순회 (권한이 없다면 여기서 아무것도 반환되지 않음)
        for blob in blobs:
            found_files.append(blob.name) # 예: 'adapters/adapter-xxx/file.bin'
            
            # 'adapters/' 접두사를 제거한 나머지 경로를 봅니다.
            relative_path = blob.name[len(prefix):] # 예: 'adapter-xxx/file.bin'
            
            # 경로에 '/'가 포함되어 있다면 (즉, 하위 폴더가 있다면)
            if '/' in relative_path:
                # 첫 번째 '/' 앞부분(폴더 이름)을 추출합니다.
                folder_name = relative_path.split('/')[0]
                found_folders.add(folder_name)

        # --- 로그 출력 ---
        print(f"--- [Debug] 발견된 총 파일 수: {len(found_files)} ---")
        if found_files:
            # 너무 많으면 터미널이 멈추므로 최대 5개만 출력
            print(f"--- [Debug] 발견된 파일 (최대 5개): {found_files[:5]} ---")
        else:
            print(f"--- [Debug] '{prefix}'로 시작하는 파일을 GCS에서 '하나도' 찾지 못했습니다. ---")
            
        print(f"--- [Debug] 추출된 폴더 (set): {found_folders} ---")
        # -----------------
        
        return list(found_folders) # set을 list로 변환하여 반환
        
    except Exception as e:
        print(f"--- ❌ GCS 어댑터 목록 조회 실패 (예외 발생): {e} ---")
        return None

def download_gcs_directory(bucket_name, gcs_prefix, local_dir):
    """GCS의 특정 폴더(prefix)를 로컬 디렉토리로 다운로드 (덮어쓰기)"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=gcs_prefix) # 이 폴더 안의 모든 파일

        # 1. 기존 로컬 어댑터 폴더를 깨끗하게 삭제
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        
        # 2. 빈 폴더 다시 생성
        os.makedirs(local_dir, exist_ok=True)
        
        download_count = 0
        for blob in blobs:
            # GCS 경로에서 로컬 경로로 변환
            relative_path = os.path.relpath(blob.name, gcs_prefix)
            local_path = os.path.join(local_dir, relative_path)
            
            # 파일이 속한 하위 디렉토리 생성 (필요한 경우)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 파일 다운로드
            blob.download_to_filename(local_path)
            download_count += 1
            
        return download_count
    except Exception as e:
        print(f"--- ❌ GCS 어댑터 다운로드 실패: {e} ---")
        return None

# --- 1. 서버 시작 시 모델 로드 이벤트 ---
@app.on_event("startup")
async def load_model_and_schema():
    """
    FastAPI 서버가 시작될 때 모델, 토크나이저, 
    그리고 CSV 변환을 위한 '스키마 딕셔너리'를 미리 로드합니다.
    """
    print("--- FastAPI 서버 시작 ---")
    
    try:
        print(f"1. 전체 스키마 로드: {SCHEMA_FILE_PATH}")
        full_schema = read_file_content(SCHEMA_FILE_PATH)
        model_cache["full_schema_sql"] = full_schema
        
        # --- (신규 추가 2) 스키마 딕셔너리를 파싱하여 캐시 ---
        print("2. 스키마 파싱 (Dict 생성)...")
        model_cache["schema_dict"] = parse_schema(full_schema)
        # -----------------------------------------------

        # (신규 추가 4) 핫스왑을 위해 VRAM 로드 로직을 함수로 분리
        print("3. VRAM에 모델 핫스왑 로드 시도...")
        load_model_into_vram()
        
        print("--- 🚀 모델, 스키마 딕셔너리 로드 완료. 서버 준비됨. ---")
        
    except FileNotFoundError as e:
        print(f"--- 🚨 치명적 오류: {e} ---")
        print("--- 'sql-lora-adapter'가 존재하지 않거나 'schema.sql' 파일이 없습니다. ---")

# (신규 추가 5) VRAM에 모델을 로드하는 핵심 로직 (재사용 가능하게 분리)
def load_model_into_vram():
    """
    로컬 ADAPTER_DIR의 어댑터를 VRAM으로 로드합니다.
    (주의: 이 함수는 model_lock으로 보호된 상태에서 호출되어야 함)
    """
    try:
        # 1. 기존 모델이 VRAM에 있다면 비우기 (핫스왑)
        if "model" in model_cache:
            print("--- [Hot-Swap] 기존 모델 VRAM에서 제거 시도... ---")
            del model_cache["model"]
            del model_cache["tokenizer"]
            torch.cuda.empty_cache() # VRAM 찌꺼기 청소
            print("--- [Hot-Swap] VRAM 제거 완료. ---")

        # 2. 새 파일로 토크나이저/모델 로드
        print(f"--- [Hot-Swap] VRAM 로드 시작: {ADAPTER_DIR} ---")
        
        # (startup과 동일한 로직)
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL_NAME, quantization_config=bnb_config,
            dtype=DTYPE, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        model.eval()
        
        # 3. 캐시에 저장
        model_cache["tokenizer"] = tokenizer
        model_cache["model"] = model
        print("--- [Hot-Swap] VRAM 로드 완료. ---")
        
    except Exception as e:
        print(f"--- 🚨 핫스왑 중 VRAM 로드 실패: {e} ---")
        # 모델 로드에 실패하면 서버가 응답 불가능 상태가 되므로,
        # model_cache를 비워 /verify-model이 실패하도록 유도
        model_cache.pop("model", None)
        model_cache.pop("tokenizer", None)
        raise e # 상위 핸들러가 오류를 잡도록 예외를 다시 발생시킴

# --- (신규 추가 4) GCS 어댑터 목록 조회 엔드포인트 ---
@app.get("/list-adapters")
async def get_gcs_adapters():
    """GCS 버킷의 'adapters/' 폴더에 저장된 어댑터 버전 목록을 반환합니다."""
    adapters = list_gcs_adapters(GCS_BUCKET_NAME, GCS_ADAPTER_PREFIX)
    if adapters is None:
        raise HTTPException(status_code=500, detail="GCS에서 어댑터 목록을 가져오는 데 실패했습니다.")
    
    return {"adapters": adapters}

# --- (신규 추가 5) 어댑터 배포(다운로드/덮어쓰기) 엔드포인트 ---
@app.post("/deploy-adapter")
async def deploy_adapter_from_gcs(request: AdapterDeployRequest):
    """
    GCS에서 지정된 어댑터를 로컬 'ADAPTER_DIR'로 다운로드하여 덮어씁니다.
    (주의: 이 작업 후 서버를 '수동으로 재시작'해야 적용됩니다.)
    """
    adapter_name = request.adapter_name
    gcs_prefix = f"{GCS_ADAPTER_PREFIX.strip('/')}/{adapter_name}" # 예: adapters/adapter-xxx

    # (신규 추가 6) 다른 요청(verify)이 모델을 사용하는 것을 막음
    if not model_lock.acquire(timeout=5): # 5초간 잠금을 못 얻으면 포기
         raise HTTPException(status_code=503, detail="서버가 다른 작업(배포/추론)으로 바쁩니다. 잠시 후 다시 시도하세요.")
         
    try:
        print(f"\n--- [Deploy] 핫스왑 배포 시작: {adapter_name} ---")
        
        # 1. 파일 다운로드 (VRAM 사용 안 함)
        count = download_gcs_directory(GCS_BUCKET_NAME, gcs_prefix, ADAPTER_DIR)
        
        if count is None:
            raise HTTPException(status_code=500, detail=f"'{adapter_name}' 다운로드 중 GCS 오류 발생")
        if count == 0:
            raise HTTPException(status_code=404, detail=f"'{adapter_name}'을 GCS에서 찾을 수 없거나 파일이 없습니다.")
        
        print(f"--- [Deploy] GCS 다운로드 완료. VRAM 핫스왑 시작 (서버 멈춤) ---")

        # 2. (핵심) VRAM 핫스왑 실행
        # 이 작업은 10~20초간 블로킹됩니다.
        load_model_into_vram() 
        
        message = f"성공: '{adapter_name}'을 배포하고 VRAM에 핫스왑 완료. 서버가 새 모델로 즉시 응답합니다."
        print(message)
        return {"message": message, "deployed_adapter": adapter_name}
        
    except Exception as e:
        # 핫스왑 중 오류가 나면 500 에러 반환
        raise HTTPException(status_code=500, detail=f"배포 및 핫스왑 중 오류 발생: {str(e)}")
        
    finally:
        # (신규 추가 7) 작업이 끝나면 (성공하든 실패하든) 잠금 해제
        model_lock.release()
        print("--- [Deploy] 잠금 해제. ---")

@app.post("/upload-dataset")
async def create_training_data_from_csv(file: UploadFile = File(...)):
    """
    CSV 파일을 업로드받아 'prepare_training_data.py' 로직을 수행하고
    서버 로컬에 'final_training_data.json' 파일로 저장합니다.
    """
    # 0. 훈련 중에는 이 엔드포인트 사용 불가 (VRAM 충돌 방지)
    # (나중에는 상태 관리가 필요하지만, 지금은 '수동 전환'을 신뢰합니다.)
    
    # 1. 파일 형식 검사
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드할 수 있습니다.")

    # 2. 캐시된 스키마 딕셔너리 가져오기
    schema_dict = model_cache.get("schema_dict")
    if not schema_dict:
        raise HTTPException(status_code=500, detail="서버에 스키마 딕셔너리가 로드되지 않았습니다. 서버를 재시작하세요.")

    print(f"\n--- CSV 파일 수신: {file.filename} ---")
    processed_entries = []
    error_logs = []
    
    try:
        # 3. 업로드된 파일 내용을 텍스트로 읽기
        contents = await file.read()
        decoded_content = contents.decode('utf-8')
        csv_file = io.StringIO(decoded_content)
        csv_reader = csv.reader(csv_file)

        # 4. prepare_training_data.py 로직 수행
        try:
            header = next(csv_reader)
        except StopIteration:
            raise HTTPException(status_code=400, detail="CSV 파일이 비어있습니다.")
            
        for i, row in enumerate(csv_reader, start=2):
            try:
                if not row or len(row) < 4:
                    continue # 빈 줄이나 짧은 줄 건너뛰기
                    
                status = row[0].strip()
                
                # "검토완료" 항목만 처리
                if status == "검토완료":
                    question = row[1].strip()
                    sql_response = row[2].strip()
                    table_names = [name.strip() for name in row[3:] if name.strip()]
                    
                    schema_parts = []
                    missing_tables = False
                    for table_name in table_names:
                        if table_name in schema_dict:
                            schema_parts.append(schema_dict[table_name])
                        else:
                            missing_tables = True
                            log = f"경고: {i}행 - 테이블 '{table_name}'의 스키마를 찾을 수 없습니다."
                            print(log)
                            error_logs.append(log)
                    
                    schema_string = "\n".join(schema_parts)
                    if not schema_parts and missing_tables:
                         schema_string = "### ERROR: Referenced schemas not found ###"
                    
                    instruction = f"### Schema:\n{schema_string}\n\n### Question:\n{question}"
                    final_sql_response = sql_response.replace("{lang}", "#{lang}")
                    
                    data_entry = {
                        "instruction": instruction,
                        "response": final_sql_response
                    }
                    processed_entries.append(data_entry)
                    
            except Exception as e:
                log = f"경고: CSV {i}행 처리 중 예외 발생: {e}"
                print(log)
                error_logs.append(log)

    except Exception as e:
        print(f"--- ❌ CSV 파일 파싱 중 치명적 오류: {e} ---")
        raise HTTPException(status_code=500, detail=f"CSV 파일 처리 중 오류 발생: {str(e)}")

    # 2. (수정) 로컬 저장을 GCS 업로드로 변경
    if not processed_entries:
        return {"message": "처리할 '검토완료' 항목이 없습니다.", "processed_count": 0}

    try:
        # JSONL 데이터를 메모리 상의 문자열로 만듭니다.
        jsonl_content = "\n".join([json.dumps(entry, ensure_ascii=False) for entry in processed_entries])
        
        # GCS 클라이언트 초기화 (VM 서비스 계정으로 자동 인증)
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_JSON_PATH)
        
        # GCS로 문자열 업로드
        blob.upload_from_string(jsonl_content, content_type='application/jsonl')
        
        print(f"--- GCS 업로드 완료: {GCS_JSON_PATH} ---")

    except Exception as e:
        print(f"--- ❌ GCS 업로드 중 오류: {e} ---")
        raise HTTPException(status_code=500, detail=f"GCS 파일 업로드 중 오류 발생: {str(e)}")

    return {
        "message": f"처리 완료: 총 {len(processed_entries)}개 항목을 GCS('{GCS_JSON_PATH}')에 저장했습니다.",
        "gcs_path": f"gs://{GCS_BUCKET_NAME}/{GCS_JSON_PATH}",
        "processed_count": len(processed_entries),
        "errors": error_logs
    }


# --- 2. SQL 생성 엔드포인트 ---
@app.post("/verify-model")
async def verify_sql_generation(request: SQLRequest):
    # (신규 추가 8) 배포 작업이 모델을 교체하는 중이면 대기
    if not model_lock.acquire(timeout=5):
        raise HTTPException(status_code=503, detail="서버가 모델 배포 작업으로 바쁩니다. 잠시 후 다시 시도하세요.")
        
    try:
        if "model" not in model_cache or "tokenizer" not in model_cache:
            raise HTTPException(status_code=500, detail="모델이 VRAM에 로드되지 않았습니다. /deploy-adapter를 먼저 실행하거나 서버를 재시작하세요.")

        tokenizer = model_cache["tokenizer"]
        model = model_cache["model"]
        full_schema = model_cache["full_schema_sql"]
        
        # ... (이하 추론 로직은 기존과 동일) ...
        print(f"\n--- 요청 수신: {request.prompt} (테이블: {request.tables}) ---")
        filtered_schema = extract_schemas(full_schema, request.tables)
        if not filtered_schema:
            raise HTTPException(status_code=400, detail=f"입력된 테이블({request.tables}) 중 유효한 스키마를 찾지 못했습니다.")

        input_text = (f"{SQL_PREFIX}### Schema:\n{filtered_schema}\n\n### Question:\n{request.prompt}")
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=TRAIN_MAX_SEQ_LENGTH).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=512, num_beams=5, early_stopping=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
            )
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"generated_sql": generated_sql}

    except Exception as e:
        print(f"--- ❌ 추론 중 오류 발생: {e} ---")
        raise HTTPException(status_code=500, detail=f"SQL 생성 중 서버 오류 발생: {str(e)}")
    
    finally:
        # (신규 추가 9) 추론이 끝나면 잠금 해제
        model_lock.release()

# --- 3. 서버 실행 (터미널에서 uvicorn으로 직접 실행) ---
if __name__ == "__main__":
    # 이 파일(`main.py`)이 있는 디렉토리에서
    # 'uvicorn main:app --host 0.0.0.0 --port 8000' 명령을 실행하세요.
    
    # ⚠️ TRAIN_MAX_SEQ_LENGTH 값을 훈련 시 설정과
    #    동일하게 맞췄는지 다시 한번 확인하세요. (기본값: 1024)
    
    print("--- 서버를 시작하려면 터미널에서 다음 명령을 실행하세요: ---")
    print("uvicorn main:app --host 0.0.0.0 --port 8000")
    print("--- --------------------------------------------- ---")