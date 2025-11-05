import argparse
import torch
import re  # <-- 정규식 라이브러리 임포트
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

DTYPE = torch.float16 

def read_file_content(filepath):
    """파일 내용을 읽어 문자열로 반환합니다."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
        exit(1)
    except Exception as e:
        print(f"오류: {filepath} 파일 읽기 실패 - {e}")
        exit(1)

def extract_schemas(full_sql, table_names):
    """
    full_sql 문자열을 파싱하여, table_names 목록에 있는
    테이블의 'CREATE TABLE ...;' 구문만 추출합니다.
    """
    extracted = []
    for table_name in table_names:
        # 'CREATE TABLE table_name (...);' 전체 블록을 찾는 정규식
        # re.IGNORECASE: 대소문자 무시
        # re.DOTALL: .이 줄바꿈 문자(\n)도 포함하도록 함
        pattern = rf"(CREATE TABLE\s+{re.escape(table_name)}\s*\(.*?;)"
        
        match = re.search(pattern, full_sql, re.IGNORECASE | re.DOTALL)
        
        if match:
            extracted.append(match.group(1).strip())
        else:
            print(f"--- 경고: '{table_name}' 테이블의 스키마를 schema.sql에서 찾지 못했습니다.")
            
    return "\n\n".join(extracted)

def generate_sql(args):
    """
    학습된 LoRA 어댑터를 로드하고 질문에 대한 SQL을 생성합니다.
    """
    
    # 1. 모델/토크나이저 로드 (이전과 동일)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    print(f"토크나이저 로드 중: {args.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    print(f"기본 모델 로드 중 (8-bit): {args.base_model}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        dtype=DTYPE,
        device_map="auto",
    )

    print(f"LoRA 어댑터 로드 중: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    
    print("\n[ 모델 로드 완료 ]")

    # 2. (핵심) 스키마 필터링 및 프롬프트 구성
    
    # schema.sql 파일 전체를 읽어옴
    full_schema_sql = read_file_content(args.schema_file)
    
    # 입력받은 테이블 이름 목록 (쉼표로 분리)
    table_list = [name.strip() for name in args.tables.split(',')]
    
    # 전체 스키마에서 필요한 테이블 스키마만 추출
    filtered_schema = extract_schemas(full_schema_sql, table_list)
    
    if not filtered_schema:
        print("--- 오류: 스키마를 전혀 추출하지 못했습니다. 테이블 이름을 확인하세요.")
        return

    prefix = "SQL 쿼리 생성: " # 학습 때 사용한 접두사
    
    # (핵심) 훈련 데이터와 '똑같은' 형식의 프롬프트 완성
    input_text = (
        f"{prefix}### Schema:\n{filtered_schema}\n\n"
        f"### Question:\n{args.question}"
    )

    print("-" * 30)
    print(f"입력 질문: {args.question}")
    print(f"필터링된 테이블: {table_list}")
    print("-" * 30)
    # print(f"생성된 프롬프트 (일부): {input_text[:500]}...") # 디버깅용

    # 3. 입력 토큰화
    # 훈련 시 사용한 max_seq_length와 '반드시' 동일해야 함
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=args.max_seq_length 
    ).to("cuda")

    # 4. SQL 생성 (추론)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512, 
            num_beams=5,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # 5. 결과 디코딩 및 출력
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n[ 생성된 SQL ]")
    print("=" * 30)
    print(generated_sql)
    print("=" * 30)


def main():
    parser = argparse.ArgumentParser(description="학습된 Text-to-SQL LoRA 어댑터를 테스트합니다. (v_final)")
    
    parser.add_argument("--question", type=str, required=True, 
                        help="모델에게 물어볼 자연어 질문")
    parser.add_argument("--schema_file", type=str, required=True, 
                        help="참조할 '전체' schema.sql 파일 경로")
    parser.add_argument("--tables", type=str, required=True, 
                        help="질문과 관련된 테이블 이름 목록 (쉼표로 구분. 예: 'member_master,member_lang')")
    
    parser.add_argument("--adapter_dir", type=str, default="./sql-lora-adapter",
                        help="학습된 LoRA 어댑터가 저장된 디렉토리")
    parser.add_argument("--base_model", type=str, default="paust/pko-t5-base",
                        help="학습에 사용된 기본 T5 모델")
    parser.add_argument("--max_seq_length", type=int, default=1024, # 훈련 시 1024로 했다면 1024
                        help="입력 시퀀스 최대 길이 (train_model.py와 '반드시' 동일해야 함)")

    args = parser.parse_args()
    generate_sql(args)

if __name__ == "__main__":
    main()