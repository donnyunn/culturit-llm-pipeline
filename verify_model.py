import argparse
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

# T4 GPU에서 추론을 위한 데이터 타입
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

def generate_sql(args):
    """
    학습된 LoRA 어댑터를 로드하고 질문에 대한 SQL을 생성합니다.
    """
    
    # 1. 8-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    # 2. 토크나이저 로드 (학습 시 저장한 어댑터 폴더에서)
    print(f"토크나이저 로드 중: {args.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    # 3. 기본 모델 로드 (8-bit)
    print(f"기본 모델 로드 중 (8-bit): {args.base_model}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        dtype=DTYPE,
        device_map="auto",
    )

    # 4. LoRA 어댑터 적용
    print(f"LoRA 어댑터 로드 중: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval() # 추론 모드로 설정
    
    print("\n[ 모델 로드 완료 ]")

    # 5. 스키마 및 질문으로 프롬프트 구성
    # 모델은 학습 때와 '똑같은' 형식의 입력을 받아야 합니다.
    # schema.sql 파일 전체와 질문을 조합합니다.
    schema_content = read_file_content(args.schema_file)
    
    prefix = "SQL 쿼리 생성: " # 학습 때 사용한 접두사
    input_text = (
        f"{prefix}### Schema:\n{schema_content}\n\n"
        f"### Question:\n{args.question}"
    )

    print("-" * 30)
    print(f"입력 질문: {args.question}")
    print(f"사용 스키마: {args.schema_file}")
    print("-" * 30)

    # 6. 입력 토큰화
    # T4의 VRAM에 맞게 입력을 잘라냅니다 (train_model.py의 max_seq_length와 동일하게)
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=args.max_seq_length
    ).to("cuda") # T4 GPU로 이동

    # 7. SQL 생성 (추론)
    # torch.no_grad()로 불필요한 그래디언트 계산을 방지하여 VRAM 절약
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,  # 생성할 SQL의 최대 길이
            num_beams=5,     # 빔 서치 (더 나은 결과)
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # 8. 결과 디코딩 및 출력
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n[ 생성된 SQL ]")
    print("=" * 30)
    print(generated_sql)
    print("=" * 30)


def main():
    parser = argparse.ArgumentParser(description="학습된 Text-to-SQL LoRA 어댑터를 테스트합니다.")
    
    # 필수 인자
    parser.add_argument("--question", type=str, required=True, 
                        help="모델에게 물어볼 자연어 질문 (예: '개발팀 직원 알려줘')")
    parser.add_argument("--schema_file", type=str, required=True, 
                        help="질문의 문맥이 되는 schema.sql 파일 경로")
    
    # 선택 인자 (기본값)
    parser.add_argument("--adapter_dir", type=str, default="./sql-lora-adapter",
                        help="학습된 LoRA 어댑터가 저장된 디렉토리")
    parser.add_argument("--base_model", type=str, default="paust/pko-t5-base",
                        help="학습에 사용된 기본 T5 모델")
    parser.add_argument("--max_seq_length", type=int, default=1536, 
                        help="입력 시퀀스 최대 길이 (학습 시 설정과 동일해야 함)")

    args = parser.parse_args()
    generate_sql(args)

if __name__ == "__main__":
    main()