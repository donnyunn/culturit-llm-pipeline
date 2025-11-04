import argparse
import os
import re # <-- 정규식 라이브러리 임포트
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType

DTYPE = torch.float16

# --- (추가 1) verify_model.py에서 파일 읽기 함수 가져오기 ---
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

# --- (수정 1) 전처리 함수 수정 ---
def preprocess_function(examples, tokenizer, max_seq_length, full_schema_str, prefix="SQL 쿼리 생성: "):
    """
    'instruction'에서 질문만 추출하고, '전체 스키마'와 결합하여 프롬프트를 재구성합니다.
    """
    inputs = []
    targets = []
    
    question_pattern = re.compile(r"### Question:\n(.*?)$", re.DOTALL)

    for instruction in examples["instruction"]:
        # instruction에서 '### Question:' 뒷부분(질문)만 추출
        match = question_pattern.search(instruction)
        if match:
            question = match.group(1).strip()
            # (핵심) 전체 스키마와 질문을 결합해 새로운 입력 생성
            new_input_text = (
                f"{prefix}### Schema:\n{full_schema_str}\n\n"
                f"### Question:\n{question}"
            )
            inputs.append(new_input_text)
        else:
            # 패턴을 못찾으면 일단 비워둠 (오류 방지)
            inputs.append(prefix) 
    
    targets = [doc for doc in examples["response"]]

    model_inputs = tokenizer(
        inputs, 
        max_length=max_seq_length, 
        truncation=True, 
        padding=False
    )

    labels = tokenizer(
        text_target=targets, 
        max_length=512, 
        truncation=True, 
        padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- (수정 2) train 함수 수정 ---
def train(args):
    """
    모델 학습을 수행하는 메인 함수
    """

    # --- (추가 2) 전체 스키마 파일 미리 읽어오기 ---
    print(f"전체 스키마 로드 중: {args.schema_file}")
    FULL_SCHEMA = read_file_content(args.schema_file)
    # ----------------------------------------------

    print(f"토크나이저 로드 중: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"데이터셋 로드 중: {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    dataset = dataset.shuffle(seed=42)

    print("데이터셋 전처리 중 (전체 스키마 적용)...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        # (핵심) full_schema_str 인자를 fn_kwargs로 넘겨줌
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_seq_length": args.max_seq_length,
            "full_schema_str": FULL_SCHEMA # <-- 스키마 전달
        },
        remove_columns=dataset.column_names,
    )
    print(f"전처리 완료. 총 샘플 수: {len(tokenized_dataset)}")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"기본 모델 로드 중 (8-bit 양자화): {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        dtype=DTYPE,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    print("LoRA 적용 중...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none", 
        dataloader_num_workers=0 
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("--- 모델 학습 시작 ---")
    trainer.train()
    print("--- 모델 학습 완료 ---")

    print(f"LoRA 어댑터 및 토크나이저 저장: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

# --- (수정 3) main 함수 수정 ---
def main():
    parser = argparse.ArgumentParser(description="T5 Text-to-SQL 모델을 LoRA로 미세 조정합니다.")
    
    # --- (추가 3) schema_file 인자 추가 ---
    parser.add_argument("--schema_file", type=str, required=True, 
                        help="전체 schema.sql 파일 경로")
    # -------------------------------------
    
    parser.add_argument("--train_file", type=str, required=True, 
                        help="학습용 JSONL 파일 경로 (e.g., final_training_data.json)")
    
    parser.add_argument("--model_name", type=str, default="paust/pko-t5-base", 
                        help="기본 T5 모델 (Hugging Face Hub)")
    parser.add_argument("--output_dir", type=str, default="./sql-lora-adapter", 
                        help="학습된 LoRA 어댑터를 저장할 디렉토리")
    parser.add_argument("--epochs", type=int, default=100, # <-- 100 에포크 유지
                        help="총 학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Per-device 배치 크기 (T4는 1 또는 2 권장)")
    parser.add_argument("--grad_accum", type=int, default=8, 
                        help="Gradient accumulation steps (실질 배치 크기 = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4, 
                        help="학습률 (LoRA는 일반 fine-tuning보다 높게 설정)")
    parser.add_argument("--max_seq_length", type=int, default=1536, # <-- 1536으로 다시 시도
                        help="최대 입력 시퀀스 길이 (스키마+질문). T4 VRAM에 따라 조절 필요.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()