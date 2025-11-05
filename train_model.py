import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig  # v1의 경고를 해결한 8-bit 로더
)
from peft import get_peft_model, LoraConfig, TaskType

from google.cloud import storage
import datetime

# --- GCS 설정 ---
GCS_BUCKET_NAME = "text2sql-pipeline-bucket" # ⬅️ 2단계에서 만드신 버킷 이름
GCS_JSON_PATH = "datasets/final_training_data.json" # ⬅️ 다운로드할 JSON 경로
LOCAL_JSON_PATH = "./downloaded_training_data.json" # ⬅️ 로컬에 저장할 이름
# ----------------

# T4 GPU용 float16
DTYPE = torch.float16

def preprocess_function(examples, tokenizer, max_seq_length, prefix="SQL 쿼리 생성: "):
    """
    데이터셋을 T5 모델 입력 형식에 맞게 전처리합니다. (v2 방식)
    'instruction' 필드를 그대로 사용합니다.
    """
    inputs = [prefix + doc for doc in examples["instruction"]]
    targets = [doc for doc in examples["response"]]

    # 입력 토큰화
    model_inputs = tokenizer(
        inputs, 
        max_length=max_seq_length, 
        truncation=True,  # max_seq_length를 초과하면 잘라냅니다 (Killed 방지)
        padding=False
    )

    # v1의 경고를 해결한 레이블 토큰화 방식
    labels = tokenizer(
        text_target=targets, 
        max_length=512, 
        truncation=True, 
        padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- (신규) GCS 어댑터 업로드 헬퍼 함수 ---
def upload_directory_to_gcs(local_directory, gcs_bucket, gcs_destination_prefix):
    """로컬 디렉토리 전체를 GCS로 업로드합니다."""
    try:
        print(f"--- GCS로 어댑터 업로드 시작: {local_directory} -> gs://{gcs_bucket.name}/{gcs_destination_prefix} ---")
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket.name) # 버킷 이름으로 객체 다시 가져오기
        
        for root, dirs, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                
                # GCS 내의 상대 경로 계산
                relative_path = os.path.relpath(local_path, local_directory)
                gcs_path = os.path.join(gcs_destination_prefix, relative_path)
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to {gcs_path}")
                
        print("--- GCS 어댑터 업로드 완료 ---")
    except Exception as e:
        print(f"--- ❌ GCS 어댑터 업로드 실패: {e} ---")

def train(args):
    """
    모델 학습을 수행하는 메인 함수
    """
    
    # --- (수정 1) GCS 클라이언트와 버킷을 함수 상단에서 정의 ---
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
    except Exception as e:
        print(f"--- ❌ GCS 클라이언트 초기화 실패: {e} ---")
        return
    # ---------------------------------------------------

    # --- 1. GCS에서 훈련 데이터 다운로드 ---
    try:
        print(f"--- GCS에서 훈련 데이터 다운로드: {GCS_JSON_PATH} ---")
        blob = bucket.blob(GCS_JSON_PATH)
        blob.download_to_filename(LOCAL_JSON_PATH)
        print(f"--- 다운로드 완료: {LOCAL_JSON_PATH} ---")
    except Exception as e:
        print(f"--- ❌ GCS 다운로드 실패: {e} ---")
        print("--- 훈련을 계속할 수 없습니다. ---")
        return
    # ----------------------------------------

    print(f"토크나이저 로드 중: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- (수정 2) 'args.train_file' 대신 다운로드한 'LOCAL_JSON_PATH' 사용 ---
    print(f"데이터셋 로드 중: {LOCAL_JSON_PATH}")
    dataset = load_dataset("json", data_files=LOCAL_JSON_PATH, split="train")
    # -----------------------------------------------------------------
    
    dataset = dataset.shuffle(seed=42)

    print("데이터셋 전처리 중...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": args.max_seq_length},
        remove_columns=dataset.column_names,
    )
    print(f"전처리 완료. 총 샘플 수: {len(tokenized_dataset)}")

    # ... (bnb_config, model, lora_config, training_args, data_collator, trainer... 로직은 동일) ...
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        dtype=DTYPE,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q", "v"],
        lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
    )
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

    # --- 2. GCS로 학습된 어댑터 업로드 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gcs_adapter_path = f"adapters/adapter-{timestamp}"
    
    # (수정 1에서 정의한 'bucket' 변수를 여기서 사용)
    upload_directory_to_gcs(args.output_dir, bucket, gcs_adapter_path)
    # -----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="T5 Text-to-SQL 모델 (GCS 연동)")
    # --train_file 인자 제거됨
    
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
    parser.add_argument("--max_seq_length", type=int, default=1024, # <-- 1536으로 다시 시도
                        help="최대 입력 시퀀스 길이 (스키마+질문). T4 VRAM에 따라 조절 필요.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # train 함수가 args를 받도록 수정했다면 train(args) 호출
    train(args)

if __name__ == "__main__":
    main()