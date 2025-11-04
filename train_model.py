# train_model.py (KoBART 최종 복구 버전)

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    BartForConditionalGeneration, # <--- MT5 대신 BartForConditionalGeneration 사용
    AutoTokenizer,             
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import os

# --- 설정 변수 ---
MODEL_NAME = "gogamza/kobart-base-v2" # <--- KoBART 모델명 사용
DATA_PATH = './final_training_data.json'
# Compute Engine 환경 변수 설정
OUTPUT_DIR = os.environ.get('AIP_MODEL_DIR', './kobart_model_output') 
MAX_LEN = 1024 
NUM_EPOCHS = 50 
BATCH_SIZE = 4 

# --- 1. 데이터셋 로드 및 토큰화 ---
def prepare_data(data_path, tokenizer):
    print("1. Loading and Tokenizing Dataset...")
    
    df = pd.read_json(data_path, lines=True)
    dataset = Dataset.from_pandas(df)
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
    
    def tokenize_function(examples):
        input_texts = examples['MODEL_INPUT']
        target_texts = examples['SQL']
        
        # 입력 토큰화
        model_inputs = tokenizer(input_texts, max_length=MAX_LEN, padding="max_length", truncation=True)
        
        # 출력(정답) 토큰화
        with tokenizer.as_target_tokenizer(): 
            labels = tokenizer(target_texts, max_length=MAX_LEN, padding="max_length", truncation=True).input_ids
        
        model_inputs["labels"] = labels
        return model_inputs
    
    tokenized_datasets = dataset_split.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['natural_language', 'SQL', 'TABLES', 'SCHEMA_ENCODING', 'MODEL_INPUT'])
    
    return tokenized_datasets['train'], tokenized_datasets['test']

# --- 2. 학습 실행 ---
def run_training():
    # 1. 토크나이저 및 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # KoBART는 BART 기반이므로 AutoModelForSeq2SeqLM 대신 BartForConditionalGeneration 사용 (확실한 임포트)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # 2. GPU 사용 설정 및 모델 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"2. Using device: {device}. CUDA available: {torch.cuda.is_available()}")
    
    # 3. 데이터 준비
    train_dataset, eval_dataset = prepare_data(DATA_PATH, tokenizer)
    
    # 4. 학습 설정 (eval_strategy 수정 반영)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  # <-- 수정된 올바른 인수
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(), 
        predict_with_generate=True,
    )
    
    # 5. Data Collator 설정
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='max_length')

    # 6. Trainer 객체 생성 및 학습 시작
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("3. Starting KoBART Fine-tuning...")
    trainer.train()
    
    # 7. 최종 모델 저장
    model.save_pretrained(OUTPUT_DIR + "/final_model")
    tokenizer.save_pretrained(OUTPUT_DIR + "/final_model")
    print(f"4. Model saved to {OUTPUT_DIR}/final_model")

if __name__ == '__main__':
    run_training()