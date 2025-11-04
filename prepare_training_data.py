import csv
import json
import re
import sys

# 입력 파일 및 출력 파일 이름
SCHEMA_FILE = 'schema.sql'
CSV_FILE = 'customer_dataset.csv'
OUTPUT_FILE = 'final_training_data.json'

def parse_schema(schema_content):
    """
    .sql 파일 내용을 파싱하여 테이블 이름과 CREATE TABLE 구문을 매핑하는 딕셔너리를 반환합니다.
    (수정됨: re.findall을 사용하여 CREATE TABLE ... ; 블록을 직접 찾음)
    """
    schema_dict = {}
    
    # 'CREATE TABLE'로 시작해서 세미콜론(;)으로 끝나는 모든 블록을 찾습니다.
    # re.DOTALL: .이 줄바꿈 문자(\n)도 포함하도록 합니다.
    # re.IGNORECASE: 대소문자를 구분하지 않습니다.
    statements = re.findall(r'(CREATE TABLE.*?;)', schema_content, re.DOTALL | re.IGNORECASE)
    
    if not statements:
        print(f"경고: '{SCHEMA_FILE}'에서 'CREATE TABLE ... ;' 패턴을 찾을 수 없습니다.", file=sys.stderr)
        
    for statement in statements:
        statement = statement.strip()
        
        # 정규표현식을 사용하여 테이블 이름을 추출합니다.
        # `CREATE TABLE`, `CREATE TABLE IF NOT EXISTS` 모두 처리
        # 백틱(`), 따옴표("), 또는 아무것도 없는 경우(word) 모두 처리
        match = re.search(
            r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?`?\"?(\w+)`?\"?', 
            statement, 
            re.IGNORECASE
        )
        
        if match:
            table_name = match.group(1)
            # 원본 CREATE TABLE 문장(세미콜론 포함)을 딕셔너리에 저장합니다.
            schema_dict[table_name] = statement
        else:
            print(f"경고: 'CREATE TABLE' 문을 찾았으나 테이블 이름을 파싱할 수 없습니다: {statement[:60]}...", file=sys.stderr)
            
    return schema_dict

def process_csv_and_generate_json():
    """
    CSV 파일을 읽고 스키마와 매핑하여 최종 JSON Lines 파일을 생성합니다.
    """
    
    # 1. 스키마 파일 읽기 및 파싱
    try:
        with open(SCHEMA_FILE, 'r', encoding='utf-8') as f:
            schema_content = f.read()
    except FileNotFoundError:
        print(f"오류: 스키마 파일 '{SCHEMA_FILE}'을(를) 찾을 수 없습니다.", file=sys.stderr)
        return
    except Exception as e:
        print(f"오류: 스키마 파일 읽기 중 오류 발생: {e}", file=sys.stderr)
        return

    schema_dict = parse_schema(schema_content)
    
    if not schema_dict:
        # parse_schema 내부에서 이미 경고를 출력했을 것입니다.
        print("경고: 파싱된 스키마가 없습니다. CSV 처리를 계속하지만 스키마 정보가 비어있을 수 있습니다.", file=sys.stderr)

    processed_count = 0
    
    # 2. CSV 파일 읽기 및 JSONL 파일 쓰기
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as csv_file, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as jsonl_file:
            
            csv_reader = csv.reader(csv_file)
            
            try:
                # 헤더 행 건너뛰기 (파일의 첫 번째 줄)
                header = next(csv_reader)
            except StopIteration:
                print(f"오류: CSV 파일 '{CSV_FILE}'이(가) 비어있습니다.", file=sys.stderr)
                return

            # 3. 각 행 처리 (인덱스는 1부터 시작하는 행 번호를 위해 +2)
            for i, row in enumerate(csv_reader, start=2):
                try:
                    # 행이 비어있는 경우 (예: 파일 끝의 빈 줄) 건너뛰기
                    if not row:
                        continue

                    # 행에 데이터가 충분하지 않은 경우
                    if len(row) < 4:
                        print(f"경고: {i}행 - CSV 파일의 형식이 잘못되었습니다. (열 개수 부족)", file=sys.stderr)
                        continue
                        
                    status = row[0].strip()
                    
                    # 4. "검토완료" 상태인지 확인
                    if status == "검토완료":
                        question = row[1].strip()
                        sql_response = row[2].strip()
                        
                        # 5. 테이블 이름 목록 추출 (4번째 열[인덱스 3]부터 끝까지)
                        table_names = [name.strip() for name in row[3:] if name.strip()]
                        
                        # 6. 동적 스키마 문자열 생성
                        schema_parts = []
                        missing_tables = False
                        for table_name in table_names:
                            if table_name in schema_dict:
                                schema_parts.append(schema_dict[table_name])
                            else:
                                print(f"경고: {i}행 - 테이블 '{table_name}'의 스키마를 '{SCHEMA_FILE}'에서 찾을 수 없습니다.", file=sys.stderr)
                                missing_tables = True
                        
                        # 스키마가 하나도 없으면 최종 JSON에 빈 문자열이 아닌 경고문 추가
                        if not schema_parts and missing_tables:
                             schema_string = "### ERROR: Referenced schemas not found ###"
                        else:
                             schema_string = "\n".join(schema_parts)
                        
                        # 7. instruction 생성
                        instruction = f"### Schema:\n{schema_string}\n\n### Question:\n{question}"
                        
                        # 8. response(SQL) 변환
                        final_sql_response = sql_response.replace("{lang}", "#{lang}")
                        
                        # 9. JSON 객체 생성
                        data_entry = {
                            "instruction": instruction,
                            "response": final_sql_response
                        }
                        
                        # 10. JSONL 파일에 쓰기 (ensure_ascii=False로 한글 유지)
                        jsonl_file.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
                        processed_count += 1
                        
                except Exception as e:
                    # 각 행 처리 중 예상치 못한 오류 발생 시
                    print(f"경고: {i}행 처리 중 예외 발생: {e}", file=sys.stderr)

    except FileNotFoundError:
        print(f"오류: CSV 파일 '{CSV_FILE}'을(를) 찾을 수 없습니다.", file=sys.stderr)
        return
    except Exception as e:
        print(f"오류: CSV 파일 처리 중 치명적 오류 발생: {e}", file=sys.stderr)
        return

    print(f"\n처리 완료: 총 {processed_count}개의 '검토완료' 항목을 '{OUTPUT_FILE}' 파일에 저장했습니다.")

if __name__ == "__main__":
    process_csv_and_generate_json()