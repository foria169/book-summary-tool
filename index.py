import os
import torch
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# Flask 앱 설정
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# KoBART 모델 로드
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = 'digit82/kobart-summarization'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 모델을 평가 모드로 설정하고 CPU로 이동
model.eval()
model = model.to('cpu')

def preprocess_text(text):
    """텍스트 전처리 함수"""
    # 기본 클리닝
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
    text = re.sub(r'[\n\t\r]', ' ', text)  # 개행문자 제거
    
    # 문장 부호 정리
    text = re.sub(r'\.{2,}', '.', text)  # 연속된 마침표 정리
    text = re.sub(r'\s*\.\s*', '. ', text)  # 마침표 주변 공백 정리
    text = text.strip('. ') + '.'  # 마지막 마침표 보장
    
    return text

def split_into_sentences(text):
    """텍스트를 문장 단위로 분할"""
    # 문장 분할 패턴
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    return [s.strip() + '.' for s in sentences if s.strip()]

def create_chunks(sentences, max_len=1000):
    """문장들을 청크로 결합"""
    chunks = []
    current_chunk = ''
    
    for sentence in sentences:
        # 단일 문장이 max_len을 초과하는 경우
        if len(sentence) > max_len:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 문장을 단어 단위로 분할
            words = sentence.split()
            temp_chunk = ''
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_len:
                    temp_chunk += word + ' '
                else:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = word + ' '
            if temp_chunk:
                chunks.append(temp_chunk.strip())
            current_chunk = ''
            continue
            
        # 현재 청크에 문장을 추가할 수 있는 경우
        if len(current_chunk) + len(sentence) + 1 <= max_len:
            current_chunk += sentence + ' '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_text(text):
    """텍스트 처리 메인 함수"""
    # 텍스트가 비어있는 경우
    if not text or not text.strip():
        return []
        
    # 전처리 및 청크 생성
    clean_text = preprocess_text(text)
    sentences = split_into_sentences(clean_text)
    chunks = create_chunks(sentences, max_len=2000)
    
    return chunks

def read_file_with_encoding(filepath):
    """다양한 인코딩으로 파일 읽기 시도"""
    encodings = ['utf-8', 'cp949', 'euc-kr']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
                if not content.strip():
                    raise ValueError("파일이 비어있습니다.")
                return content
        except UnicodeDecodeError:
            continue
    raise ValueError("지원하는 인코딩으로 파일을 읽을 수 없습니다.")

def summarize_text(text):
    """텍스트 요약 실행"""
    if not text or not text.strip():
        raise ValueError("텍스트가 비어있습니다.")

    # 텍스트 처리 및 청크 분할
    chunks = process_text(text)
    if not chunks:
        raise ValueError("처리할 텍스트가 없습니다.")

    # 각 청크 요약
    summaries = []
    for chunk in chunks:
        if len(chunk.strip()) < 200:
            continue

        # 입력 텍스트 인코딩
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        
        # 요약 생성
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=300,
                min_length=100,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # 생성된 요약문 디코딩
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)

    if not summaries:
        raise ValueError("요약할 만한 충분한 길이의 텍스트가 없습니다.")

    return " ".join(summaries)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # 텍스트 입력 처리
        if 'text' in request.form:
            text = request.form['text'].strip()
            if text:
                summary = summarize_text(text)
                return jsonify({'success': True, 'summary': summary})

        # 파일 업로드 처리
        elif 'file' in request.files:
            file = request.files['file']
            if file and file.filename.endswith('.txt'):
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    text = read_file_with_encoding(filepath)
                    summary = summarize_text(text)
                    return jsonify({'success': True, 'summary': summary})

                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)})
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)

        return jsonify({'success': False, 'error': '텍스트 입력이나 파일을 제공해주세요.'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True, port=5000)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)