import os
import sys
import time
import fitz
import re
import socket
import torch
from dotenv import load_dotenv
from typing import Optional, Union
from konlpy.tag import Komoran

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

komoran = Komoran()

# ---------------------------
# LLM 파이프라인 로드
# ---------------------------
def load_hf_causal_lm_pipeline(
    repo_id: str,
    model_dir: Optional[Union[str, os.PathLike]] = None,
    task: str = "text-generation",
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 20,
    repetition_penalty: float = 1.2,
    torch_dtype=torch.bfloat16,
    expose_prompt: bool = False
):
    hostname = socket.gethostname()
    if hostname == 'ubuntu' and model_dir is None:
        model_dir = "/volume/hf_cache/hub"

    model = AutoModelForCausalLM.from_pretrained(
        repo_id, cache_dir=model_dir, torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, cache_dir=model_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        return_full_text=expose_prompt
    )
    return HuggingFacePipeline(pipeline=pipe)


# ---------------------------
# 환경 변수 로드
# ---------------------------
load_dotenv()

pdf_folder = r"/volume/bgr_storage/embedding_data/유산별 보고서/무형유산"
persist_dir = "./Intangible_db"

print("Gemma-3 로딩 중...")
llm = load_hf_causal_lm_pipeline(
    repo_id="unsloth/gemma-3-12b-it-bnb-4bit"
)
print("Gemma-3 초기화 완료!")

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# PDF 텍스트 전처리
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r',,+', ',', text)
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    text = re.sub(r'\s*,\s*\n', '\n', text)
    return text.strip()


def pdf_to_markdown(pdf_path):
    md_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                page_text = page.get_text("text")
                if page_text.strip():
                    md_text += clean_text(page_text) + "\n\n"
        return md_text
    except Exception:
        print(f"[오류] PDF 변환 실패 : {pdf_path}")
        return ""


def load_pdf_safe(pdf_path):
    md_text = pdf_to_markdown(pdf_path)
    if not md_text.strip():
        return None
    return [Document(
        page_content=md_text,
        metadata={"source": os.path.splitext(os.path.basename(pdf_path))[0]}
    )]


def load_all_pdfs_recursive(root_folder):
    all_docs = []
    failed_pdfs = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        pdf_files = [f for f in filenames if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dirpath, pdf_file)
            try:
                docs = load_pdf_safe(pdf_path)
                if not docs:
                    failed_pdfs.append(pdf_path)
                    print(f"{pdf_path} → 텍스트 부족")
                else:
                    all_docs.extend(docs)
                    print(f"{pdf_path} 처리 완료 ({len(docs)} 문서)")
            except Exception as e:
                print(f"{pdf_path} 처리 오류: {e}")
                failed_pdfs.append(pdf_path)

    print(f"\n총 Document 수: {len(all_docs)}")
    if failed_pdfs:
        print("\n--- 텍스트 변환 실패 PDF ---")
        for f in failed_pdfs:
            print("-", f)
    return all_docs

# VectorDB 생성/로드
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    print("DB 새로 생성합니다...")
    docs = load_all_pdfs_recursive(pdf_folder)

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250
        )
        split_docs = text_splitter.split_documents(docs)
        print(f"총 청크 수: {len(split_docs)}")

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        vectorstore.persist()
        print("임베딩 완료!")
    else:
        vectorstore = None
else:
    print("기존 DB 사용")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
) if vectorstore else None

# Gemma-3 Prompt 빌드 (출처 마지막 모아서 표기)
def build_gemma_prompt(context, question):
    return f"""
<start_of_turn>system
당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
- 문서 내용만 활용하며, 문서에 없는 내용은 '정보 없음'이라고 답하세요.
- 각 항목은 제목과 내용 한 줄, 빈 줄 순서로 작성하며, 제목은 'OO 소개' 또는 'OO 개요' 형태로 작성하세요.
- 문단 사이 불필요한 쉼표를 제거하고, 문장 끝은 마침표 또는 줄바꿈으로 마무리하세요.
- 답변 마지막에 참고한 PDF 출처를 [출처: PDF 제목] 형태로 모아서 표기하세요. 
- 본문에는 PDF 제목, 보고서 이름, '참고:', '참고문헌' 등의 표현을 포함하지 마세요.
- '삭제', '누락', '삭제됨', '삭제 요청', '설명', '다음과 같습니다' 등의 안내 문장은 포함하지 마세요.
<end_of_turn>
<start_of_turn>user
문서 내용:
{context}

질문:
{question}
<end_of_turn>

<start_of_turn>model
""".strip()

def extract_keywords(question: str):
    nouns = komoran.nouns(question)
    # 길이 1 이상인 명사만 사용
    keywords = [w for w in nouns if len(w) > 1]
    return keywords


def title_matches_keywords(title: str, keywords: list):
    title_lower = title.lower()
    matches = sum(1 for kw in keywords if kw.lower() in title_lower)
    return matches == len(keywords)

# RAG 질의응답
def rag_answer(question):
    if not question.strip():  # 질문이 비어 있으면
        return "정보 없음"

    if not retriever:
        return "DB가 없습니다. 먼저 문서를 임베딩하세요."

    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    # 질문 키워드 추출
    question_keywords = extract_keywords(question)

    # 키워드 매칭 필터 적용
    filtered_docs = []
    for doc in retriever_docs:
        source_title = doc.metadata.get("source", "")
        if title_matches_keywords(source_title, question_keywords):
            filtered_docs.append(doc)

    # 키워드와 매칭되는 문서가 없으면 원본 문서 사용
    if not filtered_docs:
        filtered_docs = retriever_docs

    # 문서 내용과 출처 수집
    context_chunks = []
    sources_set = set()
    for doc in filtered_docs:
        text = doc.page_content.strip()
        if text:
            text = re.sub(r'\[출처:\s*.*?\]', '', text)
            text = re.sub(r'정밀실측조사보고서?', '', text)
            context_chunks.append(text)
            sources_set.add(doc.metadata.get('source', '출처 없음'))

    context_text = "\n\n".join(context_chunks)
    prompt = build_gemma_prompt(context_text, question)
    response = llm.invoke(prompt)
    answer = response.strip()

    cleaned_answer = re.sub(r'(삭제|삭제됨|누락|중간 삭제)', '', answer)
    cleaned_answer = re.sub(r'\[출처:\s*.*?\]', '', cleaned_answer)
    cleaned_answer = re.sub(r'\n{2,}', '\n\n', cleaned_answer)
    cleaned_answer = re.sub(r'^\s*참고:\s*', '', cleaned_answer)
    cleaned_answer = re.sub(r'\s+\n', '\n', cleaned_answer)
    cleaned_answer = cleaned_answer.strip()

    final_sources = [f"[출처: {s}]" for s in sorted(sources_set)]

    if final_sources:
        final_output = f"{cleaned_answer}\n\n{'-'*60}\n" + "\n".join(final_sources)
    else:
        final_output = f"{cleaned_answer}\n\n{'-'*60}\n정보 없음"

    return final_output

# 실행
if __name__ == "__main__":
    print("=" * 60)
    print("Gemma-3 RAG 질의응답 시스템")
    print("=" * 60)

    while True:
        query = input("\n질문을 입력하세요 (exit 입력 시 종료): ").strip()
        if query.lower() == "exit":
            print("프로그램 종료.")
            break

        answer = rag_answer(query)
        print("\n[답변]:\n")
        print(answer)
        print("-" * 60)
