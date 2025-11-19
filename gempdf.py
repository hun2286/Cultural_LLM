import os
import sys
import time
import fitz
import re
import socket
import torch
from dotenv import load_dotenv
from typing import Optional, Union

# LangChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# HuggingFace + LangChain Pipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Gemma-3 모델 로더
def load_hf_causal_lm_pipeline(
        repo_id : str,
        model_dir : Optional[Union[str, os.PathLike]] = None,
        task : str = "text-generation",
        max_new_tokens : int = 1024,
        temperature : float = 0.6,
        top_p : float = 0.9,
        top_k : int = 20,
        repetition_penalty : float = 1.2,
        torch_dtype = torch.bfloat16,
        expose_prompt : bool = False
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


# 환경 설정
load_dotenv()

pdf_folder = r"/volume/bgr_storage/embedding_data/유산별 보고서/문화유산"
persist_dir = "./Cultural_db"

# Gemma-3 로컬 모델 설정
print("Gemma-3 로딩 중...")

llm = load_hf_causal_lm_pipeline(
    repo_id="unsloth/gemma-3-12b-it-bnb-4bit",
    max_new_tokens=1024,
    temperature=0.3
)

print("Gemma-3 초기화 완료!")

# 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# PDF → Markdown 변환
def pdf_to_markdown(pdf_path):
    md_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for i, page in enumerate(pdf):
                page_text = page.get_text("text")

                md_text += f"# Page {i + 1}\n\n"
                if page_text.strip():
                    md_text += page_text + "\n\n"

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

# 재귀 탐색 PDF 로딩
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
            chunk_size=1300,
            chunk_overlap=300
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
    search_kwargs={"k": 3}
) if vectorstore else None

# Gemma-3 IT RAG 질의응답
def build_gemma_prompt(context, question):
    return f"""
<start_of_turn>system
당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
- 문서 내용만 활용해 답변하세요.
- 문서에 없는 내용은 '정보 없음'이라고 하세요.
- 최소 300단어 이상 작성하세요.
<end_of_turn>

<start_of_turn>user
문서 내용:
{context}

질문:
{question}
<end_of_turn>

<start_of_turn>model
""".strip()


def rag_answer(question):
    if not retriever:
        return "DB가 없습니다. 먼저 문서를 임베딩하세요."

    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    context_text = "\n\n".join([
        doc.page_content.strip()
        for doc in retriever_docs if doc.page_content.strip()
    ])

    # Gemma-3 프롬프트 생성
    prompt = build_gemma_prompt(context_text, question)

    # LLM 호출
    response = llm.invoke(prompt)
    answer = response.strip()

    # 후처리
    lines = answer.split("\n")
    cleaned = []
    prev_empty = False

    for line in lines:
        s = line.strip()
        if not s:
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
        else:
            s = re.sub(r"^[#*\d\.\s]+", "", s)
            s = s.replace("**", "")
            cleaned.append(s)
            prev_empty = False

    cleaned_answer = "\n".join(cleaned).strip()
    if not cleaned_answer:
        cleaned_answer = "정보 없음"

    # 출처
    used_sources = []
    for d in retriever_docs:
        src = d.metadata.get("source", "출처 없음")
        if src not in used_sources:
            used_sources.append(src)

    output = cleaned_answer + "\n\n" + "-"*60 + "\n"
    for s in used_sources:
        output += f"[출처: {s}]\n"

    return output.strip()

# 타이핑 출력
def typewriter_print(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

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
        typewriter_print(answer, delay=0.02)
        print("-" * 60)
