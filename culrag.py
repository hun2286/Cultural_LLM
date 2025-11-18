# 스캔본 제외하고 PDF RAG 작업 코드 (출처 랭킹 포함)

import os
import sys
import time
import fitz
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage

# 환경 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_folder = r"C:\Users\user\Desktop\pdfs\유산청pdf자료\자연유산"
persist_dir = "./Natural_db"

# LLM / 임베딩 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=8000,
    openai_api_key=api_key
)

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# PDF → 마크다운 변환
def pdf_to_markdown(pdf_path):
    md_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for i, page in enumerate(pdf):
                # ★ 블록 대신 전체 텍스트 한 번에 추출
                page_text = page.get_text("text")

                # 페이지 헤더 추가
                md_text += f"# Page {i + 1}\n\n"

                # 그대로 본문 추가
                if page_text.strip():
                    md_text += page_text + "\n\n"

        return md_text

    except Exception:
        print(f"[오류] PDF 변환 실패 : {pdf_path}") 
        return ""

def load_pdf_safe(pdf_path):
    md_text = pdf_to_markdown(pdf_path)
    if not md_text.strip():  # 완전히 비었을 때만 제외
        return None
    return [Document(
        page_content=md_text,
        metadata={"source": os.path.splitext(os.path.basename(pdf_path))[0]}
    )]

# 전체 PDF 로드
def load_all_pdfs(pdf_folder):
    all_docs = []
    failed_pdfs = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    print(f"총 PDF 파일 수: {len(pdf_files)}\n")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            docs = load_pdf_safe(pdf_path)
            if not docs:
                failed_pdfs.append(pdf_file)
                print(f"{pdf_file} → 텍스트 부족 / 청크 생성 불가")
            else:
                all_docs.extend(docs)
                print(f"{pdf_file} 처리 완료 ({len(docs)} 문서)")
        except Exception as e:
            print(f"{pdf_file} 처리 중 오류: {e}")
            failed_pdfs.append(pdf_file)
    
    print(f"\n총 Document 수: {len(all_docs)}")
    if failed_pdfs:
        print("\n--- 텍스트 변환 실패 / 청크 생성 불가 PDF 목록 ---")
        for f in failed_pdfs:
            print(f"- {f}")
    
    return all_docs

# Vector DB 구축 (DB 재사용)
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    print("DB 새로 생성")
    docs = load_all_pdfs(pdf_folder)
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=300)
        split_docs = text_splitter.split_documents(docs)
        print(f"총 청크 수: {len(split_docs)}")
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        vectorstore.persist()
    else:
        print("청크 생성 가능한 문서가 없습니다. DB 생성 생략.")
        vectorstore = None
else:
    print("기존 DB를 재사용")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}) if vectorstore else None

# RAG 답변 함수
def rag_answer(question):
    if not retriever:
        return "DB가 없습니다. 먼저 텍스트가 있는 PDF를 처리하세요."

    # 검색
    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    # LLM 입력용 컨텍스트
    context_texts = [doc.page_content.strip() for doc in retriever_docs if doc.page_content.strip()]
    context = "\n\n".join(context_texts)

    # LLM 질의응답
    messages = [
        SystemMessage(content="""
            당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
            - 문서 내용만 활용해 답변하세요.
            - 문서에 없는 내용은 추가하지 말고, 없으면 '정보 없음'이라고 표시하세요.
            - 각 항목은 제목 내용 한 줄 빈 줄 순서로 작성하세요.
            - 최소 300단어 이상 작성하고, 가능한 한 문서 내용을 통합하세요.
        """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    answer = response.content

    # 후처리: 연속 빈 줄 1줄로 축소 + 제목/번호 제거
    lines = answer.split("\n")
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                cleaned_lines.append("")
            prev_empty = True
        else:
            stripped = re.sub(r"^[#*\d\.\s]+", "", stripped)
            cleaned_lines.append(stripped)
            prev_empty = False
    cleaned_answer = "\n".join(cleaned_lines).strip()

    # 답변이 없으면 "정보 없음"만 반환
    if not cleaned_answer or cleaned_answer.lower() == "정보 없음":
        return "정보 없음"

    # 답변이 있으면 retriever_docs 기반 출처 표시
    used_sources = []
    for doc in retriever_docs:
        src = doc.metadata.get("source", "출처 없음")
        if src not in used_sources:
            used_sources.append(src)

    # 최소 1개 강제 표시
    if not used_sources and retriever_docs:
        used_sources.append(retriever_docs[0].metadata.get("source", "출처 없음"))

    # 최종 출력
    output = cleaned_answer + "\n\n" + ("-" * 60) + "\n"
    for i, s in enumerate(used_sources):
        output += f"[출처: {s}]"
        if i < len(used_sources) - 1:
            output += "\n"

    return output

# GPT처럼 한 글자씩 출력
def typewriter_print(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# 실행
if __name__ == "__main__":
    print("=" * 60)
    print("RAG 질의응답 시스템")
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
