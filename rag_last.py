import os
import fitz
import re
import socket
import torch
from typing import Optional, Union
from konlpy.tag import Komoran

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

komoran = Komoran()

# LLM 파이프라인 로드
def load_hf_causal_lm_pipeline(
    repo_id: str,
    model_dir: Optional[Union[str, os.PathLike]] = None,
    task: str = "text-generation",
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
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


# 환경 변수 로드
pdf_folder = r"/volume/bgr_storage/embedding_data/유산별 보고서"
persist_dir = "./database/Cultural_db"

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
    search_kwargs={"k": 3}
) if vectorstore else None

# Gemma-3 Prompt 빌드 (출처 마지막 모아서 표기)
def build_gemma_prompt(context, question):
    return f"""
<start_of_turn>system
당신은 제공된 문서만을 기반으로 질문에 답하는 전문 분석가입니다.
당신의 모든 답변은 아래의 엄격한 지침을 따라야 합니다.

[핵심 지침]
1.  **정보 근거:** 답변은 반드시 제공된 [문서 내용]에 근거해야 합니다. 문서에 내용이 부족하거나 모호하여 **질문의 핵심 주제를 충실히 설명할 수 없을 때만** "정보 없음"이라고 답변하세요.
2.  **답변 형식 (Markdown):** 답변은 **마크다운(Markdown) 형식**을 사용하여 제목(`##`), 목록(`*`), 볼드체(`**`) 등으로 구조화하여 작성하십시오.
3.  **내용 확장 및 상세화 (강화):** 문서 내용이 충분하거나 관련된 정보가 여러 곳에 흩어져 있는 경우, 반드시 단순 요약을 넘어 **다음 요소를 포함하여 질문의 주제를 '완결성 있게' 상세히 설명하십시오.**
    * **핵심 개념의 의미 및 정의**
    * **구체적 사례, 시대적 변화, 역사적 맥락**
    * **관련 정보의 연결 및 종합**
4.  **금지 사항:** 문서에 근거 없는 가정, 추론, 창작, 외부 지식, '참고:', '다음과 같습니다', '설명은 생략합니다' 등의 안내 문구를 절대 포함하지 마십시오.
5.  **출처 표기:** 답변 내용에 출처 표기(예: [출처: PDF 제목])를 절대 포함하지 마십시오.
6.  **줄 바꿈 및 가독성:** 답변의 **가독성을 최우선**으로 확보해야 합니다. **목록 항목(`*`)**의 내용이 길어지더라도, 각 항목 내에서 의미 단위가 끝날 때마다 **줄 바꿈(개행)을 충분히 활용**하여 내용이 밀집되어 보이지 않도록 작성하십시오.
7.  **세부 목록 분리:** 목록 항목(`*`)이 단락처럼 길어지지 않도록, **내용을 여러 개의 짧은 목록(`*`)**으로 최대한 **쪼개서** 작성하십시오.

[답변 구조 예시] (반드시 이 구조를 따르세요. 세부 주제를 활용하여 답변을 풍부하게 작성해야 합니다.)
## [답변의 핵심 주제]
* [첫 번째 핵심 내용]
* [두 번째 핵심 내용]

### [세부 주제 1]
[세부 내용을 마크다운 문단 또는 목록으로 작성]

### [세부 주제 2]
[세부 내용을 마크다운 문단 또는 목록으로 작성]

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
    matches_count = sum(1 for kw in keywords if kw.lower() in title_lower)
    is_match = matches_count == len(keywords)
    return is_match

# RAG 질의응답
def rag_answer(question):
    if not question.strip():  # 질문이 비어 있으면
        return "정보 없음"

    if not retriever:
        return "DB가 없습니다. 먼저 문서를 임베딩하세요."
    
    question_keywords = extract_keywords(question)

    if not question_keywords:
        return "정보없음"

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
    cleaned_answer = re.sub(r'^\s*참고:\s*', '', cleaned_answer)
    cleaned_answer = cleaned_answer.strip()

    if "정보 없음" in cleaned_answer:
        return "정보 없음"

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
