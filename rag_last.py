import os
import re
import socket
import torch
import asyncio
from typing import Optional, Union, Any, cast
from functools import partial

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# API 구조의 기존 임포트 경로 사용
from config.settings import settings 
from utils.rag_utils import (
    load_pdf_safe,
    extract_keywords,
    title_matches_keywords
)

# 모델 및 프롬프트
def _load_hf_causal_lm_pipeline(
    repo_id = None,
    model_dir = None, 
    task = "text-generation",
    max_new_tokens = 1024, 
    temperature = 0.3, 
    top_p = 0.9, 
    top_k = 20,
    repetition_penalty = 1.2, 
    torch_dtype=torch.bfloat16, 
    expose_prompt = False
):
    """LLM 파이프라인 로드"""
    if repo_id is None:
        # settings.py에서 repo_id를 가져오므로, 이 부분은 예외 처리만 남김
        raise ValueError("LLM Repository ID must be provided.")
        
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
        task, model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, 
        temperature=temperature, top_p=top_p, top_k=top_k, 
        repetition_penalty=repetition_penalty, do_sample=True, 
        return_full_text=expose_prompt
    )
    return HuggingFacePipeline(pipeline=pipe)

def _load_all_pdfs_recursive(root_folder: str) -> list[Document]:
    """PDF를 재귀적으로 로드하고 Document 리스트 반환 (DB 초기화 시 사용)"""
    all_docs = []
    failed_pdfs = []
    for dirpath, _, filenames in os.walk(root_folder):
        pdf_files = [f for f in filenames if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dirpath, pdf_file)
            try:
                # load_pdf_safe는 rag_utils에서 import됨
                docs = load_pdf_safe(pdf_path) 
                if not docs:
                    failed_pdfs.append(pdf_path)
                else:
                    all_docs.extend(docs)
            except Exception as e:
                failed_pdfs.append(pdf_path)
    return all_docs

def _build_gemma_prompt(context: str, question: str, source_list_text: str) -> str:
    """Gemma-3 프롬프트 (최종 확정된 인라인 태깅 지침 포함)"""
    return f"""
<start_of_turn>system
당신은 제공된 문서만을 기반으로 질문에 답하는 전문 분석가입니다.
당신의 모든 답변은 아래의 엄격한 지침을 따라야 합니다.

[핵심 지침]
1. **정보 근거:** 답변은 반드시 제공된 [문서 내용]에 근거해야 합니다. 문서에 내용이 부족하거나 모호하여 **질문의 핵심 주제를 충실히 설명할 수 없을 때만** "정보 없음"이라고 답변하세요.
2. **답변 형식 (Markdown):** 답변은 **마크다운(Markdown) 형식**을 사용하여 제목(`##`), 목록(`*`), 볼드체(`**`) 등으로 구조화하여 작성하십시오.
3. **내용 확장 및 상세화 (강화):** 문서 내용이 충분하거나 관련된 정보가 여러 곳에 흩어져 있는 경우, 반드시 단순 요약을 넘어 **다음 요소를 포함하여 질문의 주제를 '완결성 있게' 상세히 설명하십시오.**
    * **핵심 개념의 의미 및 정의**
    * **구체적 사례, 시대적 변화, 역사적 맥락**
    * **관련 정보의 연결 및 종합**
4. **금지 사항:** 문서에 근거 없는 가정, 추론, 창작, 외부 지식, '참고:', '다음과 같습니다', '설명은 생략합니다' 등의 안내 문구를 절대 포함하지 마십시오.
5. **출처 표기 (본문):** 답변 내용에 출처 표기(예: [출처: PDF 제목])를 절대 포함하지 마십시오.
6. **줄 바꿈 및 가독성:** 답변의 **가독성을 최우선**으로 확보해야 합니다. **목록 항목(\*)**의 내용이 길어지더라도, 각 항목 내에서 의미 단위가 끝날 때마다 **줄 바꿈(개행)을 충분히 활용**하여 내용이 밀집되어 보이지 않도록 작성하십시오.
7. **세부 목록 분리:** 목록 항목(\*)이 단락처럼 길어지지 않도록, **내용을 여러 개의 짧은 목록(\*)**으로 최대한 **쪼개서** 작성하십시오.
8. **[필수] 인라인 출처 태깅:** 답변의 각 문단 또는 주요 정보 뒤에는 **반드시** 해당 정보를 추출한 문서의 번호(`[1]`, `[2]` 등)를 붙이십시오. 답변에 사용하지 않은 문서의 번호는 절대 사용하지 마십시오.

[제공된 문서 번호 목록]
{source_list_text}

<end_of_turn>
<start_of_turn>user
문서 내용:
{context}
질문:
{question}
<end_of_turn>
<start_of_turn>model
""".strip()

# RAG 시스템 서비스 클래스
class RAGService:
    def __init__(self):
        self.llm: HuggingFacePipeline | None = None
        self.embedding_model: HuggingFaceEmbeddings | None = None
        self.vectorstore: Chroma | None = None
        self.retriever: Any | None = None

    def initialize_sync(self):
        """서버 시작 시 동기적으로 LLM, Embedding, VectorDB를 초기화합니다."""
        print("RAG System 초기화 중...")

        # 1. 임베딩 모델 로드
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )

        # 2. VectorDB 생성/로드
        persist_dir = settings.PERSIST_DIR
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            print("DB 새로 생성합니다...")
            docs = _load_all_pdfs_recursive(settings.PDF_FOLDER)

            if docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=250
                )
                split_docs = text_splitter.split_documents(docs)
                print(f"총 청크 수: {len(split_docs)}")

                self.vectorstore = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embedding_model,
                    persist_directory=persist_dir
                )
                self.vectorstore.persist()
                print("임베딩 완료!")
            else:
                self.vectorstore = None
        else:
            print("기존 DB 사용")
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embedding_model
            )

        # 3. Retriever 설정
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} 
        ) if self.vectorstore else None

        # 4. LLM 로드
        print("Gemma-3 로딩 중...")
        self.llm = _load_hf_causal_lm_pipeline(
            repo_id=settings.LLM_REPO_ID,
            max_new_tokens=settings.LLM_MAX_NEW_TOKENS
        )
        print("Gemma-3 초기화 완료!")
        print("RAG System 초기화 완료!")


    async def rag_answer(self, question: str) -> str:
        """API 엔드포인트에서 호출되는 비동기 함수."""
        if not question.strip():
            return "정보 없음"
        if not self.retriever or not self.llm:
            return "RAG 시스템이 초기화되지 않았습니다."

        # 동기 검색 및 LLM 추론 작업을 별도 스레드에서 실행
        rag_worker = partial(self._rag_worker, question)
        final_output = await asyncio.to_thread(rag_worker) 

        return final_output

    # _rag_worker: 인라인 태그 기반의 출처 필터링 및 제거 로직 통합
    def _rag_worker(self, question: str) -> str:
        """실제 RAG 프로세스를 수행하는 동기 함수."""
        
        # 1. 키워드 추출 및 유효성 검사 (rag_utils.py의 extract_keywords 사용)
        question_keywords = extract_keywords(question)
        if not question_keywords:
            pass # 키워드 추출 실패 시 필터링 없이 검색된 문서 전체를 사용

        # 2. 검색 및 필터링
        retriever_docs = self.retriever.invoke(question)
        retriever_docs = cast(list[Document], retriever_docs)
        
        filtered_docs = []
        for doc in retriever_docs:
            source_title = doc.metadata.get("source", "")
            # rag_utils.py의 title_matches_keywords 사용
            if title_matches_keywords(source_title, question_keywords): 
                filtered_docs.append(doc)

        if not filtered_docs:
            filtered_docs = retriever_docs

        # 3. 컨텍스트 생성 (번호 매기기 및 매핑 추가)
        context_chunks = []
        sources_map = [] # 문서 번호(인덱스+1)와 파일명을 매핑하기 위한 리스트

        for i, doc in enumerate(filtered_docs):
            text = doc.page_content.strip()
            if text:
                text = re.sub(r'\[출처:\s*.*?\]', '', text)
                text = re.sub(r'정밀실측조사보고서?', '', text)
                
                # LLM에게 전달할 문맥에 번호 태그를 붙임
                context_chunks.append(f"[{i+1}] {text}")
                
                # 문서 번호와 파일명을 매핑 (번호는 1부터 시작)
                sources_map.append(doc.metadata.get('source', '출처 없음'))

        context_text = "\n\n".join(context_chunks)
        
        if not context_text.strip():
            return "정보 없음"
            
        # LLM에게 가이드로 전달할 출처 목록 텍스트
        source_list_text = "\n".join([f"[{i+1}]: {s}" for i, s in enumerate(sources_map)])
        
        # LLM 호출: 세 인자 전달
        prompt = _build_gemma_prompt(context_text, question, source_list_text)
        response = self.llm.invoke(prompt)
        answer = response.strip()

        # 4. 답변 후처리 및 최종 출처 추출
        cleaned_answer = re.sub(r'(삭제|삭제됨|누락|중간 삭제)', '', answer)
        cleaned_answer = re.sub(r'\[출처:\s*.*?\]', '', cleaned_answer)
        cleaned_answer = re.sub(r'^\s*참고:\s*', '', cleaned_answer)
        cleaned_answer = re.sub(r'\s*\(\s*,\s*\s*\)', '', cleaned_answer) 
        cleaned_answer = cleaned_answer.strip()

        # 5. 최종 출력 제어 (인라인 태그 추출 및 제거)
        if "정보 없음" in cleaned_answer:
            return "정보 없음"

        # 답변에서 사용된 인라인 태그만 추출
        used_tags_str = set(re.findall(r'\[(\d+)\]', cleaned_answer)) 
        
        # 태그를 제거하여 최종 답변 본문 생성
        final_body = re.sub(r'\[\d+\]', '', cleaned_answer).strip()

        final_sources_list = []
        
        # 추출된 태그 번호에 해당하는 파일명만 sources_map에서 찾아서 추가
        for tag_str in used_tags_str:
            try:
                doc_index = int(tag_str) - 1 
                if 0 <= doc_index < len(sources_map):
                    final_sources_list.append(sources_map[doc_index])
            except ValueError:
                pass 

        # 최종 출력할 출처 목록 (중복 제거 후 정렬)
        final_sources_set = set(final_sources_list)
        final_sources_sorted = sorted(list(final_sources_set))
        final_sources = [f"[출처: {s}]" for s in final_sources_sorted]

        if final_sources:
            # final_body를 사용하며, 출처 목록을 첨부
            final_output = f"{final_body}\n\n{'-'*60}\n" + "\n".join(final_sources)
        else:
            # 출처 정보가 없는 경우 (LLM이 태그를 안 붙인 경우)
            final_output = f"{final_body}\n\n{'-'*60}\n"

        return final_output

rag_service = RAGService()