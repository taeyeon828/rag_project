from pathlib import Path
import re, json
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_PDF_DIR = Path("data/raw/pdf")
OUT_TEXT_DIR = Path("data/processed/text")
OUT_CHUNKS_DIR = Path("data/processed/chunks")
OUT_REPORT_DIR = Path("data/processed/report")

for p in [OUT_TEXT_DIR, OUT_CHUNKS_DIR, OUT_REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def clean_text(t: str) -> str:
    # "스마트-\n공장" -> "스마트공장"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # 문장 중간 줄바꿈을 공백으로
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # 공백 정리
    t = re.sub(r"[ \t]+", " ", t)
    # 빈 줄 정리
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=90,
    separators=["\n\n", "\n", ". ", "다. ", " "]
)

chunks_path = OUT_CHUNKS_DIR / "chunks.jsonl"
report = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "files": []
}

with chunks_path.open("w", encoding="utf-8") as fout:
    for pdf_path in RAW_PDF_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # page_content + metadata(page)
        file_stat = {
            "file": pdf_path.name,
            "pages": len(pages),
            "empty_pages": 0,
            "chunks": 0
        }

        # 1) 페이지별 텍스트 저장(정제본)
        file_text_dir = OUT_TEXT_DIR / pdf_path.stem
        file_text_dir.mkdir(parents=True, exist_ok=True)

        cleaned_pages = []
        for i, doc in enumerate(pages, start=1):
            raw = doc.page_content or ""
            cleaned = clean_text(raw)
            
            if len(cleaned) < 10:
                file_stat["empty_pages"] += 1

            (file_text_dir / f"page_{i:03d}.txt").write_text(cleaned, encoding="utf-8")
            cleaned_pages.append((i, cleaned))

        # 2) 청킹 + jsonl 저장
        # 페이지 메타 유지하려고 page별로 쪼개서 chunk
        for page_no, text in cleaned_pages:
            if not text:
                continue
            docs = splitter.create_documents(
                [text],
                metadatas=[{
                    "source": pdf_path.name,
                    "page": page_no,
                    "doc_type": "pdf"
                }]
            )
            for j, d in enumerate(docs, start=1):
                rec = {
                    "chunk_id": f"{pdf_path.stem}_p{page_no:03d}_c{j:03d}",
                    "text": d.page_content,
                    **d.metadata,
                    "char_len": len(d.page_content)
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                file_stat["chunks"] += 1

        report["files"].append(file_stat)

# 3) 품질 리포트 저장
(OUT_REPORT_DIR / "quality_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("완료!")
print(f"- chunks: {chunks_path}")
print(f"- report: {OUT_REPORT_DIR / 'quality_report.json'}")
