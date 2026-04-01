import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document


class FAQRepository:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._documents = self._load_documents()
        self._token_index = [Counter(self._tokenize(doc.page_content)) for doc in self._documents]

    def _load_documents(self) -> List[Document]:
        documents: List[Document] = []
        with self.csv_path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                question = row["question"].strip()
                answer = row["answer"].strip()
                documents.append(Document(page_content=f"Q: {question}\nA: {answer}"))
        return documents

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        scored_documents: List[tuple[int, Document]] = []

        for document, tokens in zip(self._documents, self._token_index):
            overlap_score = sum(min(tokens[token], count) for token, count in query_counts.items())
            if overlap_score == 0:
                continue

            phrase_bonus = 3 if query.lower() in document.page_content.lower() else 0
            scored_documents.append((overlap_score + phrase_bonus, document))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored_documents[:top_k]]


class FAQSearchTool:
    def __init__(self, repository: FAQRepository) -> None:
        self.repository = repository

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        matches = self.repository.search(query=query, top_k=top_k)
        return [
            {
                "rank": index + 1,
                "content": document.page_content,
            }
            for index, document in enumerate(matches)
        ]
