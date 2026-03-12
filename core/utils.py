import json
from pathlib import Path
from typing import Final, Iterable, List, Union
from langchain.schema import Document


class DocumentProcess:

    @staticmethod
    def save_docs_jsonl(
        docs: Iterable[Document],
        path: Final[Union[str, Path]]
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for doc in docs:
                row = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                f.write(json.dumps(row, default=str) + "\n")

    @staticmethod
    def load_docs_jsonl(
        path: Final[Union[str, Path]]
    ) -> List[Document]:
        """
        Load LangChain Documents from a JSONL file.
        """
        path = Path(path)
        docs: List[Document] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(
                    Document(
                        page_content=row["page_content"],
                        metadata=row.get("metadata", {}),
                    )
                )

        return docs