"""
   Ties the pieces together: ingest rows from any DataSource's DataTable into
   the vector index, then answer questions grounded in those rows. Embedding,
   retrieval, and generation all delegate to LlmClient, so the model is swapped
   by changing the LlmClient passed in, not this class.
"""
from __future__ import annotations

from kumbia.core.domain.data_table import DataTable
from kumbia.rag.clients.llm_client import LlmClient
from kumbia.rag.embedding import embed_all
from kumbia.rag.grounding import unsupported_figures
from kumbia.rag.in_memory_vector_index import InMemoryVectorIndex


_MAX_ROW_CHARS = 4000  # keeps one very wide row under embedding input limits


class RagPipeline:
    def __init__(self, llm: LlmClient, index: InMemoryVectorIndex | None = None) -> None:
        self._llm = llm
        self._index = index if index is not None else InMemoryVectorIndex()

    @property
    def indexed_count(self) -> int:
        return self._index.size

    async def ingest(self, table: DataTable) -> None:
        """
           Embeds every row of a DataTable and loads it into the index. Each
           row is rendered as "key: value | key: value ..." so the embedding
           captures both the column names and the values, which makes later
           retrieval match on meaning rather than raw position. Rows are
           embedded concurrently (rag.embedding.embed_all), and empty cells
           are left out so wide, sparse rows stay short.
        """
        texts = [
            " | ".join(
                f"{column.key}: {row.value(column.key)}" for column in table.columns if row.value(column.key)
            )[:_MAX_ROW_CHARS]
            for row in table.rows
        ]
        for text, vector in zip(texts, await embed_all(self._llm, texts)):
            self._index.add(text, vector)

    async def answer(self, question: str, top_k: int = 4) -> str:
        """
           The RAG query. Embeds the question, pulls the most similar rows
           from the index, and asks the llm to answer using only those rows as
           context, so the answer is grounded in the ingested source rather than
           the model's training data. If the answer states a figure (number,
           IP) that isn't in the retrieved rows or the question, it is not
           returned; the matching rows are shown instead, so nothing invented
           ever reaches the caller.
        """
        query_vector = await self._llm.embed(question)
        context = self._index.search(query_vector, top_k)
        answer = await self._llm.generate(_build_prompt(question, context), temperature=0)

        unsupported = unsupported_figures(answer, question, *context)
        if not unsupported:
            return answer
        rows = "\n".join(f"- {row}" for row in context) or "(no matching rows)"
        return (
            "I couldn't give a precise answer from the data, so here are the closest matching records "
            f"(the drafted answer cited {', '.join(unsupported)}, which the records don't contain):\n{rows}"
        )


def _build_prompt(question: str, context: list[str]) -> str:
    """
       Builds the grounded prompt: the retrieved rows as context, then the
       question, with an instruction to stay within the data.
    """
    data_block = "\n".join(context) if context else "(no data)"
    return (
        "Answer the question using only the data rows below.\n"
        "If the answer is not in the data, say you don't know.\n\n"
        "Data:\n"
        f"{data_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
