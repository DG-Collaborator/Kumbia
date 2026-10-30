"""
     Generalized ConversationPipeline: one class, driven entirely by an injected
     PersonaProfile — no assistant-specific paths, prompt, or data source baked
     in. Holds turn history in memory, retrieves relevant remembered facts (and
     previously-cached lookups) from the persona's own on-disk memory, asks the
     LLM to continue the conversation grounded in that, and can query any named
     DataSource the persona was given.

     query_data_source() generalizes DAFNI's original web_search(): question is
     passed straight through as SourceQuery.statement, grounding the answer via
     a fresh, throwaway RagPipeline over whatever the source returns. That fits
     any source whose fetch() is itself a search (Tavily today, any future
     free-text lookup source).

     query_structured_source() is the counterpart for a source that needs a
     real query.address or a structured query body (SqlDataSource,
     ElasticsearchDataSource, PrometheusDataSource) — the caller (an MCP tool,
     never the LLM) supplies the already-built SourceQuery; question is used
     only to ground the answer, never to construct the query itself. Keeping
     query construction out of the LLM's hands is deliberate: it's the same
     "structural, not instructional" boundary the plan applies to persona tool
     visibility, applied here to what a persona can ask a structured store.

     narrate_structured_source() goes one step further for questions about
     trends and relationships: the caller also supplies a deterministic
     summarize function that computes the figures from the fetched table, and
     the LLM only narrates those figures — it never retrieves rows or does
     arithmetic itself. Same split as NetworkActivityPipeline (deterministic
     summary, LLM narration), generalized to any source and any summarizer.

     Explicitly-remembered facts (memory_path) and auto-cached query_data_source
     results (cache_path) are kept in separate on-disk files so a persona's
     curated facts stay distinct from automated lookups, but both feed the same
     retrieval index so the persona can draw on either when answering.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from kumbia.core.domain.data_table import DataTable
from kumbia.core.source.data_source import DataSource
from kumbia.core.source.source_query import SourceQuery
from kumbia.rag.clients.llm_client import LlmClient
from kumbia.rag.in_memory_vector_index import InMemoryVectorIndex
from kumbia.rag.memory_store import append_memory, load_memory_table
from kumbia.rag.persona_profile import PersonaProfile
from kumbia.rag.rag_pipeline import RagPipeline

_NARRATION_INSTRUCTIONS = (
    "Answer the user's question using only the figures below. They were computed "
    "deterministically from the underlying records: do not recalculate, re-derive, "
    "or contradict them, and do not introduce any value that is not listed. State "
    "peaks, changes, and relationships exactly as given."
)

_ROUTING_INSTRUCTIONS = (
    "You decide whether a user's message needs one of the data skills listed "
    "below. Reply with exactly one skill name from the list, or with none if the "
    "message is small talk, general knowledge, or not about any listed skill's "
    "data. Reply with the name only, no other words."
)

_ROUTING_HISTORY_TURNS = 4


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    text: str


@dataclass(frozen=True)
class Skill:
    """
       A data-backed capability ask() may route a chat turn to. run takes the
       user's question and returns a finished answer; what it fetches, which
       already-built SourceQuery it uses, and how it computes figures are all
       fixed by whoever registers it. The LLM only ever picks a skill by name,
       never builds a query — the same structural boundary as
       query_structured_source, applied to free-form chat.
    """
    name: str
    description: str
    run: Callable[[str], Awaitable[str]]


class ConversationPipeline:
    def __init__(
            self,
            llm: LlmClient,
            persona: PersonaProfile,
            memory_top_k: int = 3,
    ) -> None:
        self._llm = llm
        self._persona = persona
        self._history: list[Turn] = []
        self._memory_index = InMemoryVectorIndex()
        self._memory_top_k = memory_top_k
        self._memory_loaded = False
        self._skills: dict[str, Skill] = {}

    def add_skill(self, skill: Skill) -> None:
        """Makes skill reachable from ask(). Registered by the composition layer, never by the LLM."""
        self._skills[skill.name] = skill

    @property
    def turn_count(self) -> int:
        return len(self._history)

    @property
    def memory_count(self) -> int:
        return self._memory_index.size

    async def ensure_memory_loaded(self) -> None:
        """
           Loads both on-disk files (explicit memory + auto-cache) into the
           live index on first use, within whichever event loop is actually
           running tool calls — deliberately not done eagerly at startup in a
           separate asyncio.run(), since that would bind the shared http
           client's connections to a loop that gets closed before the
           server's own loop starts serving requests. Public since it's also
           callable from a status resource, not just from ask()/remember()/
           query_data_source() internally.
        """
        if self._memory_loaded:
            return
        for path in (self._persona.memory_path, self._persona.cache_path):
            table = load_memory_table(path)
            for row in table.rows:
                vector = await self._llm.embed(row.value("text"))
                self._memory_index.add(row.value("text"), vector)
        self._memory_loaded = True

    async def remember(self, fact: str) -> None:
        """
           Persists a new fact to the persona's memory_path and makes it
           immediately searchable in this session, without needing a restart.
        """
        await self.ensure_memory_loaded()
        append_memory(self._persona.memory_path, fact, source="user")
        vector = await self._llm.embed(fact)
        self._memory_index.add(fact, vector)

    async def query_data_source(self, name: str, question: str) -> str:
        """
           Generic dataset lookup for a search-shaped source: question is
           passed straight through as the fetch's own query text. Fetches
           fresh results from the named DataSource and synthesizes a
           grounded answer via a fresh, throwaway RagPipeline.
        """
        await self.ensure_memory_loaded()
        source = self._require_data_source(name)
        table = await source.fetch(SourceQuery(address=name, statement=question))
        return await self._ground_and_cache(name, table, question)

    async def query_structured_source(self, name: str, query: SourceQuery, question: str) -> str:
        """
           Dataset lookup for a structured source: the caller supplies the
           already-built SourceQuery (a real address, a real SQL statement or
           query body) — question only grounds the answer afterward, never
           drives the fetch. Use this for anything query_data_source's free-
           text-as-statement shape isn't safe or correct for.
        """
        await self.ensure_memory_loaded()
        source = self._require_data_source(name)
        table = await source.fetch(query)
        return await self._ground_and_cache(name, table, question)

    async def narrate_structured_source(
            self,
            name: str,
            query: SourceQuery,
            question: str,
            summarize: Callable[[DataTable], str],
    ) -> str:
        """
           Like query_structured_source, but for questions whose answer is a
           computation (peaks, changes, relationships) rather than a lookup:
           summarize turns the fetched table into figures deterministically,
           and the LLM is only asked to narrate those figures. Returns the
           figures followed by the narration, so the exact numbers are always
           visible next to the model's wording.
        """
        await self.ensure_memory_loaded()
        source = self._require_data_source(name)
        table = await source.fetch(query)
        if table.is_empty:
            return "I couldn't find anything for that."

        figures = summarize(table)
        system_prompt = f"{self._persona.system_prompt}\n\n{_NARRATION_INSTRUCTIONS}\n\nFigures:\n{figures}"
        narration = await self._llm.generate(question, system_prompt=system_prompt)

        answer = f"{figures}\n\n{narration}"
        await self._cache(name, answer)
        return answer

    def _require_data_source(self, name: str) -> DataSource:
        source = self._persona.data_sources.get(name)
        if source is None:
            raise RuntimeError(
                f"'{name}' is not a data source available to persona '{self._persona.name}'."
            )
        return source

    async def _ground_and_cache(self, name: str, table: DataTable, question: str) -> str:
        """
           Shared by query_data_source and query_structured_source: RAG-grounds
           an answer over a fetched DataTable, then caches it.
        """
        if table.is_empty:
            return "I couldn't find anything for that."

        lookup_pipeline = RagPipeline(self._llm)
        await lookup_pipeline.ingest(table)
        answer = await lookup_pipeline.answer(question)

        await self._cache(name, answer)
        return answer

    async def _cache(self, name: str, answer: str) -> None:
        """
           Caches a synthesized answer to the persona's cache_path (tagged
           with the source name) and makes it searchable now, so a repeated
           question doesn't need a re-fetch, while keeping auto-cached
           lookups distinct from explicitly-taught facts (remember()).
        """
        append_memory(self._persona.cache_path, answer, source=name)
        vector = await self._llm.embed(answer)
        self._memory_index.add(answer, vector)

    async def ask(self, text: str) -> str:
        """
           Continues the conversation. When skills are registered, first asks
           the LLM whether text needs one; if so the skill answers (from real
           data) and that answer becomes the assistant's turn, otherwise the
           reply is generated from history and remembered facts as before.
        """
        await self.ensure_memory_loaded()
        skill = await self._route(text)
        self._history.append(Turn("user", text))

        if skill is not None:
            reply = f"(answered from {skill.name})\n\n{await skill.run(text)}"
        else:
            memory_context: list[str] = []
            if self._memory_index.size > 0:
                query_vector = await self._llm.embed(text)
                memory_context = self._memory_index.search(query_vector, self._memory_top_k)

            prompt = _build_prompt(self._history, memory_context)
            reply = await self._llm.generate(prompt, system_prompt=self._persona.system_prompt)
        self._history.append(Turn("assistant", reply))
        return reply

    async def _route(self, text: str) -> Skill | None:
        """
           One short LLM call choosing a skill by name. Anything other than
           exactly one recognizable skill name (none, an unknown name, two
           names) falls back to plain conversation rather than guessing.
        """
        if not self._skills:
            return None
        catalog = "\n".join(f"- {skill.name}: {skill.description}" for skill in self._skills.values())
        recent = "\n".join(
            f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.text}"
            for turn in self._history[-_ROUTING_HISTORY_TURNS:]
        )
        prompt = (
            f"Skills:\n{catalog}\n\n"
            + (f"Recent conversation:\n{recent}\n\n" if recent else "")
            + f"Message: {text}\nSkill:"
        )
        choice = await self._llm.generate(prompt, system_prompt=_ROUTING_INSTRUCTIONS)
        named = set(re.findall(r"[a-z0-9_]+", choice.lower()))
        matches = [skill for name, skill in self._skills.items() if name in named]
        return matches[0] if len(matches) == 1 else None

    def reset(self) -> None:
        self._history.clear()


def _build_prompt(history: list[Turn], memory_context: list[str]) -> str:
    lines: list[str] = []
    if memory_context:
        lines.append("Remembered facts that might be relevant:")
        lines.extend(f"- {fact}" for fact in memory_context)
        lines.append("")
    lines.extend(f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.text}" for turn in history)
    lines.append("Assistant:")
    return "\n".join(lines)