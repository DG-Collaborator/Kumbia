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
     retrieval index so the persona can draw on either when answering. Each
     entry is indexed with a provenance label ("[source · as of <time>]") so
     the model can tell a fact the user taught it from an older snapshot of a
     data lookup, and say how fresh the snapshot is.

     Hallucination guards, all deterministic: routing picks a skill name from
     a fixed list (or "none" / "unavailable", never a query); narrations and
     free-form replies are checked by rag.grounding.unsupported_figures
     against the evidence they were given, and any reply stating a figure the
     evidence doesn't contain is withheld rather than returned.
"""
from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from kumbia.core.domain.data_table import DataTable
from kumbia.core.source.data_source import DataSource
from kumbia.core.source.source_query import SourceQuery
from kumbia.rag.clients.llm_client import LlmClient
from kumbia.rag.grounding import unsupported_figures, withhold_unsupported
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
    "below. Reply with exactly one skill name from the list when the message asks "
    "about that skill's data. Reply with none if the message is small talk, about "
    "this conversation, or about something the user asked you to remember. Reply "
    "with unavailable if the message asks for data (metrics, readings, hosts, "
    "events, logs, figures) that none of the listed skills provides. Reply with "
    "the single word only, no other words."
)

_ROUTING_HISTORY_TURNS = 4
_UNAVAILABLE = "unavailable"

_GROUNDING_RULES = (
    "Rules: the remembered facts are labeled with where they came from and when. "
    "A fact from a data lookup is a snapshot as of its time, not necessarily the "
    "current value; say its time when you use it. Never state a number, address, "
    "host, or measurement that is not in the remembered facts, the conversation, "
    "or the user's message. If you don't have it, say so plainly."
)

_REMEMBER_REQUEST = re.compile(r"^\s*(?:please\s+)?remember(?:\s+that|\s*:)\s+(?P<fact>.+)$", re.IGNORECASE | re.DOTALL)


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
                self._memory_index.add(
                    _labeled(row.value("text"), row.value("source"), row.value("created_at")), vector,
                )
        self._memory_loaded = True

    async def remember(self, fact: str) -> None:
        """
           Persists a new fact to the persona's memory_path and makes it
           immediately searchable in this session, without needing a restart.
        """
        await self.ensure_memory_loaded()
        append_memory(self._persona.memory_path, fact, source="user")
        vector = await self._llm.embed(fact)
        self._memory_index.add(_labeled(fact, "user", _now_iso()), vector)

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
        narration = await self._llm.generate(question, system_prompt=system_prompt, temperature=0)
        narration = withhold_unsupported(narration, figures, question)

        answer = f"{figures}\n\n{narration}"
        await self.record_lookup(name, answer)
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

        await self.record_lookup(name, answer)
        return answer

    async def record_lookup(self, source: str, answer: str) -> None:
        """
           Caches a data-backed answer to the persona's cache_path (tagged
           with the source it came from) and makes it searchable now, so
           later chat can refer back to it, while keeping auto-cached
           lookups distinct from explicitly-taught facts (remember()).
           Public so the composition root can hand it to any other producer
           of data-backed answers (another persona's tools, for example) as
           a plain callback, without that producer knowing this pipeline.
        """
        await self.ensure_memory_loaded()
        append_memory(self._persona.cache_path, answer, source=source)
        vector = await self._llm.embed(answer)
        self._memory_index.add(_labeled(answer, source, _now_iso()), vector)

    async def ask(self, text: str) -> str:
        """
           Continues the conversation. "Remember that ..." is stored directly
           as a fact (no LLM involved). Otherwise, when skills are registered,
           the LLM first picks one by name: a skill answers from real data;
           "unavailable" (a data question no skill covers) gets a plain
           statement of what data exists instead of a guess; anything else is
           answered from history and labeled remembered facts, with the reply
           withheld if it states a figure none of that evidence contains.
        """
        await self.ensure_memory_loaded()

        remember_request = _REMEMBER_REQUEST.match(text)
        if remember_request:
            fact = remember_request.group("fact").strip()
            await self.remember(fact)
            return self._record_turn(text, f"Got it, I'll remember: {fact}")

        skill, unavailable = await self._route(text)
        if skill is not None:
            return self._record_turn(text, f"(answered from {skill.name})\n\n{await skill.run(text)}")
        if unavailable:
            return self._record_turn(text, self._unavailable_reply())

        memory_context: list[str] = []
        if self._memory_index.size > 0:
            query_vector = await self._llm.embed(text)
            memory_context = self._memory_index.search(query_vector, self._memory_top_k)

        history = [*self._history, Turn("user", text)]
        system_prompt = f"{self._persona.system_prompt}\n\n{_GROUNDING_RULES}"
        reply = await self._llm.generate(_build_prompt(history, memory_context), system_prompt=system_prompt)
        if self._persona.strict_grounding:
            evidence = [text, *memory_context, *(turn.text for turn in self._history)]
            if unsupported_figures(reply, *evidence):
                reply = (
                    "I don't have data that supports a specific answer to that. "
                    + self._available_data_hint()
                )
        return self._record_turn(text, reply)

    def _record_turn(self, text: str, reply: str) -> str:
        self._history.append(Turn("user", text))
        self._history.append(Turn("assistant", reply))
        return reply

    def _unavailable_reply(self) -> str:
        return "I don't have a data source that answers that. " + self._available_data_hint()

    def _available_data_hint(self) -> str:
        if not self._skills:
            return "No data sources are connected right now."
        listed = "\n".join(f"- {skill.name}: {skill.description}" for skill in self._skills.values())
        return f"I can answer from:\n{listed}"

    async def _route(self, text: str) -> tuple[Skill | None, bool]:
        """
           One short, temperature-0 LLM call choosing a skill by name.
           Returns (skill, False) for exactly one recognizable skill name,
           (None, True) for "unavailable", and (None, False) for anything
           else (none, an unknown name, two names), which falls back to
           plain conversation rather than guessing a skill.
        """
        if not self._skills:
            return None, False
        catalog = "\n".join(f"- {skill.name}: {skill.description}" for skill in self._skills.values())
        recent = "\n".join(
            f"{'User' if turn.role == 'user' else 'Assistant'}: {_clip(turn.text)}"
            for turn in self._history[-_ROUTING_HISTORY_TURNS:]
        )
        prompt = (
            f"Skills:\n{catalog}\n\n"
            + (f"Recent conversation:\n{recent}\n\n" if recent else "")
            + f"Message: {text}\nSkill:"
        )
        choice = await self._llm.generate(prompt, system_prompt=_ROUTING_INSTRUCTIONS, temperature=0)
        named = set(re.findall(r"[a-z0-9_]+", choice.lower()))
        matches = [skill for name, skill in self._skills.items() if name in named]
        if len(matches) == 1:
            return matches[0], False
        return None, (not matches and _UNAVAILABLE in named)

    def reset(self) -> None:
        self._history.clear()


def _labeled(text: str, source: str, created_at: str) -> str:
    origin = "taught by the user" if source in ("", "user") else f"lookup from {source}"
    when = created_at[:16].replace("T", " ") + " UTC" if created_at else "unknown time"
    return f"[{origin} · as of {when}] {text}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _build_prompt(history: list[Turn], memory_context: list[str]) -> str:
    lines: list[str] = []
    if memory_context:
        lines.append("Remembered facts that might be relevant:")
        lines.extend(f"- {fact}" for fact in memory_context)
        lines.append("")
    lines.extend(f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.text}" for turn in history)
    lines.append("Assistant:")
    return "\n".join(lines)
