"""Prompt template for the extractor LLM.

We split the prompt into:
  - SYSTEM_PROMPT: fixed, identical across all chunks of all docs. This is
                   the part we cache (Anthropic prompt caching) so we pay
                   for it once per process.
  - chunk_user_prompt(): per-chunk content. Small.

Design notes:
  - We ask for JSON only (no prose) to make parsing deterministic.
  - We tell the model to emit ids local to THIS chunk (s1, s2, p1, c1).
    The merger handles cross-chunk id collisions.
  - We tell the model not to invent steps it can't see in the chunk —
    this is the main hallucination control. Cross-chunk inference is the
    merger's job, not the extractor's.
"""

SYSTEM_PROMPT = """\
You are extracting an executable workflow graph from a chunk of an industrial
Standard Operating Procedure (SOP). Output ONLY a JSON object that conforms
to the schema below. Do not include explanations, markdown fences, or
commentary — just the JSON.

Schema:
{
  "steps":         [{"id": "s1", "ordinal": 1, "text": "...", "page": 3 | null}, ...],
  "preconditions": [{"id": "p1", "text": "..."}, ...],
  "constraints":   [{"id": "c1", "text": "...", "kind": "ppe|lockout|hazard|other"}, ...],
  "edges":         [{"type": "precedes|precondition_of|constrains",
                     "from": "<id>", "to": "<id>"}, ...]
}

Definitions:
- step:          an imperative action the operator must perform.
                 e.g. "Open valve A.", "Record the temperature."
- precondition:  a clause that gates a step. e.g. "Before opening valve A,
                 ensure tank B is depressurised."  -> precondition_of edge
                 to the step it gates.
- constraint:    a safety clause attached to one or more steps. Kinds:
                 "ppe"     = personal protective equipment requirement
                 "lockout" = lockout/tagout / isolation requirement
                 "hazard"  = warning, caution, danger statement
                 "other"   = any other safety-relevant clause
- precedes:      ordering between two steps that appear in this chunk.
                 If step s2 must follow s1, emit {"type":"precedes","from":"s1","to":"s2"}.

Rules:
1. Emit ids local to THIS chunk: s1, s2, …, p1, p2, …, c1, c2, …
2. Order steps by appearance in the chunk; populate `ordinal` 1..N.
3. Do NOT invent steps, preconditions, or constraints that are not present
   in the chunk text. Empty arrays are fine.
4. Conditional skips ("If valve A is closed, skip steps 12-15") are
   preconditions attached to each affected step that we can see in this
   chunk. If the affected steps are not in this chunk, emit only the
   precondition with no edge.
5. Cross-references in step text ("see step 9") are NOT edges; they are
   informational.
6. Always include all four top-level keys, even if empty.
"""


def chunk_user_prompt(chunk_text: str, *, chunk_idx: int, doc_title: str) -> str:
    return (
        f"Document title: {doc_title}\n"
        f"Chunk index: {chunk_idx}\n"
        f"---------- CHUNK BEGINS ----------\n"
        f"{chunk_text}\n"
        f"---------- CHUNK ENDS ----------\n"
        f"Return the JSON workflow graph for this chunk."
    )
