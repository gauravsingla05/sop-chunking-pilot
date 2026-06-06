"""Workflow-graph JSON schema produced by the extractor LLM.

This is THE contract between the extractor and the rest of the pipeline.
Changing this shape requires re-running every experiment.

Top-level object:
{
  "steps":         [Step, ...],         # ordered list
  "preconditions": [Precondition, ...], # uniquely-id'd preconditions
  "constraints":   [Constraint, ...],   # uniquely-id'd safety constraints
  "edges":         [Edge, ...],         # typed edges between the above
}

Step:
  { "id": "s1", "ordinal": 1, "text": "Open valve A.", "page": 3 }
Precondition:
  { "id": "p1", "text": "Before doing X, ensure Y." }
Constraint (safety):
  { "id": "c1", "text": "Wear PPE.", "kind": "ppe|lockout|hazard|other" }
Edge:
  { "type": "precedes|precondition_of|constrains",
    "from": "<id>", "to": "<id>" }

Ids are local to a single extractor call. The merger renames them when
unioning chunks into the per-document graph.
"""

from __future__ import annotations

WORKFLOW_GRAPH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps", "preconditions", "constraints", "edges"],
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "ordinal", "text"],
                "properties": {
                    "id":      {"type": "string"},
                    "ordinal": {"type": "integer"},
                    "text":    {"type": "string"},
                    "page":    {"type": ["integer", "null"]},
                },
            },
        },
        "preconditions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "text"],
                "properties": {
                    "id":   {"type": "string"},
                    "text": {"type": "string"},
                },
            },
        },
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "text", "kind"],
                "properties": {
                    "id":   {"type": "string"},
                    "text": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["ppe", "lockout", "hazard", "other"]},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "from", "to"],
                "properties": {
                    "type": {"type": "string",
                             "enum": ["precedes", "precondition_of", "constrains"]},
                    "from": {"type": "string"},
                    "to":   {"type": "string"},
                },
            },
        },
    },
}


EMPTY_GRAPH = {
    "steps": [],
    "preconditions": [],
    "constraints": [],
    "edges": [],
}
