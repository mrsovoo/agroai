"""Knowledge graph: what the knowledge base *declares* about itself.

The roadmap wants relationships between crops, diseases, regions and sources to
be usable, e.g. ``Tomato -> disease -> late blight``, ``Tomato -> region ->
Uzbekistan``.

This graph is built **only from declared metadata** (category, crop, animal,
region, country, source, organization) - never by guessing relationships from
sentence content. That keeps an explicit promise: every edge can be traced back
to the document that declared it, and a co-occurrence edge is labelled as
exactly that.

It is a small, auditable graph today; a full ontology-based knowledge graph is a
later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from agroai.knowledge.documents import KnowledgeDocument

NODE_DOCUMENT = "document"
NODE_CATEGORY = "category"
NODE_CROP = "crop"
NODE_ANIMAL = "animal"
NODE_REGION = "region"
NODE_COUNTRY = "country"
NODE_SOURCE = "source"

RELATION_IN_CATEGORY = "in_category"
RELATION_ABOUT_CROP = "about_crop"
RELATION_ABOUT_ANIMAL = "about_animal"
RELATION_ABOUT_REGION = "about_region"
RELATION_ABOUT_COUNTRY = "about_country"
RELATION_SOURCED_FROM = "sourced_from"
#: Derived: the two entities were declared by the same document.
RELATION_CO_OCCURS = "co_occurs_in_document"

NODE_KINDS = (
    NODE_DOCUMENT,
    NODE_CATEGORY,
    NODE_CROP,
    NODE_ANIMAL,
    NODE_REGION,
    NODE_COUNTRY,
    NODE_SOURCE,
)


@dataclass(frozen=True)
class GraphNode:
    """One entity in the graph, with the documents that declared it."""

    id: str
    kind: str
    label: str
    documents: tuple[str, ...] = ()

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "documents": list(self.documents),
        }


@dataclass(frozen=True)
class GraphEdge:
    """A directed, traceable relationship between two entities."""

    relation: str
    source_id: str
    target_id: str
    document_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "source": self.source_id,
            "target": self.target_id,
            "document_id": self.document_id,
        }


@dataclass
class KnowledgeGraph:
    """Nodes and edges derived from declared document metadata."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    _edge_keys: set[tuple[str, str, str, str]] = field(default_factory=set, repr=False)

    # ------------------------------------------------------------------ #
    def add_node(self, kind: str, label: str, document_id: str = "") -> GraphNode:
        """Add (or extend) a node. Labels are matched case-insensitively."""
        clean = label.strip()
        if not clean:
            raise ValueError("a graph node needs a label")
        node_id = f"{kind}:{clean.lower()}"
        existing = self.nodes.get(node_id)
        if existing is None:
            node = GraphNode(
                id=node_id,
                kind=kind,
                label=clean,
                documents=(document_id,) if document_id else (),
            )
            self.nodes[node_id] = node
            return node
        if document_id and document_id not in existing.documents:
            updated = GraphNode(
                id=existing.id,
                kind=existing.kind,
                label=existing.label,
                documents=existing.documents + (document_id,),
            )
            self.nodes[node_id] = updated
            return updated
        return existing

    def add_edge(
        self, relation: str, source_id: str, target_id: str, document_id: str = ""
    ) -> GraphEdge | None:
        key = (relation, source_id, target_id, document_id)
        if key in self._edge_keys:
            return None
        self._edge_keys.add(key)
        edge = GraphEdge(
            relation=relation, source_id=source_id, target_id=target_id, document_id=document_id
        )
        self.edges.append(edge)
        return edge

    # ------------------------------------------------------------------ #
    def get(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def find(self, label: str, kind: str | None = None) -> GraphNode | None:
        """Find a node by label (case-insensitive), optionally of a given kind."""
        wanted = label.strip().lower()
        for node in self.nodes.values():
            if node.label.lower() == wanted and (kind is None or node.kind == kind):
                return node
        return None

    def nodes_of_kind(self, kind: str) -> list[GraphNode]:
        return sorted(
            (node for node in self.nodes.values() if node.kind == kind),
            key=lambda node: (-node.document_count, node.label.lower()),
        )

    def neighbors(self, node_id: str, relation: str | None = None) -> list[GraphNode]:
        """Entities reachable from ``node_id`` (optionally one relation only)."""
        found: list[GraphNode] = []
        seen: set[str] = set()
        for edge in self.edges:
            if edge.source_id != node_id:
                continue
            if relation and edge.relation != relation:
                continue
            if edge.target_id in seen:
                continue
            neighbours = self.nodes.get(edge.target_id)
            if neighbours is None:
                continue
            seen.add(edge.target_id)
            found.append(neighbours)
        return sorted(found, key=lambda node: node.label.lower())

    def relations(self) -> list[str]:
        return sorted({edge.relation for edge in self.edges})

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for node in self.nodes.values():
            by_kind[node.kind] = by_kind.get(node.kind, 0) + 1
        by_relation: dict[str, int] = {}
        for edge in self.edges:
            by_relation[edge.relation] = by_relation.get(edge.relation, 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "nodes_by_kind": by_kind,
            "edges_by_relation": by_relation,
        }

    def render(self, node_id: str | None = None, limit: int = 20) -> str:
        """Render the whole graph, or one node's neighbourhood."""
        lines: list[str] = []
        if node_id:
            node = self.get(node_id) or self.find(node_id)
            if node is None:
                return f"node not found: {node_id}"
            lines.append(f"{node.label} [{node.kind}] ({node.document_count} document(s))")
            for edge in sorted(self.edges, key=lambda item: item.relation):
                if edge.source_id != node.id:
                    continue
                target = self.nodes.get(edge.target_id)
                if target is None:
                    continue
                lines.append(f"  -{edge.relation}-> {target.label} [{target.kind}]")
            return "\n".join(lines[:limit])

        stats = self.stats()
        lines.append(f"Knowledge graph: {stats['nodes']} nodes, {stats['edges']} edges")
        for relation, count in sorted(stats["edges_by_relation"].items()):
            lines.append(f"  {relation}: {count}")
        for kind in NODE_KINDS:
            nodes = self.nodes_of_kind(kind)
            if not nodes:
                continue
            labels = ", ".join(node.label for node in nodes[:8])
            more = "" if len(nodes) <= 8 else f" (+{len(nodes) - 8} more)"
            lines.append(f"  {kind}: {labels}{more}")
        return "\n".join(lines[:limit])


def build_graph(
    documents: Iterable[KnowledgeDocument], *, include_documents: bool = True
) -> KnowledgeGraph:
    """Build a graph from declared metadata (no content inference)."""
    graph = KnowledgeGraph()
    for document in documents:
        if include_documents:
            document_node = graph.add_node(NODE_DOCUMENT, document.title or document.id, document.id)
        else:
            document_node = None

        declared: list[tuple[str, str, str]] = []
        if document.category:
            declared.append((NODE_CATEGORY, document.category, RELATION_IN_CATEGORY))
        crop = str(document.metadata.get("crop", "")).strip()
        animal = str(document.metadata.get("animal", "")).strip()
        region = document.region().strip()
        country = document.country().strip()
        source = document.organization().strip()
        if crop:
            declared.append((NODE_CROP, crop, RELATION_ABOUT_CROP))
        if animal:
            declared.append((NODE_ANIMAL, animal, RELATION_ABOUT_ANIMAL))
        if region:
            declared.append((NODE_REGION, region, RELATION_ABOUT_REGION))
        if country:
            declared.append((NODE_COUNTRY, country, RELATION_ABOUT_COUNTRY))
        if source:
            declared.append((NODE_SOURCE, source, RELATION_SOURCED_FROM))

        context_nodes: list[GraphNode] = []
        for kind, label, relation in declared:
            node = graph.add_node(kind, label, document.id)
            context_nodes.append(node)
            if document_node is not None:
                graph.add_edge(relation, document_node.id, node.id, document.id)

        # Derived, explicitly named co-occurrence edges between context entities.
        for index, first in enumerate(context_nodes):
            for second in context_nodes[index + 1 :]:
                if first.kind == second.kind:
                    continue
                graph.add_edge(RELATION_CO_OCCURS, first.id, second.id, document.id)
    return graph


def graph_from_database(database: Any) -> KnowledgeGraph:
    """Build the graph from every document stored in the database."""
    return build_graph(database.iter_documents())


def related_context(graph: KnowledgeGraph, crop: str = "", region: str = "") -> dict[str, list[str]]:
    """Return the declared context around a crop and/or region (for prompts)."""
    result: dict[str, list[str]] = {"categories": [], "regions": [], "sources": [], "crops": []}
    for node, key in ((crop, "crops"), (region, "regions")):
        if not node:
            continue
        found = graph.find(node)
        if found is None:
            continue
        for neighbour in graph.neighbors(found.id):
            if neighbour.kind == NODE_CATEGORY:
                result["categories"].append(neighbour.label)
            elif neighbour.kind == NODE_REGION and key == "crops":
                result["regions"].append(neighbour.label)
            elif neighbour.kind == NODE_CROP and key == "regions":
                result["crops"].append(neighbour.label)
            elif neighbour.kind == NODE_SOURCE:
                result["sources"].append(neighbour.label)
    return {key: sorted(set(values)) for key, values in result.items()}


__all__ = [
    "NODE_ANIMAL",
    "NODE_CATEGORY",
    "NODE_COUNTRY",
    "NODE_CROP",
    "NODE_DOCUMENT",
    "NODE_KINDS",
    "NODE_REGION",
    "NODE_SOURCE",
    "RELATION_ABOUT_ANIMAL",
    "RELATION_ABOUT_COUNTRY",
    "RELATION_ABOUT_CROP",
    "RELATION_ABOUT_REGION",
    "RELATION_CO_OCCURS",
    "RELATION_IN_CATEGORY",
    "RELATION_SOURCED_FROM",
    "GraphEdge",
    "GraphNode",
    "KnowledgeGraph",
    "build_graph",
    "graph_from_database",
    "related_context",
]
