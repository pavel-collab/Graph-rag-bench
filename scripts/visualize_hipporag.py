"""
Visualize HippoRAG 2 knowledge graph.

Loads the igraph pickle from storage, builds an interactive HTML (pyvis)
and optionally exports GraphML for Gephi / /graphify.

Usage:
    uv run scripts/visualize_hipporag.py
    uv run scripts/visualize_hipporag.py --out data/graph_hipporag.html
    uv run scripts/visualize_hipporag.py --graphml data/graph_hipporag.graphml
    uv run scripts/visualize_hipporag.py --max-nodes 500 --no-chunks
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── helpers ──────────────────────────────────────────────────────────────────

def find_pickle(save_dir: str) -> Path:
    matches = list(Path(save_dir).rglob("graph.pickle"))
    if not matches:
        raise FileNotFoundError(
            f"graph.pickle not found under {save_dir!r}. "
            "Run scripts/ingest_hipporag.py first."
        )
    return matches[0]


def load_igraph(pickle_path: Path):
    import igraph as ig
    g = ig.Graph.Read_Pickle(str(pickle_path))
    return g


def print_stats(g) -> None:
    names = g.vs["name"] if "name" in g.vs.attribute_names() else []
    n_chunks = sum(1 for n in names if n.startswith("chunk-"))
    n_entities = len(names) - n_chunks
    print(f"  Nodes  : {g.vcount()}  ({n_entities} entities, {n_chunks} passages)")
    print(f"  Edges  : {g.ecount()}")
    print(f"  Directed: {g.is_directed()}")


def _node_label(v) -> str:
    """Human-readable label for a graph vertex."""
    attrs = v.attributes()
    content = attrs.get("content", "")
    if content:
        return content[:60] + ("…" if len(content) > 60 else "")
    name = attrs.get("name", str(v.index))
    # strip hash prefix, show up to 40 chars
    for prefix in ("entity-", "chunk-"):
        if name.startswith(prefix):
            return name[len(prefix) :][:40]
    return name[:40]


# ── pyvis HTML ────────────────────────────────────────────────────────────────

def build_html(g, out: Path, max_nodes: int, hide_chunks: bool) -> None:
    from pyvis.network import Network

    # --- collect vertices ---
    all_vs = list(g.vs)
    chunk_vs = [v for v in all_vs if v["name"].startswith("chunk-")] if "name" in g.vs.attribute_names() else []
    entity_vs = [v for v in all_vs if not v["name"].startswith("chunk-")] if "name" in g.vs.attribute_names() else all_vs

    if hide_chunks:
        keep_names = {v["name"] for v in entity_vs[:max_nodes]}
    else:
        n_ent = min(len(entity_vs), int(max_nodes * 0.80))
        n_chunk = min(len(chunk_vs), max_nodes - n_ent)
        keep_names = (
            {v["name"] for v in entity_vs[:n_ent]}
            | {v["name"] for v in chunk_vs[:n_chunk]}
        )

    if len(all_vs) > max_nodes:
        print(f"  Showing {len(keep_names)} of {g.vcount()} nodes (--max-nodes {max_nodes})")

    net = Network(
        height="900px",
        width="100%",
        directed=g.is_directed(),
        bgcolor="#0d1117",
        font_color="#c9d1d9",
        notebook=False,
        cdn_resources="in_line",
    )

    name_to_id: dict[str, str] = {}

    for v in g.vs:
        name = v["name"] if "name" in v.attributes() else str(v.index)
        if name not in keep_names:
            continue
        label = _node_label(v)
        is_chunk = name.startswith("chunk-")
        tooltip = (
            f"[PASSAGE]\n{v.attributes().get('content', name)}"
            if is_chunk
            else f"[ENTITY]\n{v.attributes().get('content', name)}"
        )
        net.add_node(
            name,
            label="" if is_chunk else label,
            title=tooltip,
            color="#e05c5c" if is_chunk else "#58a6ff",
            size=12 if is_chunk else 7,
            shape="square" if is_chunk else "dot",
        )
        name_to_id[name] = name

    for e in g.es:
        src = g.vs[e.source]["name"] if "name" in g.vs[e.source].attributes() else str(e.source)
        tgt = g.vs[e.target]["name"] if "name" in g.vs[e.target].attributes() else str(e.target)
        if src in keep_names and tgt in keep_names:
            weight = e.attributes().get("weight", 1.0)
            net.add_edge(src, tgt, value=float(weight), color="#30363d")

    net.set_options("""{
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.003,
          "springLength": 120,
          "springConstant": 0.05,
          "damping": 0.4
        },
        "minVelocity": 0.5,
        "maxVelocity": 50
      },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }""")

    out.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out))
    print(f"  HTML → {out}")


# ── GraphML export ─────────────────────────────────────────────────────────────

def export_graphml(g, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    g.write_graphml(str(out))
    print(f"  GraphML → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize HippoRAG 2 knowledge graph")
    parser.add_argument("--save-dir", default=settings.hipporag_working_dir,
                        help="HippoRAG save_dir (default: from config)")
    parser.add_argument("--out", type=Path, default=Path("data/graph_hipporag.html"),
                        help="Output HTML path")
    parser.add_argument("--graphml", type=Path, default=None,
                        help="Also export GraphML (for Gephi / /graphify)")
    parser.add_argument("--max-nodes", type=int, default=1000,
                        help="Max nodes to render (default: 1000)")
    parser.add_argument("--no-chunks", action="store_true",
                        help="Hide passage nodes, show entities only")
    args = parser.parse_args()

    pickle_path = find_pickle(args.save_dir)
    print(f"Graph: {pickle_path}")
    g = load_igraph(pickle_path)
    print_stats(g)

    print("Building HTML…")
    build_html(g, args.out, args.max_nodes, args.no_chunks)

    if args.graphml:
        print("Exporting GraphML…")
        export_graphml(g, args.graphml)

    print(f"\nOpen: open {args.out}")


if __name__ == "__main__":
    main()
