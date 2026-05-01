"""Query the Cognee knowledge graph directly — no LLM calls.

Usage:
    python query.py "bailey lemma"              # semantic search for entities
    python query.py --neighbors "bailey lemma"  # find + show connections
    python query.py --path "bailey lemma" "cylindric partitions"  # shortest path
"""

import asyncio
import sqlite3
import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import lancedb

DB_PATH = ".cognee_system/databases/cognee_db"


def find_lance_path():
    """Find the LanceDB path with data (handles user ID changes)."""
    import glob, os
    paths = glob.glob(".cognee_system/databases/*/*.lance.db")
    if not paths:
        raise FileNotFoundError("No LanceDB found in .cognee_system/databases/")
    # Pick the one that has an Entity_name table (non-empty)
    for p in sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True):
        try:
            db = lancedb.connect(p)
            if "Entity_name" in db.table_names():
                return p
        except Exception:
            continue
    return paths[0]


def load_model():
    return SentenceTransformer("RobBobin/math-embed")


def embed(model, text):
    v = model.encode([text], convert_to_numpy=True).astype(np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v[0]


def get_sqlite():
    return sqlite3.connect(DB_PATH)


def get_entity_name(conn, slug):
    c = conn.cursor()
    c.execute("SELECT attributes FROM nodes WHERE slug = ?", (slug,))
    r = c.fetchone()
    if r and r[0]:
        try:
            return json.loads(r[0]).get("name", slug[:12])
        except json.JSONDecodeError:
            pass
    return slug[:12]


def get_entity_paper(conn, slug):
    """Find the source paper for an entity via data_id -> TextDocument."""
    c = conn.cursor()
    c.execute("SELECT data_id FROM nodes WHERE slug = ?", (slug,))
    r = c.fetchone()
    if not r:
        return ""
    data_id = r[0]
    c.execute("SELECT attributes FROM nodes WHERE data_id = ? AND type = 'TextDocument'", (data_id,))
    r2 = c.fetchone()
    if r2 and r2[0]:
        try:
            return json.loads(r2[0]).get("name", "").replace(".pdf", "")
        except json.JSONDecodeError:
            pass
    return ""


def semantic_search(query, top_k=10):
    """Find entities by semantic similarity using math-embed."""
    model = load_model()
    q_vec = embed(model, query)

    db = lancedb.connect(find_lance_path())
    table = db.open_table("Entity_name")
    results = table.search(q_vec).limit(top_k).to_pandas()

    conn = get_sqlite()
    hits = []
    for _, row in results.iterrows():
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        name = payload.get("text", payload.get("name", ""))
        desc = payload.get("description", "")
        eid = row["id"]

        # Find paper
        paper = get_entity_paper(conn, eid)

        # LanceDB IDs have hyphens, SQLite slugs don't
        slug = eid.replace("-", "")
        hits.append({"id": slug, "name": name, "description": desc, "paper": paper,
                      "score": float(row.get("_distance", 0))})
    conn.close()
    return hits


def find_neighbors(query, top_k=5):
    """Find entity by semantic search, then show all its graph connections."""
    hits = semantic_search(query, top_k=1)
    if not hits:
        print("No matching entity found.")
        return

    target = hits[0]
    print(f"Entity: {target['name']}")
    if target["description"]:
        print(f"  {target['description']}")
    if target["paper"]:
        print(f"  Paper: {target['paper']}")
    print()

    conn = get_sqlite()
    c = conn.cursor()

    # Outgoing edges
    c.execute("""
        SELECT e.destination_node_id, e.relationship_name, n.type, n.attributes
        FROM edges e
        JOIN nodes n ON e.destination_node_id = n.slug
        WHERE e.source_node_id = ?
        AND e.relationship_name != 'is_a'
        AND n.type IN ('Entity', 'EntityType')
    """, (target["id"],))

    print("  Connections:")
    seen = set()
    for r in c.fetchall():
        attrs = json.loads(r[3]) if r[3] else {}
        name = attrs.get("name", r[0][:12])
        rel = r[1]
        if "relationship_name:" in rel:
            rel = rel.split("relationship_name:")[1].split(";")[0].strip()
        paper = get_entity_paper(conn, r[0])
        key = (rel, name)
        if key in seen:
            continue
        seen.add(key)
        paper_str = f"  [{paper}]" if paper else ""
        print(f"    --[{rel}]--> {name}{paper_str}")

    # Incoming edges
    c.execute("""
        SELECT e.source_node_id, e.relationship_name, n.type, n.attributes
        FROM edges e
        JOIN nodes n ON e.source_node_id = n.slug
        WHERE e.destination_node_id = ?
        AND e.relationship_name != 'is_a'
        AND n.type IN ('Entity', 'EntityType')
    """, (target["id"],))

    for r in c.fetchall():
        attrs = json.loads(r[3]) if r[3] else {}
        name = attrs.get("name", r[0][:12])
        rel = r[1]
        if "relationship_name:" in rel:
            rel = rel.split("relationship_name:")[1].split(";")[0].strip()
        key = (rel, name)
        if key in seen:
            continue
        seen.add(key)
        paper = get_entity_paper(conn, r[0])
        paper_str = f"  [{paper}]" if paper else ""
        print(f"    <--[{rel}]-- {name}{paper_str}")

    conn.close()


def find_path(start_query, end_query):
    """Find shortest path between two entities in the graph."""
    hits_start = semantic_search(start_query, top_k=1)
    hits_end = semantic_search(end_query, top_k=1)
    if not hits_start or not hits_end:
        print("Could not find one or both entities.")
        return

    start_id = hits_start[0]["id"]
    end_id = hits_end[0]["id"]
    print(f"From: {hits_start[0]['name']}")
    print(f"To:   {hits_end[0]['name']}")
    print()

    conn = get_sqlite()
    c = conn.cursor()

    # Build adjacency from edges (skip structural noise)
    SKIP = {"is_a", "authored", "wrote_about", "co_authored", "co_author",
            "contains", "is_part_of", "made_from"}
    c.execute("SELECT source_node_id, destination_node_id, relationship_name FROM edges")
    adj = {}
    for src, dst, rel in c.fetchall():
        if "relationship_name:" in rel:
            rel = rel.split("relationship_name:")[1].split(";")[0].strip()
        if rel in SKIP:
            continue
        adj.setdefault(src, []).append((dst, rel))
        adj.setdefault(dst, []).append((src, rel))

    # BFS
    from collections import deque
    visited = {start_id: None}
    queue = deque([(start_id, [])])
    found = False

    while queue:
        node, path = queue.popleft()
        if node == end_id:
            print(f"Path ({len(path)} hops):")
            for step_node, step_rel in path:
                name = get_entity_name(conn, step_node)
                print(f"  --[{step_rel}]--> {name}")
            name = get_entity_name(conn, end_id)
            print(f"  = {name}")
            found = True
            break
        for neighbor, rel in adj.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append((neighbor, path + [(node, rel)]))

    if not found:
        print("No path found.")
    conn.close()


def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        return

    if args[0] == "--neighbors":
        query = " ".join(args[1:])
        find_neighbors(query)
    elif args[0] == "--path":
        # Split on quotes or find the two arguments
        if '"' in " ".join(args[1:]):
            import shlex
            parts = shlex.split(" ".join(args[1:]))
        else:
            parts = [args[1], " ".join(args[2:])]
        find_path(parts[0], parts[1])
    else:
        query = " ".join(args)
        hits = semantic_search(query, top_k=10)
        for i, h in enumerate(hits, 1):
            paper = f"  [{h['paper']}]" if h["paper"] else ""
            print(f"[{i}] {h['name']}{paper}")
            if h["description"]:
                print(f"    {h['description'][:200]}")
            print()


if __name__ == "__main__":
    main()
