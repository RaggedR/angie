# Angie — Q-Series Knowledge Graph & Warnaar 2.7 Proof Strategy Wiki

**[Live Wiki → raggedr.github.io/angie](https://raggedr.github.io/angie/)**

An interactive wiki mapping three proof strategies for [Warnaar's Conjecture 2.7](https://arxiv.org/abs/2101.01966) (positivity of coefficients in the $A_2$ Andrews-Gordon identities), built from a knowledge graph of ~70 q-series and combinatorics papers.

## The Wiki

The wiki identifies three attack vectors for proving Conjecture 2.7, extracted from a knowledge graph (1,765 nodes, 3,348 edges):

| # | Approach | Strategy | Colour |
|---|----------|----------|--------|
| ① | **$A_2$ Bailey Lemma** | Algebraic — show the Bailey machinery produces manifestly non-negative expressions | Amber |
| ② | **Cylindric Partitions** | Combinatorial — exhibit objects counted by $a^G_{\lambda;r}(q)$ | Green |
| ③ | **Hall-Littlewood / Schur** | Representation-theoretic — expand in a known positive basis | Purple |

Each page includes graph connections, key theorems, and arXiv-linked references.

- **Wiki**: [raggedr.github.io/angie](https://raggedr.github.io/angie/)
- **Interactive graph visualization**: [raggedr.github.io/angie/graph.html](https://raggedr.github.io/angie/graph.html)

## Knowledge Graph Tools

The wiki was built from a [Cognee](https://github.com/topoteretes/cognee)-powered knowledge graph using a custom math-domain embedding model.

### Query the graph locally

Requires the `.cognee_system/` databases (not in git — regenerate with the ingestion scripts).

```bash
# Semantic search for entities
python query.py "bailey lemma"

# Find an entity + all its graph connections
python query.py --neighbors "cylindric partitions"

# Shortest path between two concepts
python query.py --path "bailey lemma" "cylindric partitions"
```

### Rebuild the graph

```bash
# 1. Ingest papers (PDFs → text chunks)
python ingest_papers.py

# 2. Run Cognee's cognify pipeline (extract entities + relationships)
python run_cognify.py

# 3. Export graph data for the D3 visualization
python export_graph.py
```

### Custom embedding model

The graph uses [`RobBobin/math-embed`](https://huggingface.co/RobBobin/math-embed), a 768-dim model fine-tuned on combinatorics/q-series papers using knowledge-graph-guided contrastive learning on top of `allenai/specter2_base`. Achieves MRR 0.816 vs OpenAI's 0.461 on mathematical paper retrieval.

- `MathEmbedEngine.py` — Cognee adapter for math-embed
- `cognee_math_embed.patch` — patch to Cognee's embedding engine to register the provider

## Files

| File | Purpose |
|------|---------|
| `index.html` | The wiki (served by GitHub Pages) |
| `graph.html` | Interactive D3.js force-directed knowledge graph |
| `graph_data.json` | Graph data (nodes + edges) for the visualization |
| `query.py` | CLI tool for querying the knowledge graph |
| `ingest_papers.py` | Ingest PDFs into Cognee |
| `run_cognify.py` | Run Cognee's entity/relationship extraction |
| `resume_cognify.py` | Resume an interrupted cognify run |
| `export_graph.py` | Export graph to JSON for D3 visualization |
| `math_graph_prompt.txt` | LLM prompt used for entity extraction |
| `MathEmbedEngine.py` | Custom embedding engine for Cognee |
| `cognee_math_embed.patch` | Patch to register math-embed in Cognee |

## License

The tools and wiki are open source. The underlying papers are linked to their arXiv pages.
