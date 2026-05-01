"""Export the math knowledge graph as a D3.js visualization.

Usage:
    python export_graph.py                          # Full graph
    python export_graph.py --center "conjecture 2.7" --hops 3   # Ego graph
"""

import sqlite3
import json
import sys
import re
from collections import Counter, deque

DB_PATH = ".cognee_system/databases/cognee_db"
OUTPUT_HTML = "graph.html"

# Skip noisy edge types
SKIP_EDGES = {"contains", "is_part_of", "made_from", "is_a"}
# Only traverse these edges for ego graph (structural math relationships)
TRAVERSAL_EDGES = {
    "proves", "uses", "generalizes", "specializes_to", "extends",
    "implies", "equivalent_to", "computes", "instance_of",
}


def clean_rel(rel):
    if "relationship_name:" in rel:
        rel = rel.split("relationship_name:")[1].split(";")[0].strip()
    return rel


def build_graph(center=None, max_hops=3):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # All entities
    c.execute("SELECT slug, data_id, attributes FROM nodes WHERE type = 'Entity'")
    slug_to_node = {}
    for r in c.fetchall():
        if not r[2]:
            continue
        try:
            attrs = json.loads(r[2])
        except:
            continue
        slug_to_node[r[0]] = {
            "name": attrs.get("name", ""),
            "desc": attrs.get("description", ""),
            "data_id": r[1],
        }

    # Entity types (for category coloring)
    c.execute("SELECT slug, attributes FROM nodes WHERE type = 'EntityType'")
    type_names = {}
    for r in c.fetchall():
        if not r[1]:
            continue
        try:
            type_names[r[0]] = json.loads(r[1]).get("name", "").lower()
        except:
            pass

    # is_a edges to assign categories
    c.execute("SELECT source_node_id, destination_node_id FROM edges WHERE relationship_name = 'is_a'")
    entity_cat = {}
    for r in c.fetchall():
        if r[1] in type_names:
            entity_cat[r[0]] = type_names[r[1]]

    # Paper names
    c.execute("SELECT data_id, attributes FROM nodes WHERE type = 'TextDocument'")
    papers = {}
    for r in c.fetchall():
        if not r[1]:
            continue
        try:
            papers[r[0]] = json.loads(r[1]).get("name", "").replace(".pdf", "")
        except:
            pass

    # All non-noise edges
    c.execute("SELECT source_node_id, destination_node_id, relationship_name FROM edges")
    all_edges = []
    adj = {}
    for r in c.fetchall():
        rel = clean_rel(r[2])
        if rel in SKIP_EDGES:
            continue
        src, dst = r[0], r[1]
        if src not in slug_to_node or dst not in slug_to_node:
            continue
        all_edges.append((src, dst, rel))
        if rel in TRAVERSAL_EDGES:
            adj.setdefault(src, []).append((dst, rel))
            adj.setdefault(dst, []).append((src, rel))

    conn.close()

    # If center specified, BFS to limit hops
    if center:
        # Find center by name match
        center_id = None
        for slug, info in slug_to_node.items():
            if info["name"].lower() == center.lower():
                center_id = slug
                break
        if not center_id:
            for slug, info in slug_to_node.items():
                if center.lower() in info["name"].lower():
                    center_id = slug
                    break
        if not center_id:
            print(f"Could not find '{center}'")
            sys.exit(1)

        distances = {center_id: 0}
        queue = deque([center_id])
        while queue:
            nid = queue.popleft()
            if distances[nid] >= max_hops:
                continue
            for neighbor, rel in adj.get(nid, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[nid] + 1
                    queue.append(neighbor)
        reachable = set(distances.keys())
    else:
        # Full graph — only keep nodes with edges
        reachable = set()
        for src, dst, rel in all_edges:
            reachable.add(src)
            reachable.add(dst)
        distances = {s: 0 for s in reachable}
        center_id = None

    # Build output
    nodes = []
    for slug in reachable:
        info = slug_to_node[slug]
        cat = entity_cat.get(slug, "")
        paper = papers.get(info["data_id"], "")
        nodes.append({
            "id": slug,
            "name": info["name"],
            "description": info["desc"],
            "category": cat,
            "paper": paper,
            "distance": distances.get(slug, 99),
            "isCenter": slug == center_id,
        })

    links = []
    seen = set()
    for src, dst, rel in all_edges:
        if src in reachable and dst in reachable:
            key = (src, dst, rel)
            if key not in seen:
                seen.add(key)
                links.append({"source": src, "target": dst, "rel": rel})

    return nodes, links, center_id is not None


def generate_html(nodes, links, has_center):
    edge_counts = Counter(l["rel"] for l in links)
    cat_counts = Counter(n["category"] for n in nodes if n["category"])

    # Category colors
    TYPE_COLORS = {
        "theorem": "#E74C3C",
        "conjecture": "#F39C12",
        "identity": "#9B59B6",
        "mathobject": "#4A90D9",
        "technique": "#2ECC71",
        "formula": "#1ABC9C",
        "proposition": "#E74C3C",
        "corollary": "#E74C3C",
        "lemma": "#E74C3C",
    }

    graph_data = {"nodes": nodes, "links": links}

    html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Q-Series Knowledge Graph</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0a; color: #eee; font-family: 'Menlo', monospace; overflow: hidden; }
  svg { width: 100vw; height: 100vh; }
  .node text { font-size: 10px; fill: #bbb; pointer-events: none; text-shadow: 0 0 5px #000, 0 0 3px #000; }
  .node circle { cursor: pointer; }
  #tooltip {
    position: absolute; background: rgba(12,12,12,0.97); border: 1px solid #555;
    padding: 14px 18px; border-radius: 8px; font-size: 12px; max-width: 450px;
    display: none; line-height: 1.7; z-index: 100; max-height: 70vh; overflow-y: auto;
  }
  #tooltip::-webkit-scrollbar { width: 4px; }
  #tooltip::-webkit-scrollbar-thumb { background: #444; border-radius: 2px; }
  #tooltip .name { font-size: 15px; font-weight: bold; color: #fff; }
  #tooltip .cat { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  #tooltip .paper { color: #4488ff; font-size: 11px; margin-top: 4px; }
  #tooltip .desc { color: #999; margin-top: 6px; }
  #tooltip .edges { margin-top: 10px; border-top: 1px solid #222; padding-top: 8px; }
  #tooltip .edge-item { color: #888; font-size: 11px; margin: 3px 0; }
  #tooltip .edge-rel { font-weight: bold; }
  #tooltip .close { position: absolute; top: 8px; right: 12px; color: #666; cursor: pointer; font-size: 16px; }
  #tooltip .close:hover { color: #fff; }
  #controls {
    position: absolute; top: 16px; left: 16px; background: rgba(10,10,10,0.96);
    border: 1px solid #333; border-radius: 8px; padding: 14px 18px; font-size: 12px;
    max-height: calc(100vh - 32px); overflow-y: auto; width: 230px; z-index: 50;
  }
  #controls::-webkit-scrollbar { width: 4px; }
  #controls::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
  #controls h2 { font-size: 14px; margin-bottom: 4px; color: #fff; }
  #controls .sub { font-size: 10px; color: #555; margin-bottom: 10px; }
  .stat { color: #555; font-size: 11px; margin-bottom: 10px; }
  #search { background: #111; border: 1px solid #333; color: #eee; padding: 7px 10px;
    border-radius: 4px; width: 100%; font-family: inherit; font-size: 12px; }
  #search:focus { outline: none; border-color: #6510F4; }
  h3 { font-size: 10px; color: #555; margin: 12px 0 6px; text-transform: uppercase; letter-spacing: 1.5px; }
  .filter-group label { display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 2px 0; color: #999; font-size: 11px; }
  .filter-group label:hover { color: #fff; }
  .filter-group input[type=checkbox] { accent-color: #6510F4; flex-shrink: 0; }
  .count { color: #444; margin-left: auto; font-size: 10px; }
  .btn-row { display: flex; gap: 6px; margin-bottom: 6px; }
  .btn { background: #151515; border: 1px solid #333; color: #888; padding: 3px 8px;
    border-radius: 3px; cursor: pointer; font-size: 10px; font-family: inherit; }
  .btn:hover { color: #fff; border-color: #555; }
  .legend-item { display: flex; align-items: center; gap: 6px; margin: 3px 0; font-size: 11px; color: #888; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .link-label { font-size: 9px; fill: #666; pointer-events: none; text-shadow: 0 0 4px #000, 0 0 2px #000; }
</style>
</head>
<body>
<div id="controls">
  <h2>Q-Series Knowledge Graph</h2>
  <div class="sub">76 papers · math-aware extraction</div>
  <div class="stat" id="stats"></div>
  <input type="text" id="search" placeholder="Search... (click nodes for details)">
  <h3>Node types</h3>
  <div id="cat-legend"></div>
  <h3>Edge types</h3>
  <div class="btn-row">
    <button class="btn" onclick="toggleEdges(true)">All</button>
    <button class="btn" onclick="toggleEdges(false)">None</button>
  </div>
  <div id="edge-filters"></div>
</div>
<div id="tooltip"></div>
<svg></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = GRAPH_DATA_PLACEHOLDER;
const TYPE_COLORS = COLORS_PLACEHOLDER;
const edgeCounts = EDGE_COUNTS_PLACEHOLDER;
const hasCenter = HAS_CENTER_PLACEHOLDER;

// Build legend
const catLegend = document.getElementById('cat-legend');
Object.entries(TYPE_COLORS).forEach(([cat, color]) => {
  const cnt = data.nodes.filter(n => n.category === cat).length;
  if (cnt === 0) return;
  catLegend.innerHTML += `<div class="legend-item"><span class="legend-dot" style="background:${color}"></span>${cat} <span class="count">${cnt}</span></div>`;
});

// Edge filters
const activeEdges = new Set(Object.keys(edgeCounts));
const edgeDiv = document.getElementById('edge-filters');
Object.entries(edgeCounts).sort((a,b) => b[1]-a[1]).forEach(([rel, cnt]) => {
  edgeDiv.innerHTML += `<div class="filter-group"><label><input type="checkbox" checked data-rel="${rel}"> ${rel} <span class="count">${cnt}</span></label></div>`;
});
edgeDiv.addEventListener('change', e => {
  const r = e.target.dataset.rel; if (!r) return;
  e.target.checked ? activeEdges.add(r) : activeEdges.delete(r);
  applyFilters();
});
function toggleEdges(on) {
  activeEdges.clear();
  if (on) Object.keys(edgeCounts).forEach(r => activeEdges.add(r));
  edgeDiv.querySelectorAll('input').forEach(cb => cb.checked = on);
  applyFilters();
}

const adj = new Map();
data.links.forEach(l => {
  const s = typeof l.source === 'object' ? l.source.id : l.source;
  const t = typeof l.target === 'object' ? l.target.id : l.target;
  if (!adj.has(s)) adj.set(s, []);
  if (!adj.has(t)) adj.set(t, []);
  adj.get(s).push(l);
  adj.get(t).push(l);
});

const degreeMap = {};
data.links.forEach(l => {
  const s = typeof l.source === 'object' ? l.source.id : l.source;
  const t = typeof l.target === 'object' ? l.target.id : l.target;
  degreeMap[s] = (degreeMap[s]||0)+1;
  degreeMap[t] = (degreeMap[t]||0)+1;
});

const W = window.innerWidth, H = window.innerHeight;
const svg = d3.select('svg');
const defs = svg.append('defs');
defs.append('marker').attr('id','arrow').attr('viewBox','0 0 10 6')
  .attr('refX',20).attr('refY',3).attr('markerWidth',6).attr('markerHeight',4)
  .attr('orient','auto')
  .append('path').attr('d','M0,0 L10,3 L0,6').attr('fill','#333');
const g = svg.append('g');
svg.call(d3.zoom().scaleExtent([0.05, 12]).on('zoom', e => g.attr('transform', e.transform)));

const centerNode = data.nodes.find(n => n.isCenter);
if (centerNode) { centerNode.fx = W/2; centerNode.fy = H/2; }

const forces = {
  link: d3.forceLink(data.links).id(d => d.id).distance(80).strength(0.3),
  charge: d3.forceManyBody().strength(-150),
  center: d3.forceCenter(W/2, H/2),
  collision: d3.forceCollide().radius(15),
};
if (hasCenter) forces.radial = d3.forceRadial(d => d.distance * 130, W/2, H/2).strength(0.25);

const simulation = d3.forceSimulation(data.nodes);
Object.entries(forces).forEach(([k,v]) => simulation.force(k, v));

const linkSel = g.append('g').selectAll('line').data(data.links).join('line')
  .attr('stroke', '#282828').attr('stroke-width', 1).attr('stroke-opacity', 0.5)
  .attr('marker-end', 'url(#arrow)');

const linkLabels = g.append('g').selectAll('text').data(data.links).join('text')
  .attr('class', 'link-label')
  .text(d => d.rel)
  .attr('text-anchor', 'middle')
  .attr('dy', -4);

const nodeSel = g.append('g').selectAll('g').data(data.nodes).join('g')
  .attr('class','node')
  .call(d3.drag()
    .on('start', (e,d)=>{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
    .on('drag', (e,d)=>{ d.fx=e.x; d.fy=e.y; })
    .on('end', (e,d)=>{ if(!e.active) simulation.alphaTarget(0); if(!d.isCenter){d.fx=null; d.fy=null;} }));

const nodeR = d => d.isCenter ? 18 : Math.max(4, Math.min(14, 3 + Math.sqrt(degreeMap[d.id]||1)*2));
const nodeColor = d => d.isCenter ? '#ff3366' : (TYPE_COLORS[d.category] || '#555');

nodeSel.append('circle')
  .attr('r', nodeR)
  .attr('fill', nodeColor)
  .attr('stroke', d => d.isCenter ? '#fff' : '#111')
  .attr('stroke-width', d => d.isCenter ? 2.5 : 0.5);

nodeSel.append('text')
  .attr('dx', d => nodeR(d) + 4).attr('dy', 3)
  .text(d => d.name.length > 40 ? d.name.slice(0,40)+'...' : d.name)
  .style('display', d => (degreeMap[d.id]||0) >= 3 || d.isCenter ? 'block' : 'none')
  .style('font-weight', d => d.isCenter ? 'bold' : 'normal')
  .style('fill', d => d.isCenter ? '#ff3366' : '#bbb');

// Click-to-pin tooltip
const tooltip = d3.select('#tooltip');
let pinned = null;

function highlightNode(d) {
  const connIds = new Set([d.id]);
  (adj.get(d.id)||[]).forEach(l => {
    connIds.add(l.source.id||l.source); connIds.add(l.target.id||l.target);
  });
  nodeSel.select('circle').attr('opacity', n => connIds.has(n.id) ? 1 : 0.1);
  nodeSel.select('text').attr('opacity', n => connIds.has(n.id) ? 1 : 0.1);
  linkSel.attr('stroke-opacity', l => {
    const s = l.source.id||l.source, t = l.target.id||l.target;
    return s===d.id||t===d.id ? 0.9 : 0.03;
  }).attr('stroke', l => {
    const s = l.source.id||l.source, t = l.target.id||l.target;
    return s===d.id||t===d.id ? '#666' : '#282828';
  });
}
function resetAll() {
  nodeSel.select('circle').attr('opacity', 1).attr('stroke', d => d.isCenter ? '#fff' : '#111').attr('stroke-width', d => d.isCenter ? 2.5 : 0.5);
  nodeSel.select('text').attr('opacity', 1);
  linkSel.attr('stroke-opacity', 0.5).attr('stroke', '#282828');
}
function dismiss() { pinned = null; tooltip.style('display','none'); resetAll(); }

nodeSel.on('click', (e, d) => {
  e.stopPropagation();
  if (pinned === d.id) { dismiss(); return; }
  pinned = d.id;
  const edgeHtml = (adj.get(d.id)||[]).map(l => {
    const s = l.source.id||l.source, t = l.target.id||l.target;
    const other = s===d.id ? (l.target.name||t) : (l.source.name||s);
    const dir = s===d.id ? '\u2192' : '\u2190';
    const c = TYPE_COLORS[l.rel] || '#6510F4';
    return `<div class="edge-item">${dir} <span class="edge-rel" style="color:${c}">${l.rel}</span> ${other}</div>`;
  }).join('');
  const catColor = TYPE_COLORS[d.category] || '#888';
  tooltip.style('display','block')
    .style('left', (e.pageX+15)+'px').style('top', (e.pageY-10)+'px')
    .html(`<span class="close" onclick="document.getElementById('tooltip').style.display='none'">\u2715</span>
     <div class="name">${d.name}</div>
     <div class="cat" style="color:${catColor}">${d.category || 'uncategorized'}</div>
     ${d.paper ? '<div class="paper">\ud83d\udcc4 '+d.paper+'</div>' : ''}
     ${d.description ? '<div class="desc">'+d.description+'</div>' : ''}
     ${edgeHtml ? '<div class="edges">'+edgeHtml+'</div>' : ''}`);
  d3.select(e.currentTarget).select('circle').attr('stroke','#fff').attr('stroke-width',2.5);
  highlightNode(d);
});
nodeSel.on('mouseover', (e, d) => { if (pinned) return; d3.select(e.currentTarget).select('circle').attr('stroke','#fff').attr('stroke-width',2); highlightNode(d); });
nodeSel.on('mouseout', (e, d) => { if (pinned) return; d3.select(e.currentTarget).select('circle').attr('stroke', d.isCenter?'#fff':'#111').attr('stroke-width', d.isCenter?2.5:0.5); resetAll(); });
svg.on('click', () => dismiss());

simulation.on('tick', () => {
  linkSel.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  linkLabels.attr('x',d=>(d.source.x+d.target.x)/2).attr('y',d=>(d.source.y+d.target.y)/2);
  nodeSel.attr('transform', d=>`translate(${d.x},${d.y})`);
});

function applyFilters() {
  linkSel.style('display', d => activeEdges.has(d.rel) ? null : 'none');
  linkLabels.style('display', d => activeEdges.has(d.rel) ? null : 'none');
  const conn = new Set();
  data.links.forEach(l => { if (activeEdges.has(l.rel)) { conn.add(l.source.id||l.source); conn.add(l.target.id||l.target); } });
  nodeSel.style('display', d => conn.has(d.id) ? null : 'none');
  document.getElementById('stats').textContent = `${conn.size} nodes \xb7 ${data.links.filter(l=>activeEdges.has(l.rel)).length} edges`;
}

document.getElementById('search').addEventListener('input', function() {
  const q = this.value.toLowerCase();
  if (!q) { resetAll(); nodeSel.select('text').style('display', d => (degreeMap[d.id]||0)>=3||d.isCenter?'block':'none'); return; }
  nodeSel.select('circle').attr('opacity', d => d.name.toLowerCase().includes(q) ? 1 : 0.06);
  nodeSel.select('text').style('display', d => d.name.toLowerCase().includes(q) ? 'block' : 'none')
    .attr('fill', d => d.name.toLowerCase().includes(q) ? '#ff0' : '#bbb');
  linkSel.attr('stroke-opacity', d => {
    const s=(d.source.name||'').toLowerCase(), t=(d.target.name||'').toLowerCase();
    return s.includes(q)||t.includes(q) ? 0.6 : 0.02;
  });
});

document.getElementById('stats').textContent = `${data.nodes.length} nodes \xb7 ${data.links.length} edges`;
</script>
</body>
</html>"""

    html = html.replace("GRAPH_DATA_PLACEHOLDER", json.dumps(graph_data))
    html = html.replace("COLORS_PLACEHOLDER", json.dumps(TYPE_COLORS))
    html = html.replace("EDGE_COUNTS_PLACEHOLDER", json.dumps(dict(edge_counts.most_common())))
    html = html.replace("HAS_CENTER_PLACEHOLDER", "true" if has_center else "false")

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"Saved {OUTPUT_HTML} ({len(html)//1024}KB)")


def main():
    args = sys.argv[1:]
    center = None
    max_hops = 3

    if "--center" in args:
        idx = args.index("--center")
        center = args[idx + 1]
        args = args[:idx] + args[idx+2:]
    if "--hops" in args:
        idx = args.index("--hops")
        max_hops = int(args[idx + 1])

    nodes, links, has_center = build_graph(center, max_hops)
    print(f"Nodes: {len(nodes)}, Edges: {len(links)}" +
          (f" (centered on '{center}', {max_hops} hops)" if center else " (full graph)"))
    generate_html(nodes, links, has_center)


if __name__ == "__main__":
    main()
