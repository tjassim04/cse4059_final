# generate_one_graph_many_patterns.py
import os
import random
from collections import Counter


def normalize_edges(edges):
    out = set()
    for u, v in edges:
        if u != v:
            out.add((min(u, v), max(u, v)))
    return sorted(out)


PATTERNS = [
    {"name": "edge_2", "n": 2, "edges": [(0, 1)]},
    {"name": "path_3", "n": 3, "edges": [(0, 1), (1, 2)]},
    {"name": "triangle_3", "n": 3, "edges": [(0, 1), (0, 2), (1, 2)]},
    {"name": "path_4", "n": 4, "edges": [(0, 1), (1, 2), (2, 3)]},
    {"name": "star_4", "n": 4, "edges": [(0, 1), (0, 2), (0, 3)]},
    {"name": "cycle_4", "n": 4, "edges": [(0, 1), (1, 2), (2, 3), (0, 3)]},
    {"name": "tailed_triangle_4", "n": 4, "edges": [(0, 1), (0, 2), (1, 2), (2, 3)]},
    {"name": "diamond_4", "n": 4, "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]},
    {"name": "clique_4", "n": 4, "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]},
    {"name": "path_5", "n": 5, "edges": [(0, 1), (1, 2), (2, 3), (3, 4)]},
    {"name": "star_5", "n": 5, "edges": [(0, 1), (0, 2), (0, 3), (0, 4)]},
    {"name": "cycle_5", "n": 5, "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]},
    {"name": "house_5", "n": 5, "edges": [(0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (2, 4)]},
    {"name": "bowtie_5", "n": 5, "edges": [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)]},
    {"name": "clique_5", "n": 5, "edges": [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4),
    ]},
]


def write_edges(path, edges):
    with open(path, "w") as f:
        for u, v in sorted(edges):
            f.write(f"{u} {v}\n")


def main():
    seed = 42
    rng = random.Random(seed)

    total_components = 300

    data_edges = set()
    current_vertex = 0
    planted_counts = Counter()

    os.makedirs("patterns", exist_ok=True)

    # Write the 15 query pattern files.
    with open("queries.txt", "w") as qf:
        for p in PATTERNS:
            pattern_file = f"patterns/pattern_{p['name']}.txt"
            write_edges(pattern_file, normalize_edges(p["edges"]))
            qf.write(f"{p['name']} {pattern_file} {p['n']}\n")

    # Make ONE data graph by planting randomly selected motif components.
    for _ in range(total_components):
        p = rng.choice(PATTERNS)
        base = current_vertex

        for u, v in normalize_edges(p["edges"]):
            data_edges.add((base + u, base + v))

        current_vertex += p["n"]
        planted_counts[p["name"]] += 1

    write_edges("data_edges.txt", data_edges)

    with open("metadata.txt", "w") as f:
        f.write(f"seed {seed}\n")
        f.write(f"num_vertices {current_vertex}\n")
        f.write(f"num_edges {len(data_edges)}\n")
        f.write(f"total_components {total_components}\n")
        for name, count in sorted(planted_counts.items()):
            f.write(f"planted_components {name} {count}\n")

    print("Generated ONE mixed data graph.")
    print(f"vertices: {current_vertex}")
    print(f"edges: {len(data_edges)}")
    print("Pattern files written to ./patterns/")
    print("Query list written to queries.txt")


if __name__ == "__main__":
    main()