// dataset_generator.cpp
//
// Multi-mode synthetic graph generator for the subgraph matching benchmark.
//
// Modes:
//   motif    - Plant random motif components (original behavior, default).
//   er       - Erdős–Rényi G(n, p) random graph by edge-skipping (Batagelj-Brandes).
//   ba       - Barabási–Albert preferential attachment.
//   overlap  - Motif planting with shared "hub" vertices, producing
//              non-trivial overlap between planted motifs.
//   convert  - Convert an existing edge list (e.g. SNAP graph) into a
//              dataset directory: data_edges.txt + metadata.txt + queries.
//   suite    - Generate a multi-dataset evaluation suite covering several
//              sizes and graph models. Also emits a run_suite.sh helper.
//   help     - Print usage.
//
// Backward compatibility: when invoked with no arguments or with a numeric
// first argument, this behaves identically to the original generator
// (motif mode, seed=42, n_components=300, output to current directory).
//
// Each mode writes (relative to the chosen out_dir):
//   patterns/pattern_<name>.txt   - one pattern query graph per file
//   queries.txt                   - <name> <pattern_file> <n> per line
//   data_edges.txt                - sorted, deduplicated, undirected edges
//   metadata.txt                  - seed, mode, num_vertices, num_edges, ...
//
// Compile:
//   g++ -O2 -std=c++17 dataset_generator.cpp -o dataset_generator
// (Compiles cleanly under nvcc as well; uses POSIX mkdir, not std::filesystem.)

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

// ============================================================================
// Patterns (same 15 motifs as the original Python generator)
// ============================================================================
struct Pattern {
    std::string name;
    int n;
    std::vector<std::pair<int, int>> edges;
};

static const std::vector<Pattern> PATTERNS = {
    {"edge_2",            2, {{0, 1}}},
    {"path_3",            3, {{0, 1}, {1, 2}}},
    {"triangle_3",        3, {{0, 1}, {0, 2}, {1, 2}}},
    {"path_4",            4, {{0, 1}, {1, 2}, {2, 3}}},
    {"star_4",            4, {{0, 1}, {0, 2}, {0, 3}}},
    {"cycle_4",           4, {{0, 1}, {1, 2}, {2, 3}, {0, 3}}},
    {"tailed_triangle_4", 4, {{0, 1}, {0, 2}, {1, 2}, {2, 3}}},
    {"diamond_4",         4, {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}}},
    {"clique_4",          4, {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}},
    {"path_5",            5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}},
    {"star_5",            5, {{0, 1}, {0, 2}, {0, 3}, {0, 4}}},
    {"cycle_5",           5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 4}}},
    {"house_5",           5, {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {1, 4}, {2, 4}}},
    {"bowtie_5",          5, {{0, 1}, {0, 2}, {1, 2}, {0, 3}, {0, 4}, {3, 4}}},
    {"clique_5",          5, {
        {0, 1}, {0, 2}, {0, 3}, {0, 4},
        {1, 2}, {1, 3}, {1, 4},
        {2, 3}, {2, 4},
        {3, 4},
    }},
};

// ============================================================================
// Common helpers
// ============================================================================
static std::vector<std::pair<int, int>>
normalize_edges(const std::vector<std::pair<int, int>>& edges) {
    std::set<std::pair<int, int>> s;
    for (const auto& e : edges) {
        if (e.first != e.second) {
            s.insert({std::min(e.first, e.second), std::max(e.first, e.second)});
        }
    }
    return std::vector<std::pair<int, int>>(s.begin(), s.end());
}

static void ensure_directory(const std::string& path) {
    if (path.empty()) return;
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i) {
        char c = path[i];
        acc.push_back(c);
        bool at_end = (i + 1 == path.size());
        if (c == '/' || at_end) {
            if (acc == "/" || acc == "." || acc == "./") continue;
            std::string to_make = acc;
            if (to_make.size() > 1 && to_make.back() == '/') to_make.pop_back();
            int rc = mkdir(to_make.c_str(), 0755);
            if (rc != 0 && errno != EEXIST) {
                std::cerr << "ERROR: could not create directory '"
                          << to_make << "': " << std::strerror(errno) << "\n";
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

static void write_edges_set(const std::string& path,
                            const std::set<std::pair<int, int>>& edges) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "ERROR: could not open " << path << " for writing\n";
        std::exit(EXIT_FAILURE);
    }
    // std::set is already sorted lexicographically.
    for (const auto& e : edges) {
        f << e.first << " " << e.second << "\n";
    }
}

static void write_edges_vec(const std::string& path,
                            std::vector<std::pair<int, int>> edges) {
    std::sort(edges.begin(), edges.end());
    std::ofstream f(path);
    if (!f) {
        std::cerr << "ERROR: could not open " << path << " for writing\n";
        std::exit(EXIT_FAILURE);
    }
    for (const auto& e : edges) {
        f << e.first << " " << e.second << "\n";
    }
}

static void write_pattern_files_and_queries(const std::string& out_dir) {
    std::string patterns_dir = out_dir + "/patterns";
    ensure_directory(patterns_dir);

    std::ofstream qf(out_dir + "/queries.txt");
    if (!qf) {
        std::cerr << "ERROR: could not open " << out_dir
                  << "/queries.txt for writing\n";
        std::exit(EXIT_FAILURE);
    }
    for (const auto& p : PATTERNS) {
        std::string pf_full = patterns_dir + "/pattern_" + p.name + ".txt";
        write_edges_vec(pf_full, normalize_edges(p.edges));
        // queries.txt holds RELATIVE paths so the dataset is portable.
        qf << p.name << " patterns/pattern_" << p.name << ".txt "
           << p.n << "\n";
    }
}

// ============================================================================
// Graph specification: what every generator produces
// ============================================================================
struct GraphSpec {
    int num_vertices = 0;
    std::set<std::pair<int, int>> edges;
    // Extra "key value" lines to append to metadata.txt, in order of insertion.
    std::vector<std::pair<std::string, std::string>> extra;
};

static void write_dataset(const std::string& out_dir,
                          const GraphSpec& gs,
                          unsigned seed,
                          const std::string& mode) {
    ensure_directory(out_dir);
    write_pattern_files_and_queries(out_dir);

    write_edges_set(out_dir + "/data_edges.txt", gs.edges);

    std::ofstream f(out_dir + "/metadata.txt");
    if (!f) {
        std::cerr << "ERROR: could not open " << out_dir
                  << "/metadata.txt for writing\n";
        std::exit(EXIT_FAILURE);
    }
    // IMPORTANT ordering: numeric "key int" pairs MUST come before any line
    // whose value is non-integer (e.g. "mode er", "planted_components edge_2 17").
    // The simple parser used by the solver does `stream >> key >> int`, so the
    // first non-int value after the search target would otherwise put the
    // stream into a fail state. Put num_vertices and num_edges up top.
    f << "seed " << seed << "\n";
    f << "num_vertices " << gs.num_vertices << "\n";
    f << "num_edges " << gs.edges.size() << "\n";
    f << "mode " << mode << "\n";
    for (const auto& kv : gs.extra) {
        f << kv.first << " " << kv.second << "\n";
    }

    double avg_deg = (gs.num_vertices > 0)
        ? (2.0 * (double)gs.edges.size() / (double)gs.num_vertices)
        : 0.0;
    std::cout << "[" << mode << "] " << out_dir
              << "  V=" << gs.num_vertices
              << "  E=" << gs.edges.size()
              << "  avg_deg=" << avg_deg << "\n";
}

// ============================================================================
// Generator: motif planting (original behavior)
// ============================================================================
static GraphSpec generate_motif(unsigned seed, int n_components) {
    GraphSpec gs;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pat_dist(0, (int)PATTERNS.size() - 1);

    int next_v = 0;
    std::map<std::string, int> planted;

    for (int i = 0; i < n_components; ++i) {
        const Pattern& p = PATTERNS[pat_dist(rng)];
        int base = next_v;
        for (const auto& e : normalize_edges(p.edges)) {
            gs.edges.insert({base + e.first, base + e.second});
        }
        next_v += p.n;
        planted[p.name] += 1;
    }

    gs.num_vertices = next_v;
    gs.extra.push_back({"total_components", std::to_string(n_components)});
    for (const auto& kv : planted) {
        gs.extra.push_back({"planted_components",
                            kv.first + " " + std::to_string(kv.second)});
    }
    return gs;
}

// ============================================================================
// Generator: Erdős–Rényi G(n, p) using the Batagelj–Brandes edge-skip method.
// O(m) expected time regardless of density.
// ============================================================================
static GraphSpec generate_er(unsigned seed, int n, double avg_degree) {
    GraphSpec gs;
    gs.num_vertices = n;

    if (n < 2 || avg_degree <= 0.0) {
        gs.extra.push_back({"er_target_avg_degree", std::to_string(avg_degree)});
        return gs;
    }

    double p = avg_degree / (double)(n - 1);
    if (p > 1.0) p = 1.0;

    if (p >= 1.0) {
        // Complete graph.
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                gs.edges.insert({i, j});
            }
        }
    } else {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        const double lp = std::log(1.0 - p);

        // Standard Batagelj-Brandes loop. v iterates rows, w iterates columns
        // within each row (w < v always when an edge is emitted).
        long long v = 1, w = -1;
        while (v < n) {
            double r = u01(rng);
            double one_minus_r = std::max(1.0 - r, 1e-300);
            long long skip = (long long)std::floor(std::log(one_minus_r) / lp);
            w = w + 1 + skip;
            while (w >= v && v < n) {
                w -= v;
                v += 1;
            }
            if (v < n) {
                gs.edges.insert({(int)w, (int)v});
            }
        }
    }

    gs.extra.push_back({"er_target_p", std::to_string(p)});
    gs.extra.push_back({"er_target_avg_degree", std::to_string(avg_degree)});
    if (gs.num_vertices > 0) {
        double realized =
            2.0 * (double)gs.edges.size() / (double)gs.num_vertices;
        gs.extra.push_back({"er_realized_avg_degree", std::to_string(realized)});
    }
    return gs;
}

// ============================================================================
// Generator: Barabási–Albert preferential attachment.
// Start from a complete graph on m+1 vertices. For each new vertex v, pick
// m existing vertices weighted by degree and connect v to them.
// ============================================================================
static GraphSpec generate_ba(unsigned seed, int n, int m) {
    GraphSpec gs;
    if (n < 1) return gs;
    if (m < 1) m = 1;
    if (m >= n) m = n - 1;
    gs.num_vertices = n;

    std::mt19937 rng(seed);

    int initial = std::min(m + 1, n);
    // Endpoint multiset: each vertex appears once per incident edge. Sampling
    // a uniformly random index of this vector reproduces the degree-weighted
    // distribution.
    std::vector<int> endpoints;
    endpoints.reserve(2 * (size_t)n * (size_t)m);

    for (int i = 0; i < initial; ++i) {
        for (int j = i + 1; j < initial; ++j) {
            gs.edges.insert({i, j});
            endpoints.push_back(i);
            endpoints.push_back(j);
        }
    }

    for (int v = initial; v < n; ++v) {
        std::set<int> targets;
        // Sample m unique targets. With initial >= m+1, the endpoint pool is
        // already larger than m, so termination is essentially immediate.
        int safety_cap = std::max(1000, m * 100);
        int attempts = 0;
        while ((int)targets.size() < m && attempts < safety_cap) {
            std::uniform_int_distribution<size_t> dist(0, endpoints.size() - 1);
            int u = endpoints[dist(rng)];
            if (u != v) targets.insert(u);
            attempts++;
        }
        for (int u : targets) {
            gs.edges.insert({std::min(u, v), std::max(u, v)});
            endpoints.push_back(u);
            endpoints.push_back(v);
        }
    }

    gs.extra.push_back({"ba_m", std::to_string(m)});
    return gs;
}

// ============================================================================
// Generator: motif planting with hub overlap.
// A pool of n_hubs "hub" vertices is allocated up front. Each planted motif
// places its vertex 0 onto a randomly chosen hub; the remaining motif
// vertices get fresh IDs. Result: hubs accumulate many incident motifs and
// motifs share structure, producing a graph where embeddings are non-trivial.
// ============================================================================
static GraphSpec generate_overlap(unsigned seed, int n_components, int n_hubs) {
    GraphSpec gs;
    if (n_hubs < 1) n_hubs = 1;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pat_dist(0, (int)PATTERNS.size() - 1);
    std::uniform_int_distribution<int> hub_dist(0, n_hubs - 1);

    int next_v = n_hubs;
    std::map<std::string, int> planted;

    for (int i = 0; i < n_components; ++i) {
        const Pattern& p = PATTERNS[pat_dist(rng)];
        std::vector<int> mapping(p.n);
        mapping[0] = hub_dist(rng);
        for (int j = 1; j < p.n; ++j) mapping[j] = next_v++;

        for (const auto& e : normalize_edges(p.edges)) {
            int u = mapping[e.first];
            int v = mapping[e.second];
            if (u != v) {
                gs.edges.insert({std::min(u, v), std::max(u, v)});
            }
        }
        planted[p.name] += 1;
    }

    gs.num_vertices = next_v;
    gs.extra.push_back({"total_components", std::to_string(n_components)});
    gs.extra.push_back({"num_hubs", std::to_string(n_hubs)});
    for (const auto& kv : planted) {
        gs.extra.push_back({"planted_components",
                            kv.first + " " + std::to_string(kv.second)});
    }
    return gs;
}

// ============================================================================
// Generator: convert an existing undirected edge list into a dataset.
// Useful for SNAP graphs (e.g. ego-Facebook, com-DBLP) downloaded externally.
// Self-loops are dropped; duplicates collapsed; num_vertices is inferred.
// ============================================================================
static GraphSpec convert_existing(const std::string& edge_list_path) {
    GraphSpec gs;
    std::ifstream f(edge_list_path);
    if (!f) {
        std::cerr << "ERROR: could not open input edge list "
                  << edge_list_path << "\n";
        std::exit(EXIT_FAILURE);
    }
    int max_v = -1;
    std::string line;
    while (std::getline(f, line)) {
        // Skip blank lines and comments.
        size_t p = line.find_first_not_of(" \t");
        if (p == std::string::npos) continue;
        if (line[p] == '#' || line[p] == '%') continue;
        std::istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;
        if (u == v) continue;
        gs.edges.insert({std::min(u, v), std::max(u, v)});
        max_v = std::max(max_v, std::max(u, v));
    }
    gs.num_vertices = max_v + 1;
    gs.extra.push_back({"converted_from", edge_list_path});
    return gs;
}

// ============================================================================
// Suite generator
// ============================================================================
static void make_run_script(const std::string& base_dir,
                            const std::vector<std::string>& subdirs) {
    std::string path = base_dir + "/run_suite.sh";
    std::ofstream f(path);
    if (!f) {
        std::cerr << "ERROR: could not write " << path << "\n";
        std::exit(EXIT_FAILURE);
    }
    f << "#!/usr/bin/env bash\n";
    f << "# Run the subiso solver against every dataset in this suite.\n";
    f << "# Usage: ./run_suite.sh [path-to-Subgraph_Solution-binary]\n";
    f << "set -e\n";
    f << "BIN=\"${1:-./Subgraph_Solution}\"\n";
    f << "if [ ! -x \"$BIN\" ]; then\n";
    f << "    echo \"ERROR: solver binary not found at $BIN\" >&2\n";
    f << "    echo \"Pass a path as the first argument.\" >&2\n";
    f << "    exit 1\n";
    f << "fi\n";
    f << "HERE=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n";
    f << "BIN_ABS=\"$(cd \"$(dirname \"$BIN\")\" && pwd)/$(basename \"$BIN\")\"\n";
    f << "for d in";
    for (const auto& s : subdirs) f << " " << s;
    f << "; do\n";
    f << "    echo\n";
    f << "    echo \"========== $d ==========\"\n";
    f << "    (cd \"$HERE/$d\" && \"$BIN_ABS\" data_edges.txt metadata.txt queries.txt)\n";
    f << "done\n";
    f.close();
    chmod(path.c_str(), 0755);
}

static void generate_suite(unsigned seed_base, const std::string& base_dir) {
    ensure_directory(base_dir);

    std::vector<std::string> subdirs;

    auto run = [&](const std::string& sub, const std::string& mode,
                   GraphSpec gs, unsigned seed) {
        std::string out = base_dir + "/" + sub;
        write_dataset(out, gs, seed, mode);
        subdirs.push_back(sub);
    };

    // Small motif graph: sanity check, fastest, used to validate correctness.
    run("01_motif_n300", "motif",
        generate_motif(seed_base + 0, 300), seed_base + 0);

    // Erdős–Rényi: three sizes at moderate density. These exercise the kernel
    // because non-trivial connectivity gives larger frontiers.
    run("02_er_n1k_d6",  "er",
        generate_er(seed_base + 1, 1000, 6.0),  seed_base + 1);
    run("03_er_n2k_d8",  "er",
        generate_er(seed_base + 2, 2000, 8.0),  seed_base + 2);
    run("04_er_n5k_d10", "er",
        generate_er(seed_base + 3, 5000, 10.0), seed_base + 3);

    // Barabási–Albert: power-law degree distribution, hubs stress the
    // candidate sets and create skewed work distribution per frontier slot.
    run("05_ba_n2k_m3", "ba",
        generate_ba(seed_base + 4, 2000, 3), seed_base + 4);
    run("06_ba_n5k_m4", "ba",
        generate_ba(seed_base + 5, 5000, 4), seed_base + 5);

    // Hub-overlap motif planting: structured but with shared vertices.
    run("07_overlap_n300_h20", "overlap",
        generate_overlap(seed_base + 6, 300, 20), seed_base + 6);

    make_run_script(base_dir, subdirs);

    std::cout << "\nSuite written to " << base_dir << "/\n";
    std::cout << "Run all datasets with:\n";
    std::cout << "    bash " << base_dir
              << "/run_suite.sh /path/to/Subgraph_Solution\n";
}

// ============================================================================
// CLI
// ============================================================================
static void print_help(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " [mode] [args...]\n"
        "\n"
        "Modes:\n"
        "  motif    [seed=42] [n_components=300] [out_dir=.]\n"
        "           Plant random motif components (original behavior).\n"
        "\n"
        "  er       <seed> <n_vertices> <avg_degree> [out_dir=.]\n"
        "           Erdős-Rényi G(n, p) with p = avg_degree / (n-1).\n"
        "           Uses Batagelj-Brandes edge-skip; O(m) expected time.\n"
        "\n"
        "  ba       <seed> <n_vertices> <m> [out_dir=.]\n"
        "           Barabási-Albert preferential attachment, m edges per\n"
        "           new vertex. Produces power-law degree distributions.\n"
        "\n"
        "  overlap  <seed> <n_components> <n_hubs> [out_dir=.]\n"
        "           Motif planting where each motif's vertex 0 maps onto\n"
        "           a randomly selected shared hub vertex.\n"
        "\n"
        "  convert  <input_edge_list> [out_dir=.]\n"
        "           Convert an existing edge list (e.g. SNAP graph) into\n"
        "           a dataset directory. Comments (# or %) and blank lines\n"
        "           are skipped; vertex count is inferred from max ID.\n"
        "\n"
        "  suite    [seed=42] [base_dir=suite]\n"
        "           Generate a 7-dataset evaluation suite plus a\n"
        "           run_suite.sh script.\n"
        "\n"
        "  help     Print this message.\n"
        "\n"
        "Backward compatibility: when the first argument is numeric or absent,\n"
        "motif mode is invoked (matching the original generator).\n"
        "\n"
        "Examples:\n"
        "  " << prog << "                              # motif, seed=42, 300 components\n"
        "  " << prog << " 42 500                       # motif, seed=42, 500 components\n"
        "  " << prog << " er 42 5000 10 ./er_5k        # ER graph, n=5k, avg_deg=10\n"
        "  " << prog << " ba 42 2000 3 ./ba_2k         # BA graph, n=2k, m=3\n"
        "  " << prog << " overlap 42 500 25 ./overlap  # motifs sharing 25 hubs\n"
        "  " << prog << " convert ./snap_facebook.txt  # use a downloaded SNAP graph\n"
        "  " << prog << " suite                        # full evaluation suite\n";
}

static bool is_numeric_arg(const std::string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i++;
    if (i >= s.size()) return false;
    bool any_digit = false;
    for (; i < s.size(); ++i) {
        if (std::isdigit((unsigned char)s[i])) { any_digit = true; }
        else if (s[i] == '.') { /* ok in floats */ }
        else { return false; }
    }
    return any_digit;
}

static unsigned parse_seed(const char* s) {
    return (unsigned)std::stoul(s);
}

int main(int argc, char** argv) {
    // No args: backward-compat default = motif mode.
    if (argc < 2) {
        auto gs = generate_motif(42, 300);
        write_dataset(".", gs, 42, "motif");
        return 0;
    }

    std::string a1 = argv[1];

    // Backward compat: numeric first arg = old-style positional motif call.
    if (is_numeric_arg(a1)) {
        unsigned seed = parse_seed(argv[1]);
        int n        = (argc >= 3) ? std::stoi(argv[2]) : 300;
        std::string out = (argc >= 4) ? argv[3] : ".";
        auto gs = generate_motif(seed, n);
        write_dataset(out, gs, seed, "motif");
        return 0;
    }

    if (a1 == "help" || a1 == "--help" || a1 == "-h") {
        print_help(argv[0]);
        return 0;
    }

    if (a1 == "motif") {
        unsigned seed = (argc >= 3) ? parse_seed(argv[2]) : 42;
        int n         = (argc >= 4) ? std::stoi(argv[3]) : 300;
        std::string out = (argc >= 5) ? argv[4] : ".";
        auto gs = generate_motif(seed, n);
        write_dataset(out, gs, seed, "motif");
        return 0;
    }

    if (a1 == "er") {
        if (argc < 5) {
            std::cerr << "er mode needs: <seed> <n_vertices> <avg_degree> [out_dir]\n";
            return 1;
        }
        unsigned seed = parse_seed(argv[2]);
        int n         = std::stoi(argv[3]);
        double d      = std::stod(argv[4]);
        std::string out = (argc >= 6) ? argv[5] : ".";
        auto gs = generate_er(seed, n, d);
        write_dataset(out, gs, seed, "er");
        return 0;
    }

    if (a1 == "ba") {
        if (argc < 5) {
            std::cerr << "ba mode needs: <seed> <n_vertices> <m> [out_dir]\n";
            return 1;
        }
        unsigned seed = parse_seed(argv[2]);
        int n         = std::stoi(argv[3]);
        int m         = std::stoi(argv[4]);
        std::string out = (argc >= 6) ? argv[5] : ".";
        auto gs = generate_ba(seed, n, m);
        write_dataset(out, gs, seed, "ba");
        return 0;
    }

    if (a1 == "overlap") {
        if (argc < 5) {
            std::cerr << "overlap mode needs: <seed> <n_components> <n_hubs> [out_dir]\n";
            return 1;
        }
        unsigned seed = parse_seed(argv[2]);
        int nc        = std::stoi(argv[3]);
        int hubs      = std::stoi(argv[4]);
        std::string out = (argc >= 6) ? argv[5] : ".";
        auto gs = generate_overlap(seed, nc, hubs);
        write_dataset(out, gs, seed, "overlap");
        return 0;
    }

    if (a1 == "convert") {
        if (argc < 3) {
            std::cerr << "convert mode needs: <input_edge_list> [out_dir]\n";
            return 1;
        }
        std::string in_path = argv[2];
        std::string out     = (argc >= 4) ? argv[3] : ".";
        auto gs = convert_existing(in_path);
        write_dataset(out, gs, 0, "convert");
        return 0;
    }

    if (a1 == "suite") {
        unsigned seed = (argc >= 3) ? parse_seed(argv[2]) : 42;
        std::string base = (argc >= 4) ? argv[3] : "suite";
        generate_suite(seed, base);
        return 0;
    }

    std::cerr << "Unknown mode: '" << a1 << "'\n\n";
    print_help(argv[0]);
    return 1;
}
