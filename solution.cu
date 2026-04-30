// cuda_levelwise_subgraph_expansion.cu
// Level-wise CUDA frontier expansion for subgraph matching.
//
// This implements one CUDA kernel per pattern depth. Each thread processes
// one (partial mapping, candidate data vertex) extension attempt.
//
// Assumptions:
//   - Pattern vertex order is fixed as 0, 1, ..., Pn-1.
//   - Candidate sets are stored flat by pattern vertex.
//   - Graphs are stored in CSR format.
//   - This version counts/enumerates embeddings through frontiers.
//   - MAX_PATTERN_VERTICES should be set large enough for your pattern size.
//
// Compile example:
//   nvcc -O3 -std=c++17 cuda_levelwise_subgraph_expansion.cu -o subiso_cuda

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <fstream>
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#ifndef MAX_PATTERN_VERTICES
#define MAX_PATTERN_VERTICES 16
#endif

// A partial embedding maps pattern vertices [0, depth) to data vertices.
struct PartialMapping {
    int depth;
    int map[MAX_PATTERN_VERTICES];
};

// Binary search adjacency test in CSR row.
// This assumes CSR column indices for each row are sorted.
__device__ bool csr_has_edge(
    int u,
    int v,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx
) {
    int start = row_ptr[u];
    int end   = row_ptr[u + 1];

    int lo = start;
    int hi = end - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        int val = col_idx[mid];
        if (val == v) return true;
        if (val < v) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}

__device__ bool is_injective_valid(
    const PartialMapping& pm,
    int candidate_data_vertex
) {
    for (int i = 0; i < pm.depth; ++i) {
        if (pm.map[i] == candidate_data_vertex) return false;
    }
    return true;
}

// Checks whether adding pattern vertex `pattern_v` -> `candidate_data_v`
// is edge-consistent with all already-mapped pattern vertices.
//
// For each previous pattern vertex i:
//   if pattern has edge (pattern_v, i), data must have edge (candidate_data_v, pm.map[i])
//   if pattern has edge (i, pattern_v), data must have edge (pm.map[i], candidate_data_v)
//
// For undirected graphs stored symmetrically, these two checks are redundant but harmless.
__device__ bool is_edge_consistent(
    const PartialMapping& pm,
    int pattern_v,
    int candidate_data_v,
    const int* __restrict__ p_row_ptr,
    const int* __restrict__ p_col_idx,
    const int* __restrict__ g_row_ptr,
    const int* __restrict__ g_col_idx
) {
    for (int prev_p = 0; prev_p < pm.depth; ++prev_p) {
        int prev_g = pm.map[prev_p];

        bool p_forward = csr_has_edge(pattern_v, prev_p, p_row_ptr, p_col_idx);
        if (p_forward) {
            bool g_forward = csr_has_edge(candidate_data_v, prev_g, g_row_ptr, g_col_idx);
            if (!g_forward) return false;
        }

        bool p_backward = csr_has_edge(prev_p, pattern_v, p_row_ptr, p_col_idx);
        if (p_backward) {
            bool g_backward = csr_has_edge(prev_g, candidate_data_v, g_row_ptr, g_col_idx);
            if (!g_backward) return false;
        }
    }
    return true;
}

// Expand one frontier level.
//
// Work decomposition:
//   total_attempts = frontier_size * num_candidates_for_current_pattern_vertex
//   global thread id indexes one extension attempt.
//
// Input frontier contains mappings of depth `depth`.
// Output frontier contains mappings of depth `depth + 1`.
__global__ void expand_frontier_kernel(
    const PartialMapping* __restrict__ in_frontier,
    int frontier_size,
    PartialMapping* __restrict__ out_frontier,
    int* __restrict__ out_count,
    int out_capacity,
    int depth,
    const int* __restrict__ candidate_offsets,
    const int* __restrict__ candidate_counts,
    const int* __restrict__ candidate_flat,
    const int* __restrict__ p_row_ptr,
    const int* __restrict__ p_col_idx,
    const int* __restrict__ g_row_ptr,
    const int* __restrict__ g_col_idx
) {
    int pattern_v = depth;
    int cand_offset = candidate_offsets[pattern_v];
    int cand_count  = candidate_counts[pattern_v];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_attempts = frontier_size * cand_count;

    if (tid >= total_attempts) return;

    int mapping_idx = tid / cand_count;
    int cand_idx    = tid % cand_count;
    int candidate_g = candidate_flat[cand_offset + cand_idx];

    PartialMapping pm = in_frontier[mapping_idx];

    if (!is_injective_valid(pm, candidate_g)) return;

    if (!is_edge_consistent(
            pm,
            pattern_v,
            candidate_g,
            p_row_ptr,
            p_col_idx,
            g_row_ptr,
            g_col_idx)) {
        return;
    }

    int write_idx = atomicAdd(out_count, 1);
    if (write_idx >= out_capacity) {
        // Overflow. We incremented out_count anyway, so the host can detect overflow
        // by checking whether out_count > out_capacity after the kernel.
        return;
    }

    PartialMapping next = pm;
    next.map[depth] = candidate_g;
    next.depth = depth + 1;
    out_frontier[write_idx] = next;
}

// Utility: build CSR from edge list.
// If undirected=true, edges are inserted both ways.
struct CSRGraph {
    int n;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
};

CSRGraph build_csr(int n, const std::vector<std::pair<int, int>>& edges, bool undirected) {
    std::vector<std::vector<int>> adj(n);
    for (const auto& edge : edges) {
        int u = edge.first;
        int v = edge.second;

        adj[u].push_back(v);
        if (undirected && u != v) {
            adj[v].push_back(u);
        }
    }
    CSRGraph csr;
    csr.n = n;
    csr.row_ptr.resize(n + 1, 0);

    for (int u = 0; u < n; ++u) {
        std::sort(adj[u].begin(), adj[u].end());
        adj[u].erase(std::unique(adj[u].begin(), adj[u].end()), adj[u].end());
        csr.row_ptr[u + 1] = csr.row_ptr[u] + static_cast<int>(adj[u].size());
        for (int v : adj[u]) csr.col_idx.push_back(v);
    }
    return csr;
}

// Simple degree-based candidate sets: data vertex degree >= pattern vertex degree.
// Replace this with your existing candidate filtering if you already have it.
void build_degree_candidates(
    const CSRGraph& pattern,
    const CSRGraph& data,
    std::vector<int>& candidate_offsets,
    std::vector<int>& candidate_counts,
    std::vector<int>& candidate_flat
) {
    int Pn = pattern.n;
    candidate_offsets.assign(Pn, 0);
    candidate_counts.assign(Pn, 0);
    candidate_flat.clear();

    for (int p = 0; p < Pn; ++p) {
        candidate_offsets[p] = static_cast<int>(candidate_flat.size());
        int p_deg = pattern.row_ptr[p + 1] - pattern.row_ptr[p];

        for (int g = 0; g < data.n; ++g) {
            int g_deg = data.row_ptr[g + 1] - data.row_ptr[g];
            if (g_deg >= p_deg) {
                candidate_flat.push_back(g);
            }
        }
        candidate_counts[p] = static_cast<int>(candidate_flat.size()) - candidate_offsets[p];
    }
}

long long run_cuda_levelwise_subiso(
    const CSRGraph& pattern,
    const CSRGraph& data,
    const std::vector<int>& h_candidate_offsets,
    const std::vector<int>& h_candidate_counts,
    const std::vector<int>& h_candidate_flat,
    int frontier_capacity
) {
    int Pn = pattern.n;
    if (Pn > MAX_PATTERN_VERTICES) {
        std::cerr << "Pattern size exceeds MAX_PATTERN_VERTICES=" << MAX_PATTERN_VERTICES << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int *d_p_row = nullptr, *d_p_col = nullptr;
    int *d_g_row = nullptr, *d_g_col = nullptr;
    int *d_candidate_offsets = nullptr, *d_candidate_counts = nullptr, *d_candidate_flat = nullptr;

    CUDA_CHECK(cudaMalloc(&d_p_row, pattern.row_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_p_col, pattern.col_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_g_row, data.row_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_g_col, data.col_idx.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_p_row, pattern.row_ptr.data(), pattern.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_col, pattern.col_idx.data(), pattern.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_row, data.row_ptr.data(), data.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_col, data.col_idx.data(), data.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_candidate_offsets, h_candidate_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_counts,  h_candidate_counts.size()  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_flat,    h_candidate_flat.size()    * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_candidate_offsets, h_candidate_offsets.data(), h_candidate_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidate_counts,  h_candidate_counts.data(),  h_candidate_counts.size()  * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidate_flat,    h_candidate_flat.data(),    h_candidate_flat.size()    * sizeof(int), cudaMemcpyHostToDevice));

    PartialMapping* d_frontier_a = nullptr;
    PartialMapping* d_frontier_b = nullptr;
    int* d_out_count = nullptr;

    CUDA_CHECK(cudaMalloc(&d_frontier_a, frontier_capacity * sizeof(PartialMapping)));
    CUDA_CHECK(cudaMalloc(&d_frontier_b, frontier_capacity * sizeof(PartialMapping)));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));

    // Initialize frontier with one empty mapping.
    PartialMapping h_root{};
    h_root.depth = 0;
    for (int i = 0; i < MAX_PATTERN_VERTICES; ++i) h_root.map[i] = -1;
    CUDA_CHECK(cudaMemcpy(d_frontier_a, &h_root, sizeof(PartialMapping), cudaMemcpyHostToDevice));

    int frontier_size = 1;

    for (int depth = 0; depth < Pn; ++depth) {
        CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

        int cand_count = h_candidate_counts[depth];
        int total_attempts = frontier_size * cand_count;

        if (total_attempts == 0) {
            frontier_size = 0;
            break;
        }

        int block_size = 256;
        int grid_size = (total_attempts + block_size - 1) / block_size;

        expand_frontier_kernel<<<grid_size, block_size>>>(
            d_frontier_a,
            frontier_size,
            d_frontier_b,
            d_out_count,
            frontier_capacity,
            depth,
            d_candidate_offsets,
            d_candidate_counts,
            d_candidate_flat,
            d_p_row,
            d_p_col,
            d_g_row,
            d_g_col
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_out_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_out_count > frontier_capacity) {
            std::cerr << "ERROR: frontier overflow at depth " << depth
                      << ". Needed " << h_out_count
                      << ", capacity " << frontier_capacity << ".\n";
            std::cerr << "Increase frontier_capacity or add chunked expansion.\n";
            std::exit(EXIT_FAILURE);
        }

        frontier_size = h_out_count;
        std::cout << "Depth " << depth + 1 << " frontier size: " << frontier_size << std::endl;

        std::swap(d_frontier_a, d_frontier_b);
    }

    long long match_count = frontier_size;

    CUDA_CHECK(cudaFree(d_p_row));
    CUDA_CHECK(cudaFree(d_p_col));
    CUDA_CHECK(cudaFree(d_g_row));
    CUDA_CHECK(cudaFree(d_g_col));
    CUDA_CHECK(cudaFree(d_candidate_offsets));
    CUDA_CHECK(cudaFree(d_candidate_counts));
    CUDA_CHECK(cudaFree(d_candidate_flat));
    CUDA_CHECK(cudaFree(d_frontier_a));
    CUDA_CHECK(cudaFree(d_frontier_b));
    CUDA_CHECK(cudaFree(d_out_count));

    return match_count;
}

bool cpu_csr_has_edge(
    int u,
    int v,
    const CSRGraph& graph
) {
    int start = graph.row_ptr[u];
    int end = graph.row_ptr[u + 1];

    return std::binary_search(
        graph.col_idx.begin() + start,
        graph.col_idx.begin() + end,
        v
    );
}

bool cpu_is_valid_extension(
    const std::vector<int>& mapping,
    int depth,
    int pattern_v,
    int candidate_g,
    const CSRGraph& pattern,
    const CSRGraph& data
) {
    // Injectivity: do not reuse data vertices.
    for (int i = 0; i < depth; ++i) {
        if (mapping[i] == candidate_g) {
            return false;
        }
    }

    // Edge consistency with previously mapped pattern vertices.
    for (int prev_p = 0; prev_p < depth; ++prev_p) {
        int prev_g = mapping[prev_p];

        if (cpu_csr_has_edge(pattern_v, prev_p, pattern)) {
            if (!cpu_csr_has_edge(candidate_g, prev_g, data)) {
                return false;
            }
        }

        if (cpu_csr_has_edge(prev_p, pattern_v, pattern)) {
            if (!cpu_csr_has_edge(prev_g, candidate_g, data)) {
                return false;
            }
        }
    }

    return true;
}

long long run_cpu_levelwise_subiso(
    const CSRGraph& pattern,
    const CSRGraph& data,
    const std::vector<int>& candidate_offsets,
    const std::vector<int>& candidate_counts,
    const std::vector<int>& candidate_flat
) {
    int Pn = pattern.n;

    std::vector<std::vector<int>> frontier;
    frontier.push_back(std::vector<int>(Pn, -1));

    for (int depth = 0; depth < Pn; ++depth) {
        int pattern_v = depth;

        std::vector<std::vector<int>> next_frontier;

        int offset = candidate_offsets[pattern_v];
        int count = candidate_counts[pattern_v];

        for (const auto& mapping : frontier) {
            for (int c = 0; c < count; ++c) {
                int candidate_g = candidate_flat[offset + c];

                if (!cpu_is_valid_extension(
                        mapping,
                        depth,
                        pattern_v,
                        candidate_g,
                        pattern,
                        data
                    )) {
                    continue;
                }

                std::vector<int> next_mapping = mapping;
                next_mapping[pattern_v] = candidate_g;
                next_frontier.push_back(next_mapping);
            }
        }

        std::cout << "[CPU] Depth " << depth + 1
                  << " frontier size: " << next_frontier.size()
                  << std::endl;

        frontier = std::move(next_frontier);
    }

    return static_cast<long long>(frontier.size());
}

std::vector<std::pair<int, int>> load_edge_list(
    const std::string& filename
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: could not open edge list file: "
                  << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::pair<int, int>> edges;
    int u, v;

    while (file >> u >> v) {
        edges.push_back({u, v});
    }

    return edges;
}

int load_metadata_int(
    const std::string& filename,
    const std::string& key
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: could not open metadata file: "
                  << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string current_key;
    int value;

    while (file >> current_key >> value) {
        if (current_key == key) {
            return value;
        }
    }

    std::cerr << "ERROR: key not found in metadata: "
              << key << std::endl;
    std::exit(EXIT_FAILURE);
}

struct QueryInfo {
    std::string name;
    std::string pattern_file;
    int pattern_vertices;
};

std::vector<QueryInfo> load_queries(
    const std::string& filename
) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "ERROR: could not open queries file: "
                  << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<QueryInfo> queries;
    std::string name;
    std::string pattern_file;
    int pattern_vertices;

    while (file >> name >> pattern_file >> pattern_vertices) {
        queries.push_back({name, pattern_file, pattern_vertices});
    }

    return queries;
}

int main() {
    std::string data_file = "data_edges1.txt";
    std::string metadata_file = "metadata1.txt";
    std::string queries_file = "queries1.txt";

    std::vector<std::pair<int, int>> data_edges =
        load_edge_list(data_file);

    int num_vertices =
        load_metadata_int(metadata_file, "num_vertices");

    bool undirected = true;

    CSRGraph G = build_csr(num_vertices, data_edges, undirected);

    std::vector<QueryInfo> queries = load_queries(queries_file);

    int frontier_capacity = 1 << 20;

    for (const QueryInfo& query : queries) {
        std::cout << "\n=== Query: " << query.name << " ===" << std::endl;

        std::vector<std::pair<int, int>> pattern_edges =
            load_edge_list(query.pattern_file);

        CSRGraph P = build_csr(
            query.pattern_vertices,
            pattern_edges,
            undirected
        );

        std::vector<int> candidate_offsets;
        std::vector<int> candidate_counts;
        std::vector<int> candidate_flat;

        build_degree_candidates(
            P,
            G,
            candidate_offsets,
            candidate_counts,
            candidate_flat
        );

        auto cpu_start = std::chrono::high_resolution_clock::now();

        long long cpu_matches = run_cpu_levelwise_subiso(
            P,
            G,
            candidate_offsets,
            candidate_counts,
            candidate_flat
        );

        auto cpu_end = std::chrono::high_resolution_clock::now();

        double cpu_ms =
            std::chrono::duration<double, std::milli>(
                cpu_end - cpu_start
            ).count();

        auto gpu_start = std::chrono::high_resolution_clock::now();

        long long gpu_matches = run_cuda_levelwise_subiso(
            P,
            G,
            candidate_offsets,
            candidate_counts,
            candidate_flat,
            frontier_capacity
        );

        auto gpu_end = std::chrono::high_resolution_clock::now();

        double gpu_ms =
            std::chrono::duration<double, std::milli>(
                gpu_end - gpu_start
            ).count();

        std::cout << "CPU matches: " << cpu_matches << std::endl;
        std::cout << "GPU matches: " << gpu_matches << std::endl;

        std::cout << "CPU time: " << cpu_ms << " ms" << std::endl;
        std::cout << "GPU time: " << gpu_ms << " ms" << std::endl;

        if (gpu_ms > 0.0) {
            std::cout << "Speedup: " << cpu_ms / gpu_ms << "x" << std::endl;
        }

        if (cpu_matches == gpu_matches) {
            std::cout << "Correctness: PASS" << std::endl;
        } else {
            std::cout << "Correctness: FAIL" << std::endl;
        }
    }

    return 0;
}
