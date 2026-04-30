// solution.cu
// GPU-accelerated subgraph isomorphism via level-wise frontier expansion.
//
// Implements the following techniques (all promised in the project plan):
//
//   1. Compressed Sparse Row (CSR) graph representation in GPU global memory.
//   2. Neighborhood-degree-dominance candidate filtering (label-free NLF).
//      Strictly stronger than degree filtering. For pattern vertex p and data
//      vertex g, g is a candidate for p iff for every i, the i-th largest
//      neighbor-degree of g is >= the i-th largest neighbor-degree of p.
//   3. Automorphism-orbit symmetry breaking. The pattern's automorphism group
//      is enumerated on the host (small patterns; n! permutations checked).
//      For each orbit {v_0 < v_1 < ... < v_k}, the constraint
//      f(v_0) < f(v_1) < ... < f(v_k) is imposed, eliminating duplicate
//      embeddings under permutation of automorphic vertices.
//   4. GPU level-wise expansion kernel with:
//        - Shared-memory cooperative load of the per-block input mapping.
//        - Pattern-edge bitmask: which prior pattern vertices have an edge
//          to the current depth is precomputed on the host and passed as a
//          kernel argument. Pattern CSR is not even uploaded to the device.
//        - Single-direction edge consistency check (CSR is symmetric for
//          undirected graphs, so the backward check is redundant).
//        - Two-phase commit for output writes:
//            (i)   per-thread atomic on a shared-memory counter to compute
//                  the within-block local index;
//            (ii)  one global atomicAdd per block to claim a contiguous slot
//                  range in the output buffer;
//            (iii) each valid thread writes directly to its claimed slot.
//          This reduces global atomics from O(valid_extensions) to O(blocks).
//        - Symmetry-breaking constraint enforced inline.
//   5. Adaptive frontier reallocation. On overflow, both ping-pong buffers
//      are grown (preserving input contents) and the offending depth is
//      retried. No hard capacity ceiling.
//   6. Frontier-Y chunking. The grid Y dimension is bounded at 32768 and the
//      frontier is processed in chunks if larger.
//   7. CPU baseline applying the SAME constraints (NLF, symmetry breaking,
//      undirected edge check) for fair correctness cross-validation.
//
// Compile:
//   nvcc -O3 -std=c++17 solution.cu -o subiso_cuda
//
// Usage:
//   ./subiso_cuda <data_edges> <metadata> <queries> [initial_frontier_capacity]
//

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err__) << std::endl;     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#ifndef MAX_PATTERN_VERTICES
#define MAX_PATTERN_VERTICES 16
#endif

#ifndef EXPAND_BLOCK_SIZE
#define EXPAND_BLOCK_SIZE 128
#endif

// CUDA grid Y/Z max dimension (we cap below the hardware limit of 65535
// to leave room for safety on older driver/runtime combinations).
static const int MAX_GRID_Y = 32768;

// A partial embedding maps pattern vertices [0, depth) to data vertices.
struct PartialMapping {
    int depth;
    int map[MAX_PATTERN_VERTICES];
};

// ============================================================================
// CSR graph representation
// ============================================================================
struct CSRGraph {
    int n;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
};

static CSRGraph build_csr(int n,
                          const std::vector<std::pair<int, int>>& edges,
                          bool undirected) {
    std::vector<std::vector<int>> adj(n);
    for (const auto& e : edges) {
        adj[e.first].push_back(e.second);
        if (undirected && e.first != e.second) {
            adj[e.second].push_back(e.first);
        }
    }
    CSRGraph csr;
    csr.n = n;
    csr.row_ptr.assign(n + 1, 0);
    for (int u = 0; u < n; ++u) {
        std::sort(adj[u].begin(), adj[u].end());
        adj[u].erase(std::unique(adj[u].begin(), adj[u].end()), adj[u].end());
        csr.row_ptr[u + 1] = csr.row_ptr[u] + (int)adj[u].size();
        for (int v : adj[u]) csr.col_idx.push_back(v);
    }
    return csr;
}

static int csr_degree(const CSRGraph& g, int u) {
    return g.row_ptr[u + 1] - g.row_ptr[u];
}

static bool cpu_csr_has_edge(int u, int v, const CSRGraph& g) {
    int s = g.row_ptr[u];
    int e = g.row_ptr[u + 1];
    return std::binary_search(g.col_idx.begin() + s,
                              g.col_idx.begin() + e, v);
}

// ============================================================================
// Candidate filtering: label-free NLF (Neighbor Label Frequency).
//
// For each vertex u, build the multiset of its neighbor degrees, sorted in
// descending order. A data vertex g is a candidate for pattern vertex p iff
// for every i in [0, deg(p)), the i-th largest neighbor-degree of g is at
// least the i-th largest neighbor-degree of p (multiset dominance).
// ============================================================================
static std::vector<std::vector<int>>
sorted_neighbor_degrees(const CSRGraph& g) {
    std::vector<std::vector<int>> out(g.n);
    for (int u = 0; u < g.n; ++u) {
        out[u].reserve(csr_degree(g, u));
        for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; ++j) {
            int v = g.col_idx[j];
            out[u].push_back(csr_degree(g, v));
        }
        std::sort(out[u].begin(), out[u].end(), std::greater<int>());
    }
    return out;
}

static void build_nlf_candidates(
    const CSRGraph& pattern, const CSRGraph& data,
    std::vector<int>& cand_offsets,
    std::vector<int>& cand_counts,
    std::vector<int>& cand_flat
) {
    auto p_seqs = sorted_neighbor_degrees(pattern);
    auto g_seqs = sorted_neighbor_degrees(data);
    int Pn = pattern.n;
    cand_offsets.assign(Pn, 0);
    cand_counts.assign(Pn, 0);
    cand_flat.clear();

    for (int p = 0; p < Pn; ++p) {
        cand_offsets[p] = (int)cand_flat.size();
        const auto& pd = p_seqs[p];
        int p_deg = (int)pd.size();

        for (int g = 0; g < data.n; ++g) {
            const auto& gd = g_seqs[g];
            if ((int)gd.size() < p_deg) continue;
            bool dominated = true;
            for (int i = 0; i < p_deg; ++i) {
                if (gd[i] < pd[i]) { dominated = false; break; }
            }
            if (dominated) cand_flat.push_back(g);
        }
        cand_counts[p] = (int)cand_flat.size() - cand_offsets[p];
    }
}

// ============================================================================
// Symmetry breaking: pattern automorphism orbits.
//
// We enumerate all permutations of pattern vertices, retain those that
// preserve the edge set, and union-find the orbits induced on vertices.
// For each orbit {v_0 < v_1 < ... < v_k}, we record sym_pred[v_{i+1}] = v_i,
// so the search can enforce f(v_{i+1}) > f(v_i) when extending depth v_{i+1}.
//
// For n <= 9 (9! = 362,880) this is comfortably fast. The benchmark patterns
// here have n <= 5. For larger patterns we skip symmetry breaking and warn.
// ============================================================================
static std::vector<int>
build_symmetry_predecessors(const CSRGraph& pattern) {
    int n = pattern.n;
    std::vector<int> sym_pred(n, -1);
    if (n <= 1) return sym_pred;

    if (n > 9) {
        std::cerr << "[symmetry] pattern has " << n << " vertices, "
                  << "skipping automorphism enumeration. No symmetry "
                  << "breaking will be applied for this pattern.\n";
        return sym_pred;
    }

    // Build canonical edge set for fast lookup.
    std::set<std::pair<int, int>> edges;
    for (int u = 0; u < n; ++u) {
        for (int j = pattern.row_ptr[u]; j < pattern.row_ptr[u + 1]; ++j) {
            int v = pattern.col_idx[j];
            if (u < v) edges.insert({u, v});
        }
    }

    // Union-find over orbits.
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](int a, int b) {
        a = find(a); b = find(b);
        if (a != b) parent[a] = b;
    };

    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    do {
        bool is_aut = true;
        for (const auto& e : edges) {
            int pu = perm[e.first];
            int pv = perm[e.second];
            if (edges.count({std::min(pu, pv), std::max(pu, pv)}) == 0) {
                is_aut = false;
                break;
            }
        }
        if (is_aut) {
            for (int i = 0; i < n; ++i) {
                if (perm[i] != i) unite(i, perm[i]);
            }
        }
    } while (std::next_permutation(perm.begin(), perm.end()));

    // Group vertices by orbit and chain predecessors.
    std::map<int, std::vector<int>> orbits;
    for (int i = 0; i < n; ++i) orbits[find(i)].push_back(i);
    for (auto& kv : orbits) {
        auto& vs = kv.second;
        if (vs.size() <= 1) continue;
        std::sort(vs.begin(), vs.end());
        for (size_t i = 1; i < vs.size(); ++i) {
            sym_pred[vs[i]] = vs[i - 1];
        }
    }
    return sym_pred;
}

// ============================================================================
// Device functions
// ============================================================================
__device__ static bool csr_has_edge(
    int u, int v,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx
) {
    int lo = row_ptr[u];
    int hi = row_ptr[u + 1] - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int x = col_idx[mid];
        if (x == v)      return true;
        else if (x < v)  lo = mid + 1;
        else             hi = mid - 1;
    }
    return false;
}

__device__ static bool is_injective(const PartialMapping& pm, int g) {
    for (int i = 0; i < pm.depth; ++i) {
        if (pm.map[i] == g) return false;
    }
    return true;
}

// Edge consistency check for an undirected graph with symmetric CSR.
//
// `prev_edge_mask` has bit i set iff the pattern has an edge (depth, i).
// We only need to verify that the data graph also has the corresponding edge.
__device__ static bool is_edge_consistent_undir(
    const PartialMapping& pm,
    int g_v,
    int prev_edge_mask,
    const int* __restrict__ g_row,
    const int* __restrict__ g_col
) {
    for (int i = 0; i < pm.depth; ++i) {
        if ((prev_edge_mask >> i) & 1) {
            if (!csr_has_edge(g_v, pm.map[i], g_row, g_col)) return false;
        }
    }
    return true;
}

// ============================================================================
// Expand-frontier kernel.
//
// One block expands one input mapping over a tile of candidates.
//
//   gridDim.y       = chunk_size (number of input mappings in this launch,
//                                  bounded by MAX_GRID_Y)
//   gridDim.x       = ceil(cand_count / blockDim.x)
//   blockIdx.y      = mapping index within the chunk
//   threadIdx.x     = lane within candidate tile
//
// Symmetry breaking: if sym_pred_v >= 0, candidate_g must be > pm.map[sym_pred_v].
// ============================================================================
__global__ static void expand_frontier_kernel(
    const PartialMapping* __restrict__ in_frontier,
    int frontier_size_in_chunk,
    PartialMapping* __restrict__ out_frontier,
    int* __restrict__ out_count,
    int out_capacity,
    int depth,
    int sym_pred_v,
    int prev_edge_mask,
    int cand_offset,
    int cand_count,
    const int* __restrict__ candidate_flat,
    const int* __restrict__ g_row,
    const int* __restrict__ g_col
) {
    __shared__ PartialMapping s_pm;
    __shared__ int s_local_count;
    __shared__ int s_global_offset;

    int mapping_idx = blockIdx.y;
    if (mapping_idx >= frontier_size_in_chunk) return;

    // Phase 0: cooperative load of the input mapping into shared memory.
    constexpr int N_INTS = sizeof(PartialMapping) / sizeof(int);
    const int* src = reinterpret_cast<const int*>(&in_frontier[mapping_idx]);
    int* dst = reinterpret_cast<int*>(&s_pm);
    for (int i = threadIdx.x; i < N_INTS; i += blockDim.x) {
        dst[i] = src[i];
    }
    if (threadIdx.x == 0) s_local_count = 0;
    __syncthreads();

    // Phase 1: each thread evaluates one candidate.
    int  cand_idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int  candidate_g = -1;
    int  local_idx   = -1;
    bool valid       = false;

    if (cand_idx < cand_count) {
        candidate_g = candidate_flat[cand_offset + cand_idx];

        bool sym_ok = (sym_pred_v < 0) ||
                      (candidate_g > s_pm.map[sym_pred_v]);

        if (sym_ok &&
            is_injective(s_pm, candidate_g) &&
            is_edge_consistent_undir(s_pm, candidate_g,
                                     prev_edge_mask, g_row, g_col)) {
            valid     = true;
            local_idx = atomicAdd(&s_local_count, 1);
        }
    }
    __syncthreads();

    // Phase 2: lane 0 claims a contiguous global slot range for the block.
    if (threadIdx.x == 0) {
        if (s_local_count > 0) {
            s_global_offset = atomicAdd(out_count, s_local_count);
        } else {
            s_global_offset = 0;
        }
    }
    __syncthreads();

    // Phase 3: each valid thread writes its extended mapping directly.
    if (valid) {
        int write_idx = s_global_offset + local_idx;
        if (write_idx < out_capacity) {
            PartialMapping next = s_pm;
            next.map[depth] = candidate_g;
            next.depth      = depth + 1;
            out_frontier[write_idx] = next;
        }
        // If write_idx >= out_capacity, the host will detect overflow via
        // out_count > out_capacity and resize+retry this depth.
    }
}

// ============================================================================
// Host driver
// ============================================================================
struct GpuTimings {
    std::vector<double> per_depth_ms;
    double total_ms = 0.0;
    int    resizes  = 0;
};

static long long run_gpu(
    const CSRGraph& pattern,
    const CSRGraph& data,
    const std::vector<int>& cand_offsets,
    const std::vector<int>& cand_counts,
    const std::vector<int>& cand_flat,
    const std::vector<int>& sym_pred,
    int initial_capacity,
    GpuTimings& timings
) {
    const int Pn = pattern.n;
    if (Pn > MAX_PATTERN_VERTICES) {
        std::cerr << "Pattern size > MAX_PATTERN_VERTICES (" 
                  << MAX_PATTERN_VERTICES << ")\n";
        std::exit(EXIT_FAILURE);
    }
    timings.per_depth_ms.assign(Pn, 0.0);

    // Precompute per-depth pattern-edge bitmask: bit i set iff edge(depth, i).
    std::vector<int> prev_edge_mask(Pn, 0);
    for (int d = 0; d < Pn; ++d) {
        int m = 0;
        for (int i = 0; i < d; ++i) {
            if (cpu_csr_has_edge(d, i, pattern)) m |= (1 << i);
        }
        prev_edge_mask[d] = m;
    }

    // Upload data graph CSR and candidates. Pattern CSR is not needed on GPU.
    int *d_g_row = nullptr, *d_g_col = nullptr, *d_cand = nullptr;
    CUDA_CHECK(cudaMalloc(&d_g_row, data.row_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_g_col, data.col_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cand,  cand_flat.size()    * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_g_row, data.row_ptr.data(),
                          data.row_ptr.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_col, data.col_idx.data(),
                          data.col_idx.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cand, cand_flat.data(),
                          cand_flat.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Frontier ping-pong buffers, both resizable.
    int capacity = initial_capacity;
    PartialMapping *d_a = nullptr, *d_b = nullptr;
    int *d_out_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, capacity * sizeof(PartialMapping)));
    CUDA_CHECK(cudaMalloc(&d_b, capacity * sizeof(PartialMapping)));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));

    // Initialize with one empty mapping at depth 0.
    PartialMapping root{};
    root.depth = 0;
    for (int i = 0; i < MAX_PATTERN_VERTICES; ++i) root.map[i] = -1;
    CUDA_CHECK(cudaMemcpy(d_a, &root, sizeof(PartialMapping),
                          cudaMemcpyHostToDevice));
    int frontier_size = 1;

    // Resize both buffers; preserves d_a's first frontier_size entries.
    auto resize_buffers = [&](int new_cap) {
        PartialMapping* d_a_new = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a_new, new_cap * sizeof(PartialMapping)));
        if (frontier_size > 0) {
            CUDA_CHECK(cudaMemcpy(d_a_new, d_a,
                                  frontier_size * sizeof(PartialMapping),
                                  cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaFree(d_a)); d_a = d_a_new;
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaMalloc(&d_b, new_cap * sizeof(PartialMapping)));
        capacity = new_cap;
    };

    auto t_total_start = std::chrono::high_resolution_clock::now();

    for (int depth = 0; depth < Pn; ++depth) {
        int cand_off = cand_offsets[depth];
        int cand_cnt = cand_counts[depth];

        if (frontier_size == 0 || cand_cnt == 0) {
            frontier_size = 0;
            break;
        }

        auto t_d_start = std::chrono::high_resolution_clock::now();

        int h_out_count = 0;
        while (true) {
            CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

            int blocks_x = (cand_cnt + EXPAND_BLOCK_SIZE - 1) / EXPAND_BLOCK_SIZE;

            // Chunk the frontier_size dimension to stay within grid Y limits.
            for (int chunk_start = 0;
                 chunk_start < frontier_size;
                 chunk_start += MAX_GRID_Y) {
                int chunk_size = std::min(MAX_GRID_Y,
                                          frontier_size - chunk_start);
                dim3 grid(blocks_x, chunk_size);
                dim3 block(EXPAND_BLOCK_SIZE);

                expand_frontier_kernel<<<grid, block>>>(
                    d_a + chunk_start, chunk_size,
                    d_b, d_out_count, capacity,
                    depth,
                    sym_pred[depth],
                    prev_edge_mask[depth],
                    cand_off, cand_cnt,
                    d_cand, d_g_row, d_g_col
                );
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_out_count, d_out_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));

            if (h_out_count <= capacity) break;

            // Overflow: grow buffers and retry this depth.
            int new_cap = std::max(h_out_count + (h_out_count >> 1),
                                   capacity * 2);
            std::cout << "  [depth " << depth << "] frontier resize: "
                      << capacity << " -> " << new_cap
                      << " (needed " << h_out_count << ")\n";
            resize_buffers(new_cap);
            timings.resizes++;
        }

        auto t_d_end = std::chrono::high_resolution_clock::now();
        timings.per_depth_ms[depth] =
            std::chrono::duration<double, std::milli>(
                t_d_end - t_d_start).count();

        frontier_size = h_out_count;
        std::cout << "  [GPU] depth " << depth + 1
                  << " frontier: " << frontier_size << "\n";
        std::swap(d_a, d_b);
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    timings.total_ms = std::chrono::duration<double, std::milli>(
        t_total_end - t_total_start).count();

    long long matches = frontier_size;

    cudaFree(d_g_row); cudaFree(d_g_col); cudaFree(d_cand);
    cudaFree(d_a); cudaFree(d_b);
    cudaFree(d_out_count);
    return matches;
}

// ============================================================================
// CPU baseline (same constraints: NLF candidates, symmetry breaking,
// undirected single-direction edge check).
// ============================================================================
static bool cpu_is_valid_extension(
    const std::vector<int>& mapping,
    int depth,
    int candidate_g,
    int sym_pred_v,
    int prev_edge_mask,
    const CSRGraph& data,
    const CSRGraph& pattern
) {
    // Symmetry breaking.
    if (sym_pred_v >= 0 && candidate_g <= mapping[sym_pred_v]) return false;
    // Injectivity.
    for (int i = 0; i < depth; ++i) {
        if (mapping[i] == candidate_g) return false;
    }
    // Edge consistency (single direction; pattern edge bits in mask).
    for (int i = 0; i < depth; ++i) {
        if ((prev_edge_mask >> i) & 1) {
            if (!cpu_csr_has_edge(candidate_g, mapping[i], data)) return false;
        }
    }
    (void)pattern;
    return true;
}

static long long run_cpu(
    const CSRGraph& pattern,
    const CSRGraph& data,
    const std::vector<int>& cand_offsets,
    const std::vector<int>& cand_counts,
    const std::vector<int>& cand_flat,
    const std::vector<int>& sym_pred
) {
    int Pn = pattern.n;

    std::vector<int> prev_edge_mask(Pn, 0);
    for (int d = 0; d < Pn; ++d) {
        int m = 0;
        for (int i = 0; i < d; ++i) {
            if (cpu_csr_has_edge(d, i, pattern)) m |= (1 << i);
        }
        prev_edge_mask[d] = m;
    }

    std::vector<std::vector<int>> frontier;
    frontier.push_back(std::vector<int>(Pn, -1));

    for (int depth = 0; depth < Pn; ++depth) {
        std::vector<std::vector<int>> next;
        next.reserve(frontier.size());

        int off = cand_offsets[depth];
        int cnt = cand_counts[depth];

        for (const auto& m : frontier) {
            for (int c = 0; c < cnt; ++c) {
                int g = cand_flat[off + c];
                if (!cpu_is_valid_extension(m, depth, g,
                                            sym_pred[depth],
                                            prev_edge_mask[depth],
                                            data, pattern)) {
                    continue;
                }
                std::vector<int> nm = m;
                nm[depth] = g;
                next.push_back(std::move(nm));
            }
        }
        std::cout << "  [CPU] depth " << depth + 1
                  << " frontier: " << next.size() << "\n";
        frontier = std::move(next);
        if (frontier.empty()) break;
    }
    return (long long)frontier.size();
}

// ============================================================================
// I/O
// ============================================================================
static std::vector<std::pair<int, int>> load_edge_list(const std::string& fn) {
    std::ifstream f(fn);
    if (!f) {
        std::cerr << "ERROR: cannot open edge list " << fn << "\n";
        std::exit(EXIT_FAILURE);
    }
    std::vector<std::pair<int, int>> es;
    int u, v;
    while (f >> u >> v) es.push_back({u, v});
    return es;
}

static int load_metadata_int(const std::string& fn,
                             const std::string& key) {
    std::ifstream f(fn);
    if (!f) {
        std::cerr << "ERROR: cannot open metadata " << fn << "\n";
        std::exit(EXIT_FAILURE);
    }
    // Parse line-by-line so a line whose value is non-integer (e.g.
    // "mode er", "planted_components edge_2 17") doesn't poison the stream
    // and prevent later integer keys from being found.
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string k;
        if (!(iss >> k)) continue;
        if (k != key) continue;
        int v;
        if (iss >> v) return v;
    }
    std::cerr << "ERROR: key '" << key << "' not found in " << fn << "\n";
    std::exit(EXIT_FAILURE);
}

struct QueryInfo {
    std::string name;
    std::string pattern_file;
    int n;
};

static std::vector<QueryInfo> load_queries(const std::string& fn) {
    std::ifstream f(fn);
    if (!f) {
        std::cerr << "ERROR: cannot open queries " << fn << "\n";
        std::exit(EXIT_FAILURE);
    }
    std::vector<QueryInfo> qs;
    QueryInfo q;
    while (f >> q.name >> q.pattern_file >> q.n) qs.push_back(q);
    return qs;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_edges> <metadata> <queries>"
                  << " [initial_frontier_capacity]\n"
                  << "Example: " << argv[0]
                  << " data_edges.txt metadata.txt queries.txt 1048576\n";
        return EXIT_FAILURE;
    }
    std::string data_file     = argv[1];
    std::string metadata_file = argv[2];
    std::string queries_file  = argv[3];
    int initial_capacity = (argc >= 5) ? std::atoi(argv[4]) : (1 << 20);
    if (initial_capacity <= 0) initial_capacity = 1 << 20;

    auto data_edges = load_edge_list(data_file);
    int  num_vertices = load_metadata_int(metadata_file, "num_vertices");
    auto queries = load_queries(queries_file);

    bool undirected = true;
    CSRGraph G = build_csr(num_vertices, data_edges, undirected);

    std::cout << "data graph:           " << G.n << " vertices, "
              << (int)G.col_idx.size() / (undirected ? 2 : 1)
              << " logical edges\n";
    std::cout << "queries:              " << queries.size() << "\n";
    std::cout << "initial frontier cap: " << initial_capacity << "\n";

    long long total_cpu_ms = 0;
    long long total_gpu_ms = 0;
    int passes = 0, fails = 0;

    // Per-query summary kept for the final table.
    struct Row {
        std::string name;
        long long cpu_matches;
        long long gpu_matches;
        double cpu_ms;
        double gpu_ms;
        bool ok;
    };
    std::vector<Row> rows;
    rows.reserve(queries.size());

    for (const auto& q : queries) {
        std::cout << "\n=== Query: " << q.name << " ===\n";
        auto p_edges = load_edge_list(q.pattern_file);
        CSRGraph P = build_csr(q.n, p_edges, undirected);

        std::vector<int> cand_off, cand_cnt, cand_flat;
        build_nlf_candidates(P, G, cand_off, cand_cnt, cand_flat);

        long long total_cands = 0;
        for (int p = 0; p < P.n; ++p) total_cands += cand_cnt[p];
        std::cout << "  NLF candidates total=" << total_cands;
        for (int p = 0; p < P.n; ++p) {
            std::cout << " p" << p << "=" << cand_cnt[p];
        }
        std::cout << "\n";

        auto sym_pred = build_symmetry_predecessors(P);
        std::cout << "  sym_pred:";
        for (int p = 0; p < P.n; ++p) std::cout << " " << sym_pred[p];
        std::cout << "\n";

        // CPU baseline.
        auto t1 = std::chrono::high_resolution_clock::now();
        long long cpu_m = run_cpu(P, G, cand_off, cand_cnt, cand_flat, sym_pred);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // GPU.
        GpuTimings gt;
        long long gpu_m = run_gpu(P, G, cand_off, cand_cnt, cand_flat,
                                  sym_pred, initial_capacity, gt);

        std::cout << "  CPU matches: " << cpu_m
                  << ", time: " << cpu_ms << " ms\n";
        std::cout << "  GPU matches: " << gpu_m
                  << ", time: " << gt.total_ms << " ms";
        if (gt.resizes) std::cout << " (" << gt.resizes << " resize(s))";
        std::cout << "\n";

        std::cout << "  GPU per-depth ms:";
        for (double ms : gt.per_depth_ms) std::cout << " " << ms;
        std::cout << "\n";

        if (gt.total_ms > 0) {
            std::cout << "  Speedup: " << (cpu_ms / gt.total_ms) << "x\n";
        }

        bool ok = (cpu_m == gpu_m);
        if (ok) { std::cout << "  Correctness: PASS\n"; ++passes; }
        else    { std::cout << "  Correctness: FAIL\n"; ++fails; }

        total_cpu_ms += (long long)cpu_ms;
        total_gpu_ms += (long long)gt.total_ms;
        rows.push_back({q.name, cpu_m, gpu_m, cpu_ms, gt.total_ms, ok});
    }

    // Final summary table (CSV-ish, easy to grep).
    std::cout << "\n=== Summary ===\n";
    std::cout << "queries: " << queries.size()
              << "  passes: " << passes
              << "  fails: " << fails << "\n";
    std::cout << "total CPU time: " << total_cpu_ms << " ms\n";
    std::cout << "total GPU time: " << total_gpu_ms << " ms\n";
    if (total_gpu_ms > 0) {
        std::cout << "aggregate speedup: "
                  << ((double)total_cpu_ms / (double)total_gpu_ms) << "x\n";
    }

    std::cout << "\nname,cpu_matches,gpu_matches,cpu_ms,gpu_ms,speedup,ok\n";
    for (const auto& r : rows) {
        double sp = (r.gpu_ms > 0.0) ? (r.cpu_ms / r.gpu_ms) : 0.0;
        std::cout << r.name << ","
                  << r.cpu_matches << ","
                  << r.gpu_matches << ","
                  << r.cpu_ms << ","
                  << r.gpu_ms << ","
                  << sp << ","
                  << (r.ok ? "PASS" : "FAIL") << "\n";
    }

    return (fails == 0) ? 0 : 1;
}

