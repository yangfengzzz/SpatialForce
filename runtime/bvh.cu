//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <algorithm>

#include "bvh.h"

namespace wp {
namespace {
struct PartitionPredicateMedian {
    PartitionPredicateMedian(const bounds3* bounds, int a) : bounds(bounds), axis(a) {}

    bool operator()(int a, int b) const { return bounds[a].center()[axis] < bounds[b].center()[axis]; }

    const bounds3* bounds;
    int axis;
};

class MedianBVHBuilder {
public:
    void build(bvh_t& bvh, const bounds3* items, int n);

private:
    static bounds3 calc_bounds(const bounds3* bounds, const int* indices, int start, int end);

    static int partition_median(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);
    static int partition_midpoint(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);
    static int partition_sah(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds);

    int build_recursive(bvh_t& bvh, const bounds3* bounds, int* indices, int start, int end, int depth, int parent);
};

//////////////////////////////////////////////////////////////////////

void MedianBVHBuilder::build(bvh_t& bvh, const bounds3* items, int n) {
    bvh.max_depth = 0;
    bvh.max_nodes = 2 * n - 1;
    bvh.num_nodes = 0;

    bvh.node_lowers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_uppers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_parents = new int[bvh.max_nodes];
    bvh.node_counts = nullptr;

    // root is always in first slot for top down builders
    bvh.root = 0;

    if (n == 0) return;

    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    build_recursive(bvh, items, &indices[0], 0, n, 0, -1);
}

bounds3 MedianBVHBuilder::calc_bounds(const bounds3* bounds, const int* indices, int start, int end) {
    bounds3 u;

    for (int i = start; i < end; ++i) u = bounds_union(u, bounds[indices[i]]);

    return u;
}

int MedianBVHBuilder::partition_median(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds) {
    assert(end - start >= 2);

    vec3 edges = range_bounds.edges();

    int axis = longest_axis(edges);

    const int k = (start + end) / 2;

    std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(&bounds[0], axis));

    return k;
}

struct PartitionPredictateMidPoint {
    PartitionPredictateMidPoint(const bounds3* bounds, int a, float m) : bounds(bounds), axis(a), mid(m) {}

    bool operator()(int index) const { return bounds[index].center()[axis] <= mid; }

    const bounds3* bounds;
    int axis;
    float mid;
};

int MedianBVHBuilder::partition_midpoint(
        const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds) {
    assert(end - start >= 2);

    vec3 edges = range_bounds.edges();
    vec3 center = range_bounds.center();

    int axis = longest_axis(edges);
    float mid = center[axis];

    int* upper = std::partition(indices + start, indices + end, PartitionPredictateMidPoint(&bounds[0], axis, mid));

    int k = upper - indices;

    // if we failed to split items then just split in the middle
    if (k == start || k == end) k = (start + end) / 2;

    return k;
}

int MedianBVHBuilder::partition_sah(const bounds3* bounds, int* indices, int start, int end, bounds3 range_bounds) {
    assert(end - start >= 2);

    int n = end - start;
    vec3 edges = range_bounds.edges();

    int longestAxis = longest_axis(edges);

    // sort along longest axis
    std::sort(&indices[0] + start, &indices[0] + end, PartitionPredicateMedian(&bounds[0], longestAxis));

    // total area for range from [0, split]
    std::vector<float> left_areas(n);
    // total area for range from (split, end]
    std::vector<float> right_areas(n);

    bounds3 left;
    bounds3 right;

    // build cumulative bounds and area from left and right
    for (int i = 0; i < n; ++i) {
        left = bounds_union(left, bounds[indices[start + i]]);
        right = bounds_union(right, bounds[indices[end - i - 1]]);

        left_areas[i] = left.area();
        right_areas[n - i - 1] = right.area();
    }

    float invTotalArea = 1.0f / range_bounds.area();

    // find split point i that minimizes area(left[i]) + area(right[i])
    int minSplit = 0;
    float minCost = FLT_MAX;

    for (int i = 0; i < n; ++i) {
        float pBelow = left_areas[i] * invTotalArea;
        float pAbove = right_areas[i] * invTotalArea;

        float cost = pBelow * i + pAbove * (n - i);

        if (cost < minCost) {
            minCost = cost;
            minSplit = i;
        }
    }

    return start + minSplit + 1;
}

int MedianBVHBuilder::build_recursive(
        bvh_t& bvh, const bounds3* bounds, int* indices, int start, int end, int depth, int parent) {
    assert(start < end);

    const int n = end - start;
    const int node_index = bvh.num_nodes++;

    assert(node_index < bvh.max_nodes);

    if (depth > bvh.max_depth) bvh.max_depth = depth;

    bounds3 b = calc_bounds(bounds, indices, start, end);

    const int kMaxItemsPerLeaf = 1;

    if (n <= kMaxItemsPerLeaf) {
        bvh.node_lowers[node_index] = make_node(b.lower, indices[start], true);
        bvh.node_uppers[node_index] = make_node(b.upper, indices[start], false);
        bvh.node_parents[node_index] = parent;
    } else {
        // int split = partition_midpoint(bounds, indices, start, end, b);
        int split = partition_median(bounds, indices, start, end, b);
        // int split = partition_sah(bounds, indices, start, end, b);

        if (split == start || split == end) {
            // partitioning failed, split down the middle
            split = (start + end) / 2;
        }

        int left_child = build_recursive(bvh, bounds, indices, start, split, depth + 1, node_index);
        int right_child = build_recursive(bvh, bounds, indices, split, end, depth + 1, node_index);

        bvh.node_lowers[node_index] = make_node(b.lower, left_child, false);
        bvh.node_uppers[node_index] = make_node(b.upper, right_child, false);
        bvh.node_parents[node_index] = parent;
    }

    return node_index;
}

class LinearBVHBuilderCPU {
public:
    void build(bvh_t& bvh, const bounds3* items, int n);

private:
    // calculate Morton codes
    struct KeyIndexPair {
        uint32_t key;
        int index;

        inline bool operator<(const KeyIndexPair& rhs) const { return key < rhs.key; }
    };

    static bounds3 calc_bounds(const bounds3* bounds, const KeyIndexPair* keys, int start, int end);
    static int find_split(const KeyIndexPair* pairs, int start, int end);
    int build_recursive(bvh_t& bvh, const KeyIndexPair* keys, const bounds3* bounds, int start, int end, int depth);
};

void LinearBVHBuilderCPU::build(bvh_t& bvh, const bounds3* items, int n) {
    memset(&bvh, 0, sizeof(bvh_t));

    bvh.max_nodes = 2 * n - 1;

    bvh.node_lowers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.node_uppers = new BVHPackedNodeHalf[bvh.max_nodes];
    bvh.num_nodes = 0;

    // root is always in first slot for top down builders
    bvh.root = 0;

    std::vector<KeyIndexPair> keys;
    keys.reserve(n);

    bounds3 totalbounds3;
    for (int i = 0; i < n; ++i) totalbounds3 = bounds_union(totalbounds3, items[i]);

    // ensure non-zero edge length in all dimensions
    totalbounds3.expand(0.001f);

    vec3 edges = totalbounds3.edges();
    vec3 invEdges = cw_div(vec3(1.0f), edges);

    for (int i = 0; i < n; ++i) {
        vec3 center = items[i].center();
        vec3 local = cw_mul(center - totalbounds3.lower, invEdges);

        KeyIndexPair l{};
        l.key = morton3<1024>(local[0], local[1], local[2]);
        l.index = i;

        keys.push_back(l);
    }

    // sort by key
    std::sort(keys.begin(), keys.end());

    build_recursive(bvh, &keys[0], items, 0, n, 0);

    printf("Created BVH for %d items with %d nodes, max depth of %d\n", n, bvh.num_nodes, bvh.max_depth);
}

inline bounds3 LinearBVHBuilderCPU::calc_bounds(const bounds3* bounds, const KeyIndexPair* keys, int start, int end) {
    bounds3 u;

    for (int i = start; i < end; ++i) u = bounds_union(u, bounds[keys[i].index]);

    return u;
}

inline int LinearBVHBuilderCPU::find_split(const KeyIndexPair* pairs, int start, int end) {
    if (pairs[start].key == pairs[end - 1].key) return (start + end) / 2;

    // find split point between keys, xor here means all bits
    // of the result are zero up until the first differing bit
    int common_prefix = clz(pairs[start].key ^ pairs[end - 1].key);

    // use binary search to find the point at which this bit changes
    // from zero to a 1
    const int mask = 1 << (31 - common_prefix);

    while (end - start > 0) {
        int index = (start + end) / 2;

        if (pairs[index].key & mask) {
            end = index;
        } else
            start = index + 1;
    }

    assert(start == end);

    return start;
}

int LinearBVHBuilderCPU::build_recursive(
        bvh_t& bvh, const KeyIndexPair* keys, const bounds3* bounds, int start, int end, int depth) {
    assert(start < end);

    const int n = end - start;
    const int nodeIndex = bvh.num_nodes++;

    assert(nodeIndex < bvh.max_nodes);

    if (depth > bvh.max_depth) bvh.max_depth = depth;

    bounds3 b = calc_bounds(bounds, keys, start, end);

    const int kMaxItemsPerLeaf = 1;

    if (n <= kMaxItemsPerLeaf) {
        bvh.node_lowers[nodeIndex] = make_node(b.lower, keys[start].index, true);
        bvh.node_uppers[nodeIndex] = make_node(b.upper, keys[start].index, false);
    } else {
        int split = find_split(keys, start, end);

        int leftChild = build_recursive(bvh, keys, bounds, start, split, depth + 1);
        int rightChild = build_recursive(bvh, keys, bounds, split, end, depth + 1);

        bvh.node_lowers[nodeIndex] = make_node(b.lower, leftChild, false);
        bvh.node_uppers[nodeIndex] = make_node(b.upper, rightChild, false);
    }

    return nodeIndex;
}
}  // namespace

BVH::BVH(Stream& stream, const Array<vec3f>& lowers, const Array<vec3f>& uppers) : stream_{stream} {}

BVH::~BVH() {
    Device::free(descriptor_.node_lowers);
    descriptor_.node_lowers = nullptr;
    Device::free(descriptor_.node_uppers);
    descriptor_.node_uppers = nullptr;
    Device::free(descriptor_.node_parents);
    descriptor_.node_parents = nullptr;
    Device::free(descriptor_.node_counts);
    descriptor_.node_counts = nullptr;
    Device::free(descriptor_.bounds);
    descriptor_.bounds = nullptr;
}

__global__ void bvh_refit_kernel(int n,
                                 const int* __restrict__ parents,
                                 int* __restrict__ child_count,
                                 BVHPackedNodeHalf* __restrict__ lowers,
                                 BVHPackedNodeHalf* __restrict__ uppers,
                                 const bounds3* bounds) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        bool leaf = lowers[index].b;

        if (leaf) {
            // update the leaf node
            const int leaf_index = lowers[index].i;
            const bounds3& b = bounds[leaf_index];

            make_node(lowers + index, b.lower, leaf_index, true);
            make_node(uppers + index, b.upper, 0, false);
        } else {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;) {
            int parent = parents[index];

            // reached root
            if (parent == -1) return;

            // ensure all writes are visible
            __threadfence();

            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the the next parent in the hierarchy
            if (finished == 1) {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                vec3 left_lower = vec3(lowers[left_child].x, lowers[left_child].y, lowers[left_child].z);

                vec3 left_upper = vec3(uppers[left_child].x, uppers[left_child].y, uppers[left_child].z);

                vec3 right_lower = vec3(lowers[right_child].x, lowers[right_child].y, lowers[right_child].z);

                vec3 right_upper = vec3(uppers[right_child].x, uppers[right_child].y, uppers[right_child].z);

                // union of child bounds
                vec3 lower = min(left_lower, right_lower);
                vec3 upper = max(left_upper, right_upper);

                // write new BVH nodes
                make_node(lowers + parent, lower, left_child, false);
                make_node(uppers + parent, upper, right_child, false);

                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
}

__global__ void set_bounds_from_lowers_and_uppers(int n, bounds3* b, const vec3* lowers, const vec3* uppers) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        b[tid] = bounds3(lowers[tid], uppers[tid]);
    }
}

void BVH::refit() {
    wp_launch_device((CUstream)stream_.handle(), wp::set_bounds_from_lowers_and_uppers, descriptor_.num_bounds,
                     (descriptor_.num_bounds, descriptor_.bounds, descriptor_.lowers, descriptor_.uppers));
    BVH::refit(stream_, descriptor_, descriptor_.bounds);
}

void BVH::refit(Stream& stream, bvh_t& bvh, const bounds3* b) {
    // clear child counters
    stream.memset(bvh.node_counts, 0, sizeof(int) * bvh.max_nodes);

    wp_launch_device((CUstream)stream.handle(), bvh_refit_kernel, bvh.max_nodes,
                     (bvh.max_nodes, bvh.node_parents, bvh.node_counts, bvh.node_lowers, bvh.node_uppers, b));
}
}  // namespace wp