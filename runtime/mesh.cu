//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/bvh.h"
#include "runtime/mesh.h"
#include "runtime/scan.h"
#include "runtime/stream.h"

namespace wp {
namespace {
__global__ void compute_triangle_bounds(int n, const vec3* points, const int* indices, bounds3* b) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // if leaf then update bounds
        int i = indices[tid * 3 + 0];
        int j = indices[tid * 3 + 1];
        int k = indices[tid * 3 + 2];

        vec3 p = points[i];
        vec3 q = points[j];
        vec3 r = points[k];

        vec3 lower = min(min(p, q), r);
        vec3 upper = max(max(p, q), r);

        b[tid] = bounds3(lower, upper);
    }
}

__global__ void compute_mesh_edge_lengths(int n, const vec3* points, const int* indices, float* edge_lengths) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // if leaf then update bounds
        int i = indices[tid * 3 + 0];
        int j = indices[tid * 3 + 1];
        int k = indices[tid * 3 + 2];

        vec3 p = points[i];
        vec3 q = points[j];
        vec3 r = points[k];

        edge_lengths[tid] = length(p - q) + length(p - r) + length(q - r);
    }
}

__global__ void compute_average_mesh_edge_length(int n, const float* sum_edge_lengths, mesh_t* m) {
    m->average_edge_length = sum_edge_lengths[n - 1] / (3 * n);
}

__global__ void bvh_refit_with_solid_angle_kernel(int n,
                                                  const int* __restrict__ parents,
                                                  int* __restrict__ child_count,
                                                  BVHPackedNodeHalf* __restrict__ lowers,
                                                  BVHPackedNodeHalf* __restrict__ uppers,
                                                  const vec3* points,
                                                  const int* indices,
                                                  SolidAngleProps* solid_angle_props) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        bool leaf = lowers[index].b;

        if (leaf) {
            // update the leaf node
            const int leaf_index = lowers[index].i;
            precompute_triangle_solid_angle_props(points[indices[leaf_index * 3 + 0]],
                                                  points[indices[leaf_index * 3 + 1]],
                                                  points[indices[leaf_index * 3 + 2]], solid_angle_props[index]);

            make_node(lowers + index, solid_angle_props[index].box.lower, leaf_index, true);
            make_node(uppers + index, solid_angle_props[index].box.upper, 0, false);
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
                // printf("Compute non-leaf at %d\n", index);
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

                // combine
                SolidAngleProps* left_child_data = &solid_angle_props[left_child];
                SolidAngleProps* right_child_data =
                        (left_child != right_child) ? &solid_angle_props[right_child] : nullptr;

                combine_precomputed_solid_angle_props(solid_angle_props[parent], left_child_data, right_child_data);

                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
}
}  // namespace
Mesh::Mesh(Stream& stream,
           const Array<vec3>& points,
           const Array<vec3>& velocities,
           const Array<int>& indices,
           bool support_winding_number)
    : stream_{stream} {}

Mesh::~Mesh() {}

void Mesh::refit() {
    // we compute mesh the average edge length
    // for use in mesh_query_point_sign_normal()
    // since it relies on an epsilon for welding

    // re-use bounds memory temporarily for computing edge lengths
    auto* length_tmp_ptr = (float*)descriptor_.bounds;
    wp_launch_device((CUstream)stream_.handle(), wp::compute_mesh_edge_lengths, descriptor_.num_tris,
                     (descriptor_.num_tris, descriptor_.points, descriptor_.indices, length_tmp_ptr));

    scan(stream_, length_tmp_ptr, length_tmp_ptr, descriptor_.num_tris, true);

    wp_launch_device((CUstream)stream_.handle(), wp::compute_average_mesh_edge_length, 1,
                     (descriptor_.num_tris, length_tmp_ptr, (wp::mesh_t*)id_));
    wp_launch_device((CUstream)stream_.handle(), wp::compute_triangle_bounds, descriptor_.num_tris,
                     (descriptor_.num_tris, descriptor_.points, descriptor_.indices, descriptor_.bounds));

    if (descriptor_.solid_angle_props) {
        bvh_refit_with_solid_angle_device(descriptor_.bvh, descriptor_);
    } else {
        BVH::refit(stream_, descriptor_.bvh, descriptor_.bounds);
    }
}

void Mesh::bvh_refit_with_solid_angle_device(bvh_t& bvh, mesh_t& mesh) {
    // clear child counters
    stream_.memset(bvh.node_counts, 0, sizeof(int) * bvh.max_nodes);

    wp_launch_device((CUstream)stream_.handle(), bvh_refit_with_solid_angle_kernel, bvh.max_nodes,
                     (bvh.max_nodes, bvh.node_parents, bvh.node_counts, bvh.node_lowers, bvh.node_uppers, mesh.points,
                      mesh.indices, mesh.solid_angle_props));
}

}  // namespace wp