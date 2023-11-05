//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "collide.h"
#include "core/mesh.h"
//#include "core/volume.h"

namespace wp {
CUDA_CALLABLE_DEVICE float sphere_sdf(vec3f center, float radius, vec3f p) { return length(p - center) - radius; }

CUDA_CALLABLE_DEVICE vec3f sphere_sdf_grad(vec3f center, float radius, vec3f p) { return normalize(p - center); }

CUDA_CALLABLE_DEVICE float box_sdf(vec3f upper, vec3f p) {
    // adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    auto qx = abs(p[0]) - upper[0];
    auto qy = abs(p[1]) - upper[1];
    auto qz = abs(p[2]) - upper[2];

    auto e = vec3(max(qx, 0.f), max(qy, 0.f), max(qz, 0.f));

    return length(e) + min(max(qx, max(qy, qz)), 0.f);
}

CUDA_CALLABLE_DEVICE vec3f box_sdf_grad(vec3f upper, vec3f p) {
    auto qx = abs(p[0]) - upper[0];
    auto qy = abs(p[1]) - upper[1];
    auto qz = abs(p[2]) - upper[2];

    // exterior case
    if (qx > 0.0 || qy > 0.0 || qz > 0.0) {
        auto x = clamp(p[0], -upper[0], upper[0]);
        auto y = clamp(p[1], -upper[1], upper[1]);
        auto z = clamp(p[2], -upper[2], upper[2]);

        return normalize(p - vec3(x, y, z));
    }
    auto sx = sign(p[0]);
    auto sy = sign(p[1]);
    auto sz = sign(p[2]);

    //   x projection
    if (qx > qy && qx > qz || qy == 0.0 && qz == 0.0) {
        return {sx, 0.0, 0.0};
    }
    //  y projection
    if (qy > qx && qy > qz || qx == 0.0 && qz == 0.0) {
        return {0.0, sy, 0.0};
    }
    //   z projection
    return {0.0, 0.0, sz};
}

CUDA_CALLABLE_DEVICE float capsule_sdf(float radius, float half_height, vec3f p) {
    if (p[1] > half_height) {
        return length(vec3(p[0], p[1] - half_height, p[2])) - radius;
    }
    if (p[1] < -half_height) {
        return length(vec3(p[0], p[1] + half_height, p[2])) - radius;
    }
    return length(vec3(p[0], 0.0, p[2])) - radius;
}

CUDA_CALLABLE_DEVICE vec3f capsule_sdf_grad(float radius, float half_height, vec3f p) {
    if (p[1] > half_height) {
        return normalize(vec3(p[0], p[1] - half_height, p[2]));
    }
    if (p[1] < -half_height) {
        return normalize(vec3(p[0], p[1] + half_height, p[2]));
    }
    return normalize(vec3(p[0], 0.0, p[2]));
}

CUDA_CALLABLE_DEVICE float cylinder_sdf(float radius, float half_height, vec3f p) {
    auto dx = length(vec3(p[0], 0.0, p[2])) - radius;
    auto dy = abs(p[1]) - half_height;
    return min(max(dx, dy), 0.f) + length(vec2(max(dx, 0.f), max(dy, 0.f)));
}

CUDA_CALLABLE_DEVICE vec3f cylinder_sdf_grad(float radius, float half_height, vec3f p) {
    auto dx = length(vec3(p[0], 0.0, p[2])) - radius;
    auto dy = abs(p[1]) - half_height;
    if (dx > dy) {
        return normalize(vec3(p[0], 0.0, p[2]));
    }
    return {0.0, sign(p[1]), 0.0};
}

CUDA_CALLABLE_DEVICE float cone_sdf(float radius, float half_height, vec3f p) {
    auto dx = length(vec3(p[0], 0.f, p[2])) - radius * (p[1] + half_height) / (2.f * half_height);
    auto dy = abs(p[1]) - half_height;
    return min(max(dx, dy), 0.f) + length(vec2(max(dx, 0.f), max(dy, 0.f)));
}

CUDA_CALLABLE_DEVICE vec3f cone_sdf_grad(float radius, float half_height, vec3f p) {
    auto dx = length(vec3(p[0], 0.0, p[2])) - radius * (p[1] + half_height) / (2.f * half_height);
    auto dy = abs(p[1]) - half_height;
    if (dy < 0.0 || dx == 0.0) return {0.0, sign(p[1]), 0.0};
    return normalize(vec3(p[0], 0.0, p[2])) + vec3(0.0, radius / (2.f * half_height), 0.0);
}

CUDA_CALLABLE_DEVICE float plane_sdf(float width, float length, vec3f p) {
    // SDF for a quad in the xz plane
    if (width > 0.0 && length > 0.0) {
        auto d = max(abs(p[0]) - width, abs(p[2]) - length);
        return max(d, abs(p[1]));
    }
    return p[1];
}

CUDA_CALLABLE_DEVICE vec3f closest_point_plane(float width, float length, vec3f point) {
    // projects the point onto the quad in the xz plane (if width and length > 0.0, otherwise the plane is infinite)
    float x, z;
    if (width > 0.f) {
        x = clamp(point[0], -width, width);
    } else {
        x = point[0];
    }
    if (length > 0.f) {
        z = clamp(point[2], -length, length);
    } else {
        z = point[2];
    }
    return {x, 0.0, z};
}

CUDA_CALLABLE_DEVICE vec3f closest_point_line_segment(vec3f a, vec3f b, vec3f point) {
    auto ab = b - a;
    auto ap = point - a;
    auto t = dot(ap, ab) / dot(ab, ab);
    t = clamp(t, 0.f, 1.f);
    return a + t * ab;
}

CUDA_CALLABLE_DEVICE vec3f closest_point_box(vec3f upper, vec3f point) {
    // closest point to box surface
    auto x = clamp(point[0], -upper[0], upper[0]);
    auto y = clamp(point[1], -upper[1], upper[1]);
    auto z = clamp(point[2], -upper[2], upper[2]);
    if (abs(point[0]) <= upper[0] && abs(point[1]) <= upper[1] && abs(point[2]) <= upper[2]) {
        // the point is inside, find closest face
        auto sx = abs(abs(point[0]) - upper[0]);
        auto sy = abs(abs(point[1]) - upper[1]);
        auto sz = abs(abs(point[2]) - upper[2]);
        // return closest point on closest side, handle corner cases
        if (sx < sy && sx < sz || sy == 0.0 && sz == 0.0) {
            x = sign(point[0]) * upper[0];
        } else if (sy < sx && sy < sz || sx == 0.0 && sz == 0.0) {
            y = sign(point[1]) * upper[1];
        } else {
            z = sign(point[2]) * upper[2];
        }
    }
    return {x, y, z};
}

CUDA_CALLABLE_DEVICE vec3f get_box_vertex(int point_id, vec3f upper) {
    // get the vertex of the box given its ID (0-7);
    auto sign_x = float(point_id % 2) * 2.f - 1.f;
    auto sign_y = float((point_id / 2) % 2) * 2.f - 1.f;
    auto sign_z = float((point_id / 4) % 2) * 2.f - 1.f;
    return {sign_x * upper[0], sign_y * upper[1], sign_z * upper[2]};
}

CUDA_CALLABLE_DEVICE spatial_vectorf get_box_edge(int edge_id, vec3f upper) {
    int i, j;
    // get the edge of the box given its ID (0-11)
    if (edge_id < 4) {
        // edges along x: 0-1, 2-3, 4-5, 6-7
        i = edge_id * 2;
        j = i + 1;
        return {get_box_vertex(i, upper), get_box_vertex(j, upper)};
    } else if (edge_id < 8) {
        // edges along y: 0-2, 1-3, 4-6, 5-7
        edge_id -= 4;
        i = edge_id % 2 + edge_id / 2 * 4;
        j = i + 2;
        return {get_box_vertex(i, upper), get_box_vertex(j, upper)};
    }
    // edges along z: 0-4, 1-5, 2-6, 3-7
    edge_id -= 8;
    i = edge_id;
    j = i + 4;
    return {get_box_vertex(i, upper), get_box_vertex(j, upper)};
}

CUDA_CALLABLE_DEVICE spatial_vectorf get_plane_edge(int edge_id, float plane_width, float plane_length) {
    // get the edge of the plane given its ID (0-3)
    auto p0x = (2.f * float(edge_id % 2) - 1.f) * plane_width;
    auto p0z = (2.f * float(edge_id / 2) - 1.f) * plane_length;
    float p1x, p1z;
    if (edge_id == 0 || edge_id == 3) {
        p1x = p0x;
        p1z = -p0z;
    } else {
        p1x = -p0x;
        p1z = p0z;
    }
    return {vec3(p0x, 0.0, p0z), vec3(p1x, 0.0, p1z)};
}

CUDA_CALLABLE_DEVICE float closest_edge_coordinate_box(vec3f upper, vec3f edge_a, vec3f edge_b, int max_iter) {
    //  find point on edge closest to box, return its barycentric edge coordinate
    //  Golden-section search
    auto a = 0.f;
    auto b = 1.f;
    auto h = b - a;
    auto invphi = 0.61803398875f;   // 1 / phi;
    auto invphi2 = 0.38196601125f;  // 1 / phi^2;
    auto c = a + invphi2 * h;
    auto d = a + invphi * h;
    auto query = (1.f - c) * edge_a + c * edge_b;
    auto yc = box_sdf(upper, query);
    query = (1.f - d) * edge_a + d * edge_b;
    auto yd = box_sdf(upper, query);

    for (int k = 0; k < max_iter; k++) {
        // yc > yd to find the maximum
        if (yc < yd) {
            b = d;
            d = c;
            yd = yc;
            h = invphi * h;
            c = a + invphi2 * h;
            query = (1.f - c) * edge_a + c * edge_b;
            yc = box_sdf(upper, query);
        } else {
            a = c;
            c = d;
            yc = yd;
            h = invphi * h;
            d = a + invphi * h;
            query = (1.f - d) * edge_a + d * edge_b;
            yd = box_sdf(upper, query);
        }
    }
    if (yc < yd) {
        return 0.5f * (a + d);
    }
    return 0.5f * (c + b);
}

CUDA_CALLABLE_DEVICE float closest_edge_coordinate_plane(
        float plane_width, float plane_length, vec3f edge_a, vec3f edge_b, int max_iter) {
    // find point on edge closest to plane, return its barycentric edge coordinate
    // Golden-section search
    auto a = float(0.0);
    auto b = float(1.0);
    auto h = b - a;
    auto invphi = 0.61803398875f;   // 1 / phi
    auto invphi2 = 0.38196601125f;  // 1 / phi^2
    auto c = a + invphi2 * h;
    auto d = a + invphi * h;
    auto query = (1.f - c) * edge_a + c * edge_b;
    auto yc = plane_sdf(plane_width, plane_length, query);
    query = (1.f - d) * edge_a + d * edge_b;
    auto yd = plane_sdf(plane_width, plane_length, query);

    for (int k = 0; k < max_iter; k++) {
        if (yc < yd) {  // yc > yd to find the maximum
            b = d;
            d = c;
            yd = yc;
            h = invphi * h;
            c = a + invphi2 * h;
            query = (1.f - c) * edge_a + c * edge_b;
            yc = plane_sdf(plane_width, plane_length, query);
        } else {
            a = c;
            c = d;
            yc = yd;
            h = invphi * h;
            d = a + invphi * h;
            query = (1.f - d) * edge_a + d * edge_b;
            yd = plane_sdf(plane_width, plane_length, query);
        }
    }
    if (yc < yd) {
        return 0.5f * (a + d);
    }
    return 0.5f * (c + b);
}

CUDA_CALLABLE_DEVICE float closest_edge_coordinate_capsule(
        float radius, float half_height, vec3f edge_a, vec3f edge_b, int max_iter) {
    // find point on edge closest to capsule, return its barycentric edge coordinate
    // Golden-section search
    auto a = 0.f;
    auto b = 1.f;
    auto h = b - a;
    auto invphi = 0.61803398875f;   // 1 / phi
    auto invphi2 = 0.38196601125f;  // 1 / phi^2
    auto c = a + invphi2 * h;
    auto d = a + invphi * h;
    auto query = (1.f - c) * edge_a + c * edge_b;
    auto yc = capsule_sdf(radius, half_height, query);
    query = (1.f - d) * edge_a + d * edge_b;
    auto yd = capsule_sdf(radius, half_height, query);

    for (int k = 0; k < max_iter; k++) {
        if (yc < yd) {  // yc > yd to find the maximum
            b = d;
            d = c;
            yd = yc;
            h = invphi * h;
            c = a + invphi2 * h;
            query = (1.f - c) * edge_a + c * edge_b;
            yc = capsule_sdf(radius, half_height, query);
        } else {
            a = c;
            c = d;
            yc = yd;
            h = invphi * h;
            d = a + invphi * h;
            query = (1.f - d) * edge_a + d * edge_b;
            yd = capsule_sdf(radius, half_height, query);
        }
    }
    if (yc < yd) {
        return 0.5f * (a + d);
    }
    return 0.5f * (c + b);
}

CUDA_CALLABLE_DEVICE float mesh_sdf(uint64_t mesh, vec3f point, float max_dist) {
    auto face_index = int(0);
    auto face_u = float(0.0);
    auto face_v = float(0.0);
    auto sign = float(0.0);
    auto res = mesh_query_point_sign_normal(mesh, point, max_dist, sign, face_index, face_u, face_v);

    if (res) {
        auto closest = mesh_eval_position(mesh, face_index, face_u, face_v);
        return length(point - closest) * sign;
    }
    return max_dist;
}

CUDA_CALLABLE_DEVICE vec3f closest_point_mesh(uint64_t mesh, vec3f point, float max_dist) {
    auto face_index = int(0);
    auto face_u = float(0.0);
    auto face_v = float(0.0);
    auto sign = float(0.0);
    auto res = mesh_query_point_sign_normal(mesh, point, max_dist, sign, face_index, face_u, face_v);

    if (res) {
        return mesh_eval_position(mesh, face_index, face_u, face_v);
    }
    // return arbitrary point from mesh
    return mesh_eval_position(mesh, 0, 0.0, 0.0);
}

CUDA_CALLABLE_DEVICE float closest_edge_coordinate_mesh(
        uint64_t mesh, vec3f edge_a, vec3f edge_b, int max_iter, float max_dist) {
    //  find point on edge closest to mesh, return its barycentric edge coordinate
    //  Golden-section search
    auto a = float(0.0);
    auto b = float(1.0);
    auto h = b - a;
    auto invphi = 0.61803398875f;   // 1 / phi
    auto invphi2 = 0.38196601125f;  // 1 / phi^2
    auto c = a + invphi2 * h;
    auto d = a + invphi * h;
    auto query = (1.f - c) * edge_a + c * edge_b;
    auto yc = mesh_sdf(mesh, query, max_dist);
    query = (1.f - d) * edge_a + d * edge_b;
    auto yd = mesh_sdf(mesh, query, max_dist);

    for (int k = 0; k < max_iter; k++) {
        if (yc < yd) {  // yc > yd to find the maximum
            b = d;
            d = c;
            yd = yc;
            h = invphi * h;
            c = a + invphi2 * h;
            query = (1.f - c) * edge_a + c * edge_b;
            yc = mesh_sdf(mesh, query, max_dist);
        } else {
            a = c;
            c = d;
            yc = yd;
            h = invphi * h;
            d = a + invphi * h;
            query = (1.f - d) * edge_a + d * edge_b;
            yd = mesh_sdf(mesh, query, max_dist);
        }
    }
    if (yc < yd) {
        return 0.5f * (a + d);
    }
    return 0.5f * (c + b);
}

CUDA_CALLABLE_DEVICE void create_soft_contacts(const array_t<vec3f> &particle_x,
                                               const array_t<float> &particle_radius,
                                               const array_t<uint32_t> &particle_flags,
                                               const array_t<transformf> &body_X_wb,
                                               const array_t<transformf> &shape_X_bs,
                                               const array_t<int> &shape_body,
                                               const ModelShapeGeometry &geo,
                                               float margin,
                                               int soft_contact_max,
                                               array_t<int> &soft_contact_count,
                                               array_t<int> &soft_contact_particle,
                                               array_t<int> &soft_contact_shape,
                                               array_t<vec3f> &soft_contact_body_pos,
                                               array_t<vec3f> &soft_contact_body_vel,
                                               array_t<vec3f> &soft_contact_normal) {
    int particle_index, shape_index;
    tid(particle_index, shape_index);

    if ((particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }

    auto rigid_index = shape_body[shape_index];

    auto px = particle_x[particle_index];
    auto radius = particle_radius[particle_index];

    auto X_wb = transform_identity();
    if (rigid_index >= 0) {
        X_wb = body_X_wb[rigid_index];
    }
    auto X_bs = shape_X_bs[shape_index];

    auto X_ws = transform_multiply(X_wb, X_bs);
    auto X_sw = transform_inverse(X_ws);

    // transform particle position to shape local space
    auto x_local = transform_point(X_sw, px);

    // geo description
    auto geo_type = geo.type[shape_index];
    auto geo_scale = geo.scale[shape_index];

    // evaluate shape sdf
    auto d = 1.0e6f;
    auto n = vec3();
    auto v = vec3();

    if (geo_type == int(GeometryType::GEO_SPHERE)) {
        d = sphere_sdf(vec3(), geo_scale[0], x_local);
        n = sphere_sdf_grad(vec3(), geo_scale[0], x_local);
    }

    if (geo_type == int(GeometryType::GEO_BOX)) {
        d = box_sdf(geo_scale, x_local);
        n = box_sdf_grad(geo_scale, x_local);
    }

    if (geo_type == int(GeometryType::GEO_CAPSULE)) {
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local);
        n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local);
    }
    if (geo_type == int(GeometryType::GEO_CYLINDER)) {
        d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local);
        n = cylinder_sdf_grad(geo_scale[0], geo_scale[1], x_local);
    }
    if (geo_type == int(GeometryType::GEO_CONE)) {
        d = cone_sdf(geo_scale[0], geo_scale[1], x_local);
        n = cone_sdf_grad(geo_scale[0], geo_scale[1], x_local);
    }
    if (geo_type == int(GeometryType::GEO_MESH)) {
        auto mesh = geo.source[shape_index];

        auto face_index = int(0);
        auto face_u = float(0.0);
        auto face_v = float(0.0);
        auto sign = float(0.0);

        if (mesh_query_point_sign_normal(mesh, cw_div(x_local, geo_scale), margin + radius, sign, face_index, face_u,
                                         face_v)) {
            auto shape_p = mesh_eval_position(mesh, face_index, face_u, face_v);
            auto shape_v = mesh_eval_velocity(mesh, face_index, face_u, face_v);

            shape_p = cw_mul(shape_p, geo_scale);
            shape_v = cw_mul(shape_v, geo_scale);

            auto delta = x_local - shape_p;

            d = length(delta) * sign;
            n = normalize(delta) * sign;
            v = shape_v;
        }
    }
    if (geo_type == int(GeometryType::GEO_SDF)) {
        // todo
        //        auto volume = geo.source[shape_index];
        //        auto xpred_local = volume_world_to_index(volume, cw_div(x_local, geo_scale));
        //        auto nn = vec3(0.0, 0.0, 0.0);
        //        d = volume_sample_grad_f(volume, xpred_local, volume::LINEAR, nn);
        //        n = normalize(nn);
    }
    if (geo_type == int(GeometryType::GEO_PLANE)) {
        d = plane_sdf(geo_scale[0], geo_scale[1], x_local);
        n = vec3(0.0, 1.0, 0.0);
    }

    if (d < margin + radius) {
        auto index = atomic_add(soft_contact_count, 0, 1);

        if (index < soft_contact_max) {
            // compute contact point in body local space
            auto body_pos = transform_point(X_bs, x_local - n * d);
            auto body_vel = transform_vector(X_bs, v);

            auto world_normal = transform_vector(X_ws, n);

            soft_contact_shape[index] = shape_index;
            soft_contact_body_pos[index] = body_pos;
            soft_contact_body_vel[index] = body_vel;
            soft_contact_particle[index] = particle_index;
            soft_contact_normal[index] = world_normal;
        }
    }
}

CUDA_CALLABLE_DEVICE void count_contact_points(const array_t<int> &contact_pairs,
                                               const ModelShapeGeometry &geo,
                                               array_t<int> &contact_count) {
    auto tid = wp::tid();
    auto shape_a = contact_pairs(tid, 0);
    auto shape_b = contact_pairs(tid, 1);

    int actual_type_a, actual_type_b, actual_shape_a, actual_shape_b;
    if (shape_b == -1) {
        actual_type_a = geo.type[shape_a];
        // ground plane
        actual_type_b = int(GeometryType::GEO_PLANE);
    } else {
        auto type_a = geo.type[shape_a];
        auto type_b = geo.type[shape_b];
        // unique ordering of shape pairs
        if (type_a < type_b) {
            actual_shape_a = shape_a;
            actual_shape_b = shape_b;
            actual_type_a = type_a;
            actual_type_b = type_b;
        } else {
            actual_shape_a = shape_b;
            actual_shape_b = shape_a;
            actual_type_a = type_b;
            actual_type_b = type_a;
        }
    }
    // determine how many contact points need to be evaluated
    auto num_contacts = 0;
    if (actual_type_a == int(GeometryType::GEO_SPHERE)) {
        num_contacts = 1;
    } else if (actual_type_a == int(GeometryType::GEO_CAPSULE)) {
        if (actual_type_b == int(GeometryType::GEO_PLANE)) {
            if (geo.scale[actual_shape_b][0] == 0.0 && geo.scale[actual_shape_b][1] == 0.0) {
                num_contacts = 2;  // vertex-based collision for infinite plane;
            } else {
                num_contacts = 2 + 4;  // vertex-based collision + plane edges;
            }
        } else if (actual_type_b == int(GeometryType::GEO_MESH)) {
            auto num_contacts_a = 2;
            auto mesh_b = mesh_get(geo.source[actual_shape_b]);
            auto num_contacts_b = mesh_b.points.shape[0];
            num_contacts = num_contacts_a + num_contacts_b;
        } else {
            num_contacts = 2;
        }
    } else if (actual_type_a == int(GeometryType::GEO_BOX)) {
        if (actual_type_b == int(GeometryType::GEO_BOX)) {
            num_contacts = 24;
        } else if (actual_type_b == int(GeometryType::GEO_MESH)) {
            auto num_contacts_a = 8;
            auto mesh_b = mesh_get(geo.source[actual_shape_b]);
            auto num_contacts_b = mesh_b.points.shape[0];
            num_contacts = num_contacts_a + num_contacts_b;
        } else if (actual_type_b == int(GeometryType::GEO_PLANE)) {
            if (geo.scale[actual_shape_b][0] == 0.0 && geo.scale[actual_shape_b][1] == 0.0) {
                num_contacts = 8;  // vertex-based collision
            } else {
                num_contacts = 8 + 4;  // vertex-based collision + plane edges
            }
        } else {
            num_contacts = 8;
        }
    } else if (actual_type_a == int(GeometryType::GEO_MESH)) {
        auto mesh_a = mesh_get(geo.source[actual_shape_a]);
        auto num_contacts_a = mesh_a.points.shape[0];
        int num_contacts_b;
        if (actual_type_b == int(GeometryType::GEO_MESH)) {
            auto mesh_b = mesh_get(geo.source[actual_shape_b]);
            num_contacts_b = mesh_b.points.shape[0];
        } else {
            num_contacts_b = 0;
        }
        num_contacts = num_contacts_a + num_contacts_b;
    } else if (actual_type_a == int(GeometryType::GEO_PLANE)) {
        return;  // no plane-plane contacts
    } else {
        print("count_contact_points: unsupported geometry type");
        print(actual_type_a);
        print(actual_type_b);
    }
    atomic_add(contact_count, 0, num_contacts);
}

CUDA_CALLABLE_DEVICE void broadphase_collision_pairs(const array_t<int> &contact_pairs,
                                                     const array_t<transformf> &body_q,
                                                     const array_t<transformf> &shape_X_bs,
                                                     const array_t<int> &shape_body,
                                                     const ModelShapeGeometry &geo,
                                                     const array_t<float> &collision_radius,
                                                     int rigid_contact_max,
                                                     float rigid_contact_margin,
                                                     array_t<int> &contact_count,
                                                     array_t<int> &contact_shape0,
                                                     array_t<int> &contact_shape1,
                                                     array_t<int> &contact_point_id) {
    auto tid = wp::tid();
    auto shape_a = contact_pairs(tid, 0);
    auto shape_b = contact_pairs(tid, 1);

    transformf X_ws_a;
    auto rigid_a = shape_body[shape_a];
    if (rigid_a == -1) {
        X_ws_a = shape_X_bs[shape_a];
    } else {
        X_ws_a = transform_multiply(body_q[rigid_a], shape_X_bs[shape_a]);
    }

    transformf X_ws_b;
    auto rigid_b = shape_body[shape_b];
    if (rigid_b == -1) {
        X_ws_b = shape_X_bs[shape_b];
    } else {
        X_ws_b = transform_multiply(body_q[rigid_b], shape_X_bs[shape_b]);
    }
    auto type_a = geo.type[shape_a];
    auto type_b = geo.type[shape_b];

    int actual_shape_a, actual_shape_b, actual_type_a, actual_type_b;
    transformf actual_X_ws_a, actual_X_ws_b;
    // unique ordering of shape pairs
    if (type_a < type_b) {
        actual_shape_a = shape_a;
        actual_shape_b = shape_b;
        actual_type_a = type_a;
        actual_type_b = type_b;
        actual_X_ws_a = X_ws_a;
        actual_X_ws_b = X_ws_b;
    } else {
        actual_shape_a = shape_b;
        actual_shape_b = shape_a;
        actual_type_a = type_b;
        actual_type_b = type_a;
        actual_X_ws_a = X_ws_b;
        actual_X_ws_b = X_ws_a;
    }

    auto p_a = transform_get_translation(actual_X_ws_a);
    if (actual_type_b == int(GeometryType::GEO_PLANE)) {
        if (actual_type_a == int(GeometryType::GEO_PLANE)) {
            return;
        }
        auto query_b = transform_point(transform_inverse(actual_X_ws_b), p_a);
        auto scale = geo.scale[actual_shape_b];
        auto closest = closest_point_plane(scale[0], scale[1], query_b);
        auto d = length(query_b - closest);
        auto r_a = collision_radius[actual_shape_a];
        if (d > r_a + rigid_contact_margin) {
            return;
        }
    } else {
        auto p_b = transform_get_translation(actual_X_ws_b);
        auto d = length(p_a - p_b) * 0.5 - 0.1;
        auto r_a = collision_radius[actual_shape_a];
        auto r_b = collision_radius[actual_shape_b];
        if (d > r_a + r_b + rigid_contact_margin) {
            return;
        }
    }

    // determine how many contact points need to be evaluated
    int num_contacts = 0;
    if (actual_type_a == int(GeometryType::GEO_SPHERE)) {
        num_contacts = 1;
    } else if (actual_type_a == int(GeometryType::GEO_CAPSULE)) {
        if (actual_type_b == int(GeometryType::GEO_PLANE)) {
            if (geo.scale[actual_shape_b][0] == 0.0 && geo.scale[actual_shape_b][1] == 0.0) {
                num_contacts = 2;  // vertex-based collision for infinite plane
            } else {
                num_contacts = 2 + 4;  // vertex-based collision + plane edges
            }
        } else if (actual_type_b == int(GeometryType::GEO_MESH)) {
            auto num_contacts_a = 2;
            auto mesh_b = mesh_get(geo.source[actual_shape_b]);
            auto num_contacts_b = mesh_b.points.shape[0];
            num_contacts = num_contacts_a + num_contacts_b;
            auto index = atomic_add(contact_count, 0, num_contacts);
            if (index + num_contacts - 1 >= rigid_contact_max) {
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.");
                return;
            }
            // allocate contact points from capsule A against mesh B
            for (int i = 0; i < num_contacts_a; i++) {
                contact_shape0[index + i] = actual_shape_a;
                contact_shape1[index + i] = actual_shape_b;
                contact_point_id[index + i] = i;
            }
            // allocate contact points from mesh B against capsule A
            for (int i = 0; i < num_contacts_b; i++) {
                contact_shape0[index + num_contacts_a + i] = actual_shape_b;
                contact_shape1[index + num_contacts_a + i] = actual_shape_a;
                contact_point_id[index + num_contacts_a + i] = i;
            }
            return;
        } else {
            num_contacts = 2;
        }
    } else if (actual_type_a == int(GeometryType::GEO_BOX)) {
        if (actual_type_b == int(GeometryType::GEO_BOX)) {
            auto index = atomic_add(contact_count, 0, 24);
            if (index + 23 >= rigid_contact_max) {
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.");
                return;
            }
            // allocate contact points from box A against B
            for (int i = 0; i < 12; i++) {
                contact_shape0[index + i] = shape_a;
                contact_shape1[index + i] = shape_b;
                contact_point_id[index + i] = i;
            }
            // allocate contact points from box B against A
            for (int i = 0; i < 12; i++) {
                contact_shape0[index + 12 + i] = shape_b;
                contact_shape1[index + 12 + i] = shape_a;
                contact_point_id[index + 12 + i] = i;
            }
            return;
        } else if (actual_type_b == int(GeometryType::GEO_MESH)) {
            auto num_contacts_a = 8;
            auto mesh_b = mesh_get(geo.source[actual_shape_b]);
            auto num_contacts_b = mesh_b.points.shape[0];
            num_contacts = num_contacts_a + num_contacts_b;
            auto index = atomic_add(contact_count, 0, num_contacts);
            if (index + num_contacts - 1 >= rigid_contact_max) {
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.");
                return;
            }
            // allocate contact points from box A against mesh B
            for (int i = 0; i < num_contacts_a; i++) {
                contact_shape0[index + i] = actual_shape_a;
                contact_shape1[index + i] = actual_shape_b;
                contact_point_id[index + i] = i;
            }
            // allocate contact points from mesh B against box A
            for (int i = 0; i < num_contacts_b; i++) {
                contact_shape0[index + num_contacts_a + i] = actual_shape_b;
                contact_shape1[index + num_contacts_a + i] = actual_shape_a;
                contact_point_id[index + num_contacts_a + i] = i;
            }
            return;
        } else if (actual_type_b == int(GeometryType::GEO_PLANE)) {
            if (geo.scale[actual_shape_b][0] == 0.0 && geo.scale[actual_shape_b][1] == 0.0) {
                num_contacts = 8;  // vertex - based collision;
            } else {
                num_contacts = 8 + 4;  // vertex - based collision + plane edges;
            }
        } else {
            num_contacts = 8;
        }
    } else if (actual_type_a == int(GeometryType::GEO_MESH)) {
        auto mesh_a = mesh_get(geo.source[actual_shape_a]);

        auto num_contacts_a = mesh_a.points.shape[0];
        auto num_contacts_b = 0;
        if (actual_type_b == int(GeometryType::GEO_MESH)) {
            auto mesh_b = mesh_get(geo.source[actual_shape_b]);
            num_contacts_b = mesh_b.points.shape[0];
        } else if (actual_type_b != int(GeometryType::GEO_PLANE)) {
            print("broadphase_collision_pairs: unsupported geometry type for mesh collision");
            return;
        }
        num_contacts = num_contacts_a + num_contacts_b;
        if (num_contacts > 0) {
            auto index = atomic_add(contact_count, 0, num_contacts);
            if (index + num_contacts - 1 >= rigid_contact_max) {
                print("Mesh contact: Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.");
                return;
            }
            // allocate contact points from mesh A against B
            for (int i = 0; i < num_contacts_a; i++) {
                contact_shape0[index + i] = actual_shape_a;
                contact_shape1[index + i] = actual_shape_b;
                contact_point_id[index + i] = i;
            }
            // allocate contact points from mesh B against A
            for (int i = 0; i < num_contacts_b; i++) {
                contact_shape0[index + num_contacts_a + i] = actual_shape_b;
                contact_shape1[index + num_contacts_a + i] = actual_shape_a;
                contact_point_id[index + num_contacts_a + i] = i;
            }
            return;
        }
    } else if (actual_type_a == int(GeometryType::GEO_PLANE)) {
        return;  // no plane-plane contacts
    } else {
        print("broadphase_collision_pairs: unsupported geometry type");
    }

    if (num_contacts > 0) {
        auto index = atomic_add(contact_count, 0, num_contacts);
        if (index + num_contacts - 1 >= rigid_contact_max) {
            print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.");
            return;
        }
        // allocate contact points
        for (int i = 0; i < num_contacts; i++) {
            contact_shape0[index + i] = actual_shape_a;
            contact_shape1[index + i] = actual_shape_b;
            contact_point_id[index + i] = i;
        }
    }
}

CUDA_CALLABLE_DEVICE void handle_contact_pairs(const array_t<transformf> &body_q,
                                               const array_t<transformf> &shape_X_bs,
                                               const array_t<int> &shape_body,
                                               const ModelShapeGeometry &geo,
                                               float rigid_contact_margin,
                                               const array_t<vec3f> &body_com,
                                               const array_t<int> &contact_shape0,
                                               const array_t<int> &contact_shape1,
                                               const array_t<int> &contact_point_id,
                                               const array_t<int> &rigid_contact_count,
                                               int edge_sdf_iter,
                                               array_t<int> &contact_body0,
                                               array_t<int> &contact_body1,
                                               array_t<vec3f> &contact_point0,
                                               array_t<vec3f> &contact_point1,
                                               array_t<vec3f> &contact_offset0,
                                               array_t<vec3f> &contact_offset1,
                                               array_t<vec3f> &contact_normal,
                                               array_t<float> &contact_thickness) {
    auto tid = wp::tid();
    if (tid >= rigid_contact_count[0]) {
        return;
    }
    auto shape_a = contact_shape0[tid];
    auto shape_b = contact_shape1[tid];
    if (shape_a == shape_b) {
        return;
    }
    auto point_id = contact_point_id[tid];

    auto rigid_a = shape_body[shape_a];
    auto X_wb_a = transform_identity();
    if (rigid_a >= 0) {
        X_wb_a = body_q[rigid_a];
    }
    auto X_bs_a = shape_X_bs[shape_a];
    auto X_ws_a = transform_multiply(X_wb_a, X_bs_a);
    auto X_sw_a = transform_inverse(X_ws_a);
    auto X_bw_a = transform_inverse(X_wb_a);
    auto geo_type_a = geo.type[shape_a];
    auto geo_scale_a = geo.scale[shape_a];
    auto min_scale_a = min(geo_scale_a);
    auto thickness_a = geo.thickness[shape_a];
    // is_solid_a = geo.is_solid[shape_a];

    auto rigid_b = shape_body[shape_b];
    auto X_wb_b = transform_identity();
    if (rigid_b >= 0) {
        X_wb_b = body_q[rigid_b];
    }
    auto X_bs_b = shape_X_bs[shape_b];
    auto X_ws_b = transform_multiply(X_wb_b, X_bs_b);
    auto X_sw_b = transform_inverse(X_ws_b);
    auto X_bw_b = transform_inverse(X_wb_b);
    auto geo_type_b = geo.type[shape_b];
    auto geo_scale_b = geo.scale[shape_b];
    auto min_scale_b = min(geo_scale_b);
    auto thickness_b = geo.thickness[shape_b];
    // is_solid_b = geo.is_solid[shape_b];

    // fill in contact rigid body ids
    contact_body0[tid] = rigid_a;
    contact_body1[tid] = rigid_b;

    float distance = 1.0e6;
    auto u = float(0.0);

    vec3f p_a_world, p_b_world, normal;
    if (geo_type_a == int(GeometryType::GEO_SPHERE)) {
        p_a_world = transform_get_translation(X_ws_a);
        if (geo_type_b == int(GeometryType::GEO_SPHERE)) {
            p_b_world = transform_get_translation(X_ws_b);
        } else if (geo_type_b == int(GeometryType::GEO_BOX)) {
            // contact point in frame of body B
            auto p_a_body = transform_point(X_sw_b, p_a_world);
            auto p_b_body = closest_point_box(geo_scale_b, p_a_body);
            p_b_world = transform_point(X_ws_b, p_b_body);
        } else if (geo_type_b == int(GeometryType::GEO_CAPSULE)) {
            auto half_height_b = geo_scale_b[1];
            // capsule B
            auto A_b = transform_point(X_ws_b, vec3(0.0, half_height_b, 0.0));
            auto B_b = transform_point(X_ws_b, vec3(0.0, -half_height_b, 0.0));
            p_b_world = closest_point_line_segment(A_b, B_b, p_a_world);
        } else if (geo_type_b == int(GeometryType::GEO_MESH)) {
            auto mesh_b = geo.source[shape_b];
            auto query_b_local = transform_point(X_sw_b, p_a_world);
            auto face_index = int(0);
            auto face_u = float(0.0);
            auto face_v = float(0.0);
            auto sign = float(0.0);
            auto max_dist = (thickness_a + thickness_b + rigid_contact_margin) / geo_scale_b[0];

            auto res = mesh_query_point_sign_normal(mesh_b, cw_div(query_b_local, geo_scale_b), max_dist, sign,
                                                    face_index, face_u, face_v);
            if (res) {
                auto shape_p = mesh_eval_position(mesh_b, face_index, face_u, face_v);
                shape_p = cw_mul(shape_p, geo_scale_b);
                p_b_world = transform_point(X_ws_b, shape_p);
            } else {
                contact_shape0[tid] = -1;
                contact_shape1[tid] = -1;
                return;
            }
        } else if (geo_type_b == int(GeometryType::GEO_PLANE)) {
            auto p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], transform_point(X_sw_b, p_a_world));
            p_b_world = transform_point(X_ws_b, p_b_body);
        } else {
            print("Unsupported geometry type in sphere collision handling");
            print(geo_type_b);
            return;
        }
        auto diff = p_a_world - p_b_world;
        normal = normalize(diff);
        distance = dot(diff, normal);
    } else if (geo_type_a == int(GeometryType::GEO_BOX) && geo_type_b == int(GeometryType::GEO_BOX)) {
        // edge-based box contact
        auto edge = get_box_edge(point_id, geo_scale_a);
        auto edge0_world = transform_point(X_ws_a, spatial_top(edge));
        auto edge1_world = transform_point(X_ws_a, spatial_bottom(edge));
        auto edge0_b = transform_point(X_sw_b, edge0_world);
        auto edge1_b = transform_point(X_sw_b, edge1_world);
        auto max_iter = edge_sdf_iter;
        u = closest_edge_coordinate_box(geo_scale_b, edge0_b, edge1_b, max_iter);
        p_a_world = (1.f - u) * edge0_world + u * edge1_world;

        // find the closest point + contact normal on box B
        auto query_b = transform_point(X_sw_b, p_a_world);
        auto p_b_body = closest_point_box(geo_scale_b, query_b);
        p_b_world = transform_point(X_ws_b, p_b_body);
        auto diff = p_a_world - p_b_world;
        // use center of box A to query normal to make sure we are not inside B
        query_b = transform_point(X_sw_b, transform_get_translation(X_ws_a));
        normal = transform_vector(X_ws_b, box_sdf_grad(geo_scale_b, query_b));
        distance = dot(diff, normal);
    } else if (geo_type_a == int(GeometryType::GEO_BOX) && geo_type_b == int(GeometryType::GEO_CAPSULE)) {
        auto half_height_b = geo_scale_b[1];
        // capsule B
        // depending on point id, we query an edge from 0 to 0.5 or 0.5 to 1
        auto e0 = vec3(0.0, -half_height_b * float(point_id % 2), 0.0);
        auto e1 = vec3(0.0, half_height_b * float((point_id + 1) % 2), 0.0);
        auto edge0_world = transform_point(X_ws_b, e0);
        auto edge1_world = transform_point(X_ws_b, e1);
        auto edge0_a = transform_point(X_sw_a, edge0_world);
        auto edge1_a = transform_point(X_sw_a, edge1_world);
        auto max_iter = edge_sdf_iter;
        u = closest_edge_coordinate_box(geo_scale_a, edge0_a, edge1_a, max_iter);
        p_b_world = (1.f - u) * edge0_world + u * edge1_world;
        // find the closest point + contact normal on box A
        auto query_a = transform_point(X_sw_a, p_b_world);
        auto p_a_body = closest_point_box(geo_scale_a, query_a);
        p_a_world = transform_point(X_ws_a, p_a_body);
        auto diff = p_a_world - p_b_world;
        // the contact point inside the capsule should already be outside the box
        normal = -transform_vector(X_ws_a, box_sdf_grad(geo_scale_a, query_a));
        distance = dot(diff, normal);
    } else if (geo_type_a == int(GeometryType::GEO_BOX) && geo_type_b == int(GeometryType::GEO_PLANE)) {
        auto plane_width = geo_scale_b[0];
        auto plane_length = geo_scale_b[1];
        if (point_id < 8) {
            // vertex-based contact
            auto p_a_body = get_box_vertex(point_id, geo_scale_a);
            p_a_world = transform_point(X_ws_a, p_a_body);
            auto query_b = transform_point(X_sw_b, p_a_world);
            auto p_b_body = closest_point_plane(plane_width, plane_length, query_b);
            p_b_world = transform_point(X_ws_b, p_b_body);
            auto diff = p_a_world - p_b_world;
            normal = transform_vector(X_ws_b, vec3(0.0, 1.0, 0.0));
            if (plane_width > 0.0 && plane_length > 0.0) {
                if (abs(query_b[0]) > plane_width || abs(query_b[2]) > plane_length) {
                    // skip, we will evaluate the plane edge contact with the box later
                    contact_shape0[tid] = -1;
                    contact_shape1[tid] = -1;
                    return;
                }
            }
            // the contact point is within plane boundaries
            distance = dot(diff, normal);
        } else {
            // contact between box A and edges of finite plane B
            auto edge = get_plane_edge(point_id - 8, plane_width, plane_length);
            auto edge0_world = transform_point(X_ws_b, spatial_top(edge));
            auto edge1_world = transform_point(X_ws_b, spatial_bottom(edge));
            auto edge0_a = transform_point(X_sw_a, edge0_world);
            auto edge1_a = transform_point(X_sw_a, edge1_world);
            auto max_iter = edge_sdf_iter;
            u = closest_edge_coordinate_box(geo_scale_a, edge0_a, edge1_a, max_iter);
            p_b_world = (1.f - u) * edge0_world + u * edge1_world;

            // find the closest point + contact normal on box A
            auto query_a = transform_point(X_sw_a, p_b_world);
            auto p_a_body = closest_point_box(geo_scale_a, query_a);
            p_a_world = transform_point(X_ws_a, p_a_body);
            auto query_b = transform_point(X_sw_b, p_a_world);
            if (abs(query_b[0]) > plane_width || abs(query_b[2]) > plane_length) {
                // ensure that the closest point is actually inside the plane
                contact_shape0[tid] = -1;
                contact_shape1[tid] = -1;
                return;
            }
            auto diff = p_a_world - p_b_world;
            auto com_a = transform_get_translation(X_ws_a);
            query_b = transform_point(X_sw_b, com_a);
            if (abs(query_b[0]) > plane_width || abs(query_b[2]) > plane_length) {
                // the COM is outside the plane
                normal = normalize(com_a - p_b_world);
            } else {
                normal = transform_vector(X_ws_b, vec3(0.0, 1.0, 0.0));
            }
            distance = dot(diff, normal);
        }
    } else if (geo_type_a == int(GeometryType::GEO_CAPSULE) && geo_type_b == int(GeometryType::GEO_CAPSULE)) {
        // find closest edge coordinate to capsule SDF B
        auto half_height_a = geo_scale_a[1];
        auto half_height_b = geo_scale_b[1];
        // edge from capsule A
        // depending on point id, we query an edge from 0 to 0.5 or 0.5 to 1
        auto e0 = vec3(0.0, half_height_a * float(point_id % 2), 0.0);
        auto e1 = vec3(0.0, -half_height_a * float((point_id + 1) % 2), 0.0);
        auto edge0_world = transform_point(X_ws_a, e0);
        auto edge1_world = transform_point(X_ws_a, e1);
        auto edge0_b = transform_point(X_sw_b, edge0_world);
        auto edge1_b = transform_point(X_sw_b, edge1_world);
        auto max_iter = edge_sdf_iter;
        u = closest_edge_coordinate_capsule(geo_scale_b[0], geo_scale_b[1], edge0_b, edge1_b, max_iter);
        p_a_world = (1.f - u) * edge0_world + u * edge1_world;
        auto p0_b_world = transform_point(X_ws_b, vec3(0.0, half_height_b, 0.0));
        auto p1_b_world = transform_point(X_ws_b, vec3(0.0, -half_height_b, 0.0));
        p_b_world = closest_point_line_segment(p0_b_world, p1_b_world, p_a_world);
        auto diff = p_a_world - p_b_world;
        normal = normalize(diff);
        distance = dot(diff, normal);
    } else if (geo_type_a == int(GeometryType::GEO_CAPSULE) && geo_type_b == int(GeometryType::GEO_MESH)) {
        // find the closest edge coordinate to mesh SDF B
        auto half_height_a = geo_scale_a[1];
        // edge from capsule A
        // depending on point id, we query an edge from -h to 0 or 0 to h
        auto e0 = vec3(0.0, -half_height_a * float(point_id % 2), 0.0);
        auto e1 = vec3(0.0, half_height_a * float((point_id + 1) % 2), 0.0);
        auto edge0_world = transform_point(X_ws_a, e0);
        auto edge1_world = transform_point(X_ws_a, e1);
        auto edge0_b = transform_point(X_sw_b, edge0_world);
        auto edge1_b = transform_point(X_sw_b, edge1_world);
        auto max_iter = edge_sdf_iter;
        auto max_dist = (rigid_contact_margin + thickness_a + thickness_b) / min_scale_b;
        auto mesh_b = geo.source[shape_b];
        u = closest_edge_coordinate_mesh(mesh_b, cw_div(edge0_b, geo_scale_b), cw_div(edge1_b, geo_scale_b), max_iter,
                                         max_dist);
        p_a_world = (1.f - u) * edge0_world + u * edge1_world;
        auto query_b_local = transform_point(X_sw_b, p_a_world);
        mesh_b = geo.source[shape_b];

        auto face_index = int(0);
        auto face_u = float(0.0);
        auto face_v = float(0.0);
        auto sign = float(0.0);

        auto res = mesh_query_point_sign_normal(mesh_b, cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index,
                                                face_u, face_v);

        if (res) {
            auto shape_p = mesh_eval_position(mesh_b, face_index, face_u, face_v);
            shape_p = cw_mul(shape_p, geo_scale_b);
            p_b_world = transform_point(X_ws_b, shape_p);
            p_a_world = closest_point_line_segment(edge0_world, edge1_world, p_b_world);
            // contact direction vector in world frame
            auto diff = p_a_world - p_b_world;
            normal = normalize(diff);
            distance = dot(diff, normal);
        } else {
            contact_shape0[tid] = -1;
            contact_shape1[tid] = -1;
            return;
        }
    } else if (geo_type_a == int(GeometryType::GEO_MESH) && geo_type_b == int(GeometryType::GEO_CAPSULE)) {
        //  vertex-based contact
        auto mesh = mesh_get(geo.source[shape_a]);
        auto body_a_pos = cw_mul(mesh.points[point_id], geo_scale_a);
        p_a_world = transform_point(X_ws_a, body_a_pos);
        // find closest point + contact normal on capsule B
        auto half_height_b = geo_scale_b[1];
        auto A_b = transform_point(X_ws_b, vec3(0.0, half_height_b, 0.0));
        auto B_b = transform_point(X_ws_b, vec3(0.0, -half_height_b, 0.0));
        p_b_world = closest_point_line_segment(A_b, B_b, p_a_world);
        auto diff = p_a_world - p_b_world;
        // this is more reliable in practice than using the SDF gradient
        normal = normalize(diff);
        distance = dot(diff, normal);
    } else if (geo_type_a == int(GeometryType::GEO_CAPSULE) && geo_type_b == int(GeometryType::GEO_PLANE)) {
        auto plane_width = geo_scale_b[0];
        auto plane_length = geo_scale_b[1];
        if (point_id < 2) {
            // vertex-based collision
            auto half_height_a = geo_scale_a[1];
            auto side = float(point_id) * 2.f - 1.f;
            p_a_world = transform_point(X_ws_a, vec3(0.0, side * half_height_a, 0.0));
            auto query_b = transform_point(X_sw_b, p_a_world);
            auto p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], query_b);
            p_b_world = transform_point(X_ws_b, p_b_body);
            auto diff = p_a_world - p_b_world;
            if (geo_scale_b[0] > 0.0 && geo_scale_b[1] > 0.0) {
                normal = normalize(diff);
            } else {
                normal = transform_vector(X_ws_b, vec3(0.0, 1.0, 0.0));
            }
            distance = dot(diff, normal);
        } else {
            // contact between capsule A and edges of finite plane B
            plane_width = geo_scale_b[0];
            plane_length = geo_scale_b[1];
            auto edge = get_plane_edge(point_id - 2, plane_width, plane_length);
            auto edge0_world = transform_point(X_ws_b, spatial_top(edge));
            auto edge1_world = transform_point(X_ws_b, spatial_bottom(edge));
            auto edge0_a = transform_point(X_sw_a, edge0_world);
            auto edge1_a = transform_point(X_sw_a, edge1_world);
            auto max_iter = edge_sdf_iter;
            u = closest_edge_coordinate_capsule(geo_scale_a[0], geo_scale_a[1], edge0_a, edge1_a, max_iter);
            p_b_world = (1.f - u) * edge0_world + u * edge1_world;

            // find closest point + contact normal on capsule A
            auto half_height_a = geo_scale_a[1];
            auto p0_a_world = transform_point(X_ws_a, vec3(0.0, half_height_a, 0.0));
            auto p1_a_world = transform_point(X_ws_a, vec3(0.0, -half_height_a, 0.0));
            p_a_world = closest_point_line_segment(p0_a_world, p1_a_world, p_b_world);
            auto diff = p_a_world - p_b_world;
            // normal = transform_vector(X_ws_b, vec3(0.0, 1.0, 0.0));
            normal = normalize(diff);
            distance = dot(diff, normal);
        }
    } else if (geo_type_a == int(GeometryType::GEO_MESH) && geo_type_b == int(GeometryType::GEO_BOX)) {
        // vertex-based contact
        auto mesh = mesh_get(geo.source[shape_a]);
        auto body_a_pos = cw_mul(mesh.points[point_id], geo_scale_a);
        p_a_world = transform_point(X_ws_a, body_a_pos);
        // find the closest point + contact normal on box B
        auto query_b = transform_point(X_sw_b, p_a_world);
        auto p_b_body = closest_point_box(geo_scale_b, query_b);
        p_b_world = transform_point(X_ws_b, p_b_body);
        auto diff = p_a_world - p_b_world;
        // this is more reliable in practice than using the SDF gradient
        normal = normalize(diff);
        if (box_sdf(geo_scale_b, query_b) < 0.0) {
            normal = -normal;
        }
        distance = dot(diff, normal);
    } else if (geo_type_a == int(GeometryType::GEO_BOX) && geo_type_b == int(GeometryType::GEO_MESH)) {
        // vertex-based contact
        auto query_a = get_box_vertex(point_id, geo_scale_a);
        p_a_world = transform_point(X_ws_a, query_a);
        auto query_b_local = transform_point(X_sw_b, p_a_world);
        auto mesh_b = geo.source[shape_b];
        auto max_dist = (rigid_contact_margin + thickness_a + thickness_b) / min_scale_b;
        auto face_index = int(0);
        auto face_u = float(0.0);
        auto face_v = float(0.0);
        auto sign = float(0.0);

        auto res = mesh_query_point_sign_normal(mesh_b, cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index,
                                                face_u, face_v);

        if (res) {
            auto shape_p = mesh_eval_position(mesh_b, face_index, face_u, face_v);
            shape_p = cw_mul(shape_p, geo_scale_b);
            p_b_world = transform_point(X_ws_b, shape_p);
            // contact direction vector in world frame;
            auto diff_b = p_a_world - p_b_world;
            normal = normalize(diff_b) * sign;
            distance = dot(diff_b, normal);
        } else {
            contact_shape0[tid] = -1;
            contact_shape1[tid] = -1;
            return;
        }
    } else if (geo_type_a == int(GeometryType::GEO_MESH) && geo_type_b == int(GeometryType::GEO_MESH)) {
        //    vertex-based contact
        auto mesh = mesh_get(geo.source[shape_a]);
        auto mesh_b = geo.source[shape_b];

        auto body_a_pos = cw_mul(mesh.points[point_id], geo_scale_a);
        p_a_world = transform_point(X_ws_a, body_a_pos);
        auto query_b_local = transform_point(X_sw_b, p_a_world);

        auto face_index = int(0);
        auto face_u = float(0.0);
        auto face_v = float(0.0);
        auto sign = float(0.0);
        auto min_scale = min(min_scale_a, min_scale_b);
        auto max_dist = (rigid_contact_margin + thickness_a + thickness_b) / min_scale;

        auto res = mesh_query_point_sign_normal(mesh_b, cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index,
                                                face_u, face_v);

        if (res) {
            auto shape_p = mesh_eval_position(mesh_b, face_index, face_u, face_v);
            shape_p = cw_mul(shape_p, geo_scale_b);
            p_b_world = transform_point(X_ws_b, shape_p);
            // contact direction vector in world frame
            auto diff_b = p_a_world - p_b_world;
            normal = normalize(diff_b) * sign;
            distance = dot(diff_b, normal);
        } else {
            contact_shape0[tid] = -1;
            contact_shape1[tid] = -1;
            return;
        }
    } else if (geo_type_a == int(GeometryType::GEO_MESH) && geo_type_b == int(GeometryType::GEO_PLANE)) {
        // vertex-based contact
        auto mesh = mesh_get(geo.source[shape_a]);
        auto body_a_pos = cw_mul(mesh.points[point_id], geo_scale_a);
        p_a_world = transform_point(X_ws_a, body_a_pos);
        auto query_b = transform_point(X_sw_b, p_a_world);
        auto p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], query_b);
        p_b_world = transform_point(X_ws_b, p_b_body);
        auto diff = p_a_world - p_b_world;
        normal = transform_vector(X_ws_b, vec3(0.0, 1.0, 0.0));
        distance = length(diff);

        // if the plane is infinite or the point is within the plane we fix the normal to prevent intersections
        if (geo_scale_b[0] == 0.0 && geo_scale_b[1] == 0.0 ||
            abs(query_b[0]) < geo_scale_b[0] && abs(query_b[2]) < geo_scale_b[1]) {
            normal = transform_vector(X_ws_b, vec3(0.0, 1.0, 0.0));
        } else {
            normal = normalize(diff);
        }
        distance = dot(diff, normal);
        // ignore extreme penetrations (e.g. when mesh is below the plane)
        if (distance < -rigid_contact_margin) {
            contact_shape0[tid] = -1;
            contact_shape1[tid] = -1;
            return;
        }
    } else {
        print("Unsupported geometry pair in collision handling");
        return;
    }

    auto thickness = thickness_a + thickness_b;
    auto d = distance - thickness;
    if (d < rigid_contact_margin) {
        // transform from world into body frame (so the contact point includes the shape transform);
        contact_point0[tid] = transform_point(X_bw_a, p_a_world);
        contact_point1[tid] = transform_point(X_bw_b, p_b_world);
        contact_offset0[tid] = transform_vector(X_bw_a, -thickness_a * normal);
        contact_offset1[tid] = transform_vector(X_bw_b, thickness_b * normal);
        contact_normal[tid] = normal;
        contact_thickness[tid] = thickness;
        // printf("distance: %f\tnormal: %.3f %.3f %.3f\tp_a_world: %.3f %.3f %.3f\tp_b_world: %.3f %.3f %.3f\n",
        // distance, normal[0], normal[1], normal[2], p_a_world[0], p_a_world[1], p_a_world[2], p_b_world[0],
        // p_b_world[1], p_b_world[2])
    } else {
        contact_shape0[tid] = -1;
        contact_shape1[tid] = -1;
    }
}

void collide(Model &model, State &state, int edge_sdf_iter) {}
}  // namespace wp