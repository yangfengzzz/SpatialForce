//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "inertia.h"

namespace wp {
CUDA_CALLABLE_DEVICE float triangle_inertia(vec3f p, vec3f q, vec3f r,
                                            float density, vec3f com,
                                            array_t<float> &mass,
                                            array_t<mat33f> &inertia) {
    auto pcom = p - com;
    auto qcom = q - com;
    auto rcom = r - com;

    auto Dm = mat33(pcom[0], qcom[0], rcom[0],
                    pcom[1], qcom[1], rcom[1],
                    pcom[2], qcom[2], rcom[2]);

    auto volume = determinant(Dm) / 6.f;

    // accumulate mass
    atomic_add(mass, 0, 4.f * density * volume);

    auto alpha = sqrt(5.f) / 5.f;
    auto mid = (com + p + q + r) / 4.f;
    auto off_mid = mid - com;

    // displacement of quadrature point from COM
    auto d0 = alpha * (p - mid) + off_mid;
    auto d1 = alpha * (q - mid) + off_mid;
    auto d2 = alpha * (r - mid) + off_mid;
    auto d3 = alpha * (com - mid) + off_mid;

    // accumulate inertia
    auto identity = mat33(1.f, 0.f, 0.f,
                          0.f, 1.f, 0.f,
                          0.f, 0.f, 1.f);
    auto I = dot(d0, d0) * identity - outer(d0, d0);
    I += dot(d1, d1) * identity - outer(d1, d1);
    I += dot(d2, d2) * identity - outer(d2, d2);
    I += dot(d3, d3) * identity - outer(d3, d3);

    atomic_add(inertia, 0, (density * volume) * I);

    return volume;
}

__global__ void compute_solid_mesh_inertia(vec3f com, float weight,
                                                     const array_t<int> &indices,
                                                     const array_t<vec3f> &vertices,
                                                     array_t<float> &mass,
                                                     array_t<mat33f> &inertia,
                                                     array_t<float> &volume) {
    auto i = tid();

    auto p = vertices[indices[i * 3 + 0]];
    auto q = vertices[indices[i * 3 + 1]];
    auto r = vertices[indices[i * 3 + 2]];

    auto vol = triangle_inertia(p, q, r, weight, com, mass, inertia);
    atomic_add(volume, 0, vol);
}

__global__ void compute_hollow_mesh_inertia(vec3f com, float density,
                                                      const array_t<int> &indices,
                                                      const array_t<vec3f> &vertices,
                                                      array_t<float> thickness,
                                                      array_t<float> mass,
                                                      array_t<mat33f> inertia,
                                                      array_t<float> volume) {
    auto tid = wp::tid();
    auto i = indices[tid * 3 + 0];
    auto j = indices[tid * 3 + 1];
    auto k = indices[tid * 3 + 2];

    auto vi = vertices[i];
    auto vj = vertices[j];
    auto vk = vertices[k];

    auto normal = -normalize(cross(vj - vi, vk - vi));
    auto ti = normal * thickness[i];
    auto tj = normal * thickness[j];
    auto tk = normal * thickness[k];

    // wedge vertices
    auto vi0 = vi - ti;
    auto vi1 = vi + ti;
    auto vj0 = vj - tj;
    auto vj1 = vj + tj;
    auto vk0 = vk - tk;
    auto vk1 = vk + tk;

    triangle_inertia(vi0, vj0, vk0, density, com, mass, inertia);
    triangle_inertia(vj0, vk1, vk0, density, com, mass, inertia);
    triangle_inertia(vj0, vj1, vk1, density, com, mass, inertia);
    triangle_inertia(vj0, vi1, vj1, density, com, mass, inertia);
    triangle_inertia(vj0, vi0, vi1, density, com, mass, inertia);
    triangle_inertia(vj1, vi1, vk1, density, com, mass, inertia);
    triangle_inertia(vi1, vi0, vk0, density, com, mass, inertia);
    triangle_inertia(vi1, vk0, vk1, density, com, mass, inertia);

    // compute volume
    auto a = length(cross(vj - vi, vk - vi)) * 0.5f;
    auto vol = a * (thickness[i] + thickness[j] + thickness[k]) / 3.f;
    atomic_add(volume, 0, vol);
}

void compute_sphere_inertia(float density, float r) {}

void compute_capsule_inertia(float density, float r, float h) {}

void compute_cylinder_inertia(float density, float r, float h) {}

void compute_cone_inertia(float density, float r, float h) {}

void compute_box_inertia(float density, float w, float h, float d) {}

void compute_mesh_inertia(float density, const std::vector<float> &vertices,
                          const std::vector<float> &indices, bool is_solid,
                          const std::vector<float> &thickness) {}

void transform_inertia(float m, float I, float p, float q) {}
}// namespace wp