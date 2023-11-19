//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "fields/io/io_mesh_1d.h"
#include "fields/io/gmsh2d_io.h"
#include "fields/volume_integrator.h"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

using namespace wp::fields;

namespace {
template<typename TYPE>
struct VolumeIntegratorFunctor {
    CUDA_CALLABLE VolumeIntegratorFunctor(const mesh_t<TYPE::dim, TYPE::dim> &mesh, wp::array_t<float> output)
        : output(output) {
        integrator.mesh = mesh;
    }

    struct IntegratorFunctor {
        using RETURN_TYPE = float;
        CUDA_CALLABLE float operator()(wp::vec_t<TYPE::dim, float> pt) {
            return 1.f;
        }
    };

    inline CUDA_CALLABLE void operator()(size_t i) {
        output[i] = integrator(i, IntegratorFunctor());
    }

private:
    wp::array_t<float> output;
    VolumeIntegrator<TYPE, 2> integrator;
};
}// namespace

TEST(VolumeIntegratorTest, 1D) {
    constexpr uint32_t dim = 1;

    IOMesh1D loader(0, 1, 100);
    auto mesh = loader.create_mesh();
    mesh.sync_h2d();

    std::vector<float> h_result(mesh.n_geometry(dim));
    auto d_result = wp::alloc_array(h_result);
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(mesh.n_geometry(dim)),
                     VolumeIntegratorFunctor<Interval>(mesh.handle, d_result));

    wp::copy_array_d2h(d_result, h_result);
    for (int i = 0; i < mesh.n_geometry(dim); i++) {
        EXPECT_NEAR(h_result[i], 0.01, 1.0e-7);
    }
}

TEST(VolumeIntegratorTest, 2D) {
    constexpr uint32_t dim = 2;

    GmshMesh2D loader("grids/2d/diagsquare.msh");
    auto mesh = loader.create_mesh();
    mesh.sync_h2d();

    std::vector<float> h_result(mesh.n_geometry(dim));
    auto d_result = wp::alloc_array(h_result);
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(mesh.n_geometry(dim)),
                     VolumeIntegratorFunctor<Triangle>(mesh.handle, d_result));

    wp::copy_array_d2h(d_result, h_result);
    for (int i = 0; i < mesh.n_geometry(dim); i++) {
        EXPECT_NEAR(h_result[i], 0.005, 1.0e-7);
    }
}