#include "geometry.h"
#include "mesh.h"
#include "template_geometry.h"
#include "surface_integrator.h"
#include "volume_integrator.h"
#include "host/poly_info_1d_host.h"
#include "host/poly_info_2d_host.h"
#include "host/poly_info_3d_host.h"
#include "host/grid_system_data_host.h"
#include <iostream>
#include "grid.h"

namespace wp::fields {
void test() {
    for (auto &pnt : template_geometry_t<Interval, 6>::quadrature_info().pnts.data) {
        std::cout << pnt[0] << std::endl;
    }
    mesh_t<1, 1> mesh{};
    {
        Grid<Interval> grid;
        PolyInfo<Interval, 1> poly{nullptr};
        poly_info_t<Interval, 1>::AverageBasisFuncFunctor average_basis_func_functor{grid.grid_handle, poly.handle};
        poly_info_t<Interval, 1>::UpdateLSMatrixFunctor update_ls_matrix_functor{grid.grid_handle, poly.handle};
        poly_info_t<Interval, 1>::FuncValueFunctor func_value_functor{};
        poly_info_t<Interval, 1>::FuncGradientFunctor func_gradient_functor{};
    }
    {
        Grid<Triangle> grid;
        PolyInfo<Triangle, 1> poly{nullptr};
        poly_info_t<Triangle, 1>::AverageBasisFuncFunctor average_basis_func_functor{grid.grid_handle, poly.handle};
        poly_info_t<Triangle, 1>::UpdateLSMatrixFunctor update_ls_matrix_functor{grid.grid_handle, poly.handle};
        poly_info_t<Triangle, 1>::FuncValueFunctor func_value_functor{};
        poly_info_t<Triangle, 1>::FuncGradientFunctor func_gradient_functor{};
    }
    {
        Grid<Tetrahedron> grid;
        PolyInfo<Tetrahedron, 1> poly{nullptr};
        poly_info_t<Tetrahedron, 1>::AverageBasisFuncFunctor average_basis_func_functor{grid.grid_handle, poly.handle};
        poly_info_t<Tetrahedron, 1>::UpdateLSMatrixFunctor update_ls_matrix_functor{grid.grid_handle, poly.handle};
        poly_info_t<Tetrahedron, 1>::FuncValueFunctor func_value_functor{};
        poly_info_t<Tetrahedron, 1>::FuncGradientFunctor func_gradient_functor{};
    }
    grid_system_data_t<Interval, 1, 1> system_data;
}

void test2() {
    struct IntegratorFunctor {
        using RETURN_TYPE = float;
        CUDA_CALLABLE float operator()(vec_t<1, float> pt) {
            return pt[0];
        }
    };

    SurfaceIntegrator<Interval, 2> integrator;
    IntegratorFunctor functor;
    integrator(0, functor);
}

void test3() {
    struct IntegratorFunctor {
        using RETURN_TYPE = float;
        CUDA_CALLABLE float operator()(vec_t<2, float> pt) {
            return pt[0];
        }
    };

    SurfaceIntegrator<IntervalTo2D, 2> integrator;
    IntegratorFunctor functor;
    integrator(0, functor);
}

void test4() {
    struct IntegratorFunctor {
        using RETURN_TYPE = float;
        CUDA_CALLABLE float operator()(vec_t<2, float> pt) {
            return pt[0];
        }
    };

    VolumeIntegrator<Triangle, 2> integrator;
    IntegratorFunctor functor;
    integrator(0, functor);
}

}// namespace wp::fields
