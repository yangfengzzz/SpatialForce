#include "geometry.h"
#include "mesh.h"
#include "template_geometry.h"
#include "surface_integrator.h"
#include "volume_integrator.h"
#include <iostream>

namespace wp::fields {
void test() {
    for (auto &pnt : template_geometry_t<Interval, 6>::quadrature_info().pnts.data) {
        std::cout << pnt[0] << std::endl;
    }
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
