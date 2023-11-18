//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/mat.h"
#include "poly_info.h"
#include "grid.h"

namespace wp::fields {

template<uint32_t ORDER>
struct poly_info_t<Interval, ORDER> {
    static constexpr uint32_t dim = Interval::dim;
    static constexpr uint32_t order = ORDER;
    static constexpr uint32_t n_unknown = order;
    array_t<fixed_array_t<float, n_unknown>> poly_constants;
    using grid_t = grid_t<Interval>;
    using point_t = grid_t::point_t;
    using Mat = mat_t<n_unknown, n_unknown, float>;
    using Vec = mat_t<n_unknown, 1, float>;

    struct AverageBasisFuncFunctor {
        CUDA_CALLABLE AverageBasisFuncFunctor(const grid_t &grid, const poly_info_t<Interval, order> poly) {
            bary_center = grid.bary_center;
            bary_size = grid.size;
            poly_constants = poly.poly_constants;
        }

        CUDA_CALLABLE void operator()(int basisIdx, int32_t quadIdx, fixed_array_t<float, n_unknown> &result) {
            auto center = bary_center(basisIdx);
            auto pl = point(geo_view.vertex(quadIdx, 0));
            auto pr = point(geo_view.vertex(quadIdx, 1));
            for (int m = 1; m <= order; ++m) {
                float J_i;

                J_i = pow(pr[0] - center[0], float(m) + 1);
                J_i -= pow(pl[0] - center[0], float(m) + 1);
                J_i /= float(m + 1);
                J_i /= bary_size(quadIdx);

                result[m - 1] = (J_i - poly_constants(basisIdx)[m - 1]);
                result[m - 1] *= pow(bary_size(basisIdx), 1.f - float(m));
            }
        }

        CUDA_CALLABLE void operator()(uint32_t basisIdx, array_t<int32_t> patch,
                                      array_t<fixed_array_t<float, n_unknown>> result) {
            fixed_array_t<float, n_unknown> s;
            for (int j = 0; j < patch.shape.size(); ++j) {
                operator()(basisIdx, patch[j], s);
                result[j] = s;
            }
        }

    private:
        array_t<point_t> point;
        geometry_t geo_view;

        array_t<point_t> bary_center;
        array_t<float> bary_size;

        array_t<fixed_array_t<float, n_unknown>> poly_constants;
    };

    struct UpdateLSMatrixFunctor {
        CUDA_CALLABLE UpdateLSMatrixFunctor(const grid_t &grid, const poly_info_t<Interval, order> poly)
            : averageBasisFunc(grid, poly) {}

        CUDA_CALLABLE void operator()(size_t basisIdx, const array_t<int32_t> &patch,
                                      array_t<fixed_array_t<float, n_unknown>> poly_avgs, Mat *G) {
            averageBasisFunc(basisIdx, patch, poly_avgs);

//            G[0].fill(0.0);
            for (size_t j = 0; j < patch.shape.size(); ++j) {
                fixed_array_t<float, n_unknown> poly_avg = poly_avgs[j];
                for (int t1 = 0; t1 < n_unknown; ++t1) {
                    for (int t2 = 0; t2 < n_unknown; ++t2) {
//                        G[0](t1, t2) += poly_avg[t1] * poly_avg[t2];
                    }
                }
            }
        }

    private:
        AverageBasisFuncFunctor averageBasisFunc;
    };

    struct FuncValueFunctor {
        CUDA_CALLABLE FuncValueFunctor(const grid_t &grid, const poly_info_t<Interval, order> poly) {
            bary_center = grid.bary_center;
            bary_size = grid.bry_size;
        }

        CUDA_CALLABLE void basis_function_value(int idx, const point_t &coord,
                                                fixed_array_t<float, n_unknown> &result) {
            point_t cr = coord;
            cr -= bary_center(idx);
            for (int m = 1; m <= order; ++m) {
                result[m - 1] = pow(cr[0], float(m)) - poly_constants[idx][m - 1];
                result[m - 1] *= pow(bary_size(idx), 1.f - float(m));
            }
        }

        CUDA_CALLABLE float operator()(int idx, const point_t &coord, float avg, const Vec &para) {
            fixed_array_t<float, n_unknown> aa;
            basis_function_value(idx, coord, aa);

            float temp = 0.0;
            float result = avg;
            for (int t = 0; t < n_unknown; ++t) {
                temp = para[t] * aa[t];
                result += temp;
            }
            return result;
        }

    private:
        array_t<point_t> bary_center;
        array_t<float> bary_size;

        array_t<fixed_array_t<float, n_unknown>> poly_constants;
    };

    struct FuncGradientFunctor {
        CUDA_CALLABLE explicit FuncGradientFunctor(const grid_t &grid) {
            bary_center = grid.bary_center;
            bary_size = grid.size;
        }

        CUDA_CALLABLE void basis_function_gradient(int idx, const point_t &coord,
                                                   fixed_array_t<fixed_array_t<float, n_unknown>, 1> &result) {
            point_t cr = coord;
            cr -= bary_center(idx);
            for (int m = 1; m <= order; ++m) {
                result[0][m - 1] = m * std::pow(cr[0], m - 1);
                result[0][m - 1] *= std::pow(bary_size(idx), 1 - m);
            }
        }

        CUDA_CALLABLE fixed_array_t<float, 1> operator()(int idx, const point_t &coord, const Vec &para) {
            fixed_array_t<fixed_array_t<float, n_unknown>, 1> aa;
            basis_function_gradient(idx, coord, aa);

            float temp = 0.0;
            fixed_array_t<float, 1> result;
            for (int i = 0; i < dim; ++i) {
                for (int t = 0; t < n_unknown; ++t) {
                    temp = para[t] * aa[i][t];
                    result[i] += temp;
                }
            }

            return result;
        }

    private:
        array_t<point_t> bary_center;
        array_t<float> bary_size;
    };
};

}// namespace wp::fields