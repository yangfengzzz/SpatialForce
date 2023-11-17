//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <utility>

#include "core/mat.h"
#include "poly_info.h"
#include "grid.h"

namespace wp::fields {

template<uint32_t ORDER>
struct poly_info_t<Tetrahedron, ORDER> {
    static constexpr uint32_t dim = Triangle::dim;
    static constexpr uint32_t order = ORDER;
    static constexpr uint32_t n_unknown = (order + 3) * (order + 2) * (order + 1) / (3 * 2) - 1;
    array_t<fixed_array_t<float, n_unknown>> poly_constants;
    using grid_t = grid_t<Tetrahedron>;
    using point_t = grid_t::point_t;
    using Mat = mat_t<n_unknown, n_unknown, float>;
    using Vec = mat_t<n_unknown, 1, float>;

    struct AverageBasisFuncFunctor {
        CUDA_CALLABLE AverageBasisFuncFunctor(grid_t grid, poly_info_t<Tetrahedron, order> poly)
            : grid(std::move(grid)), poly_constants(poly.poly_constants) {
        }

        struct IntegratorFunctor {
            CUDA_CALLABLE IntegratorFunctor(int i, int j, int k, point_t bc)
                : i(i), j(j), k(k), bc(bc) {}

            using RETURN_TYPE = float;
            CUDA_CALLABLE float operator()(point_t pt) {
                return pow(pt[0] - bc[0], i) * pow(pt[1] - bc[1], j) * pow(pt[2] - bc[2], k);
            }

            point_t bc;
            int i{};
            int j{};
            int k{};
        };

        /// Element average of basis function
        /// @param basisIdx the loc of basis function
        /// @param quadIdx the loc of quadrature area
        /// @param result result
        CUDA_CALLABLE void operator()(uint32_t basisIdx, int32_t quadIdx, fixed_array_t<float, n_unknown> &result) {
            point_t bc = grid.bary_center(basisIdx);
            int index = 0;
            float J0 = 0;
            for (int m = 1; m <= order; ++m) {
                for (int i = 0; i <= m; ++i) {
                    for (int j = 0; j <= m - i; ++j) {
                        int k = m - i - j;
                        J0 = grid.volume_integrator(quadIdx, IntegratorFunctor(i, j, k, bc));
                        J0 /= grid.volume(quadIdx);
                        J0 -= poly_constants(basisIdx)[index];
                        J0 *= pow(grid.size(basisIdx), (1.f - float(m)) / float(dim - 1));
                        result[index] = J0;
                        index++;
                    }
                }
            }
        }

        CUDA_CALLABLE void operator()(uint32_t basisIdx, array_t<int32_t> patch,
                                      array_t<fixed_array_t<float, n_unknown>> result) {
            fixed_array_t<float, n_unknown> s;
            for (uint32_t j = 0; j < patch.shape.size(); ++j) {
                operator()(basisIdx, patch[j], s);
                result[j] = s;
            }
        }

    private:
        grid_t grid;
        array_t<fixed_array_t<float, n_unknown>> poly_constants;
    };

    struct UpdateLSMatrixFunctor {
        CUDA_CALLABLE UpdateLSMatrixFunctor(grid_t grid, poly_info_t<Tetrahedron, order> poly)
            : averageBasisFunc(grid, poly) {}

        CUDA_CALLABLE void operator()(uint32_t basisIdx, const array_t<int32_t> &patch,
                                      array_t<fixed_array_t<float, n_unknown>> poly_avgs, Mat *G) {
            averageBasisFunc(basisIdx, patch, poly_avgs);

            G[0].fill(0.0);
            for (uint32_t j = 0; j < patch.shape.size(); ++j) {
                fixed_array_t<float, n_unknown> poly_avg = poly_avgs[j];
                for (int t1 = 0; t1 < n_unknown; ++t1) {
                    for (int t2 = 0; t2 < n_unknown; ++t2) {
                        G[0](t1, t2) += poly_avg[t1] * poly_avg[t2];
                    }
                }
            }
        }

    private:
        AverageBasisFuncFunctor averageBasisFunc;
    };

    struct FuncValueFunctor {
        CUDA_CALLABLE FuncValueFunctor(const grid_t &grid, poly_info_t<Tetrahedron, order> poly)
            : poly_constants(poly.poly_constants), bary_center(grid.bary_center), bary_size(grid.bry_size) {
        }

        CUDA_CALLABLE double operator()(uint32_t idx, const point_t &coord, const float &avg, const Vec &para) {
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

        CUDA_CALLABLE void basis_function_value(uint32_t idx, const point_t &coord,
                                                fixed_array_t<float, n_unknown> &result) {
            point_t cr = coord;
            cr -= bary_center(idx);
            int index = 0;
            float J0 = 0;
            for (int m = 1; m <= order; ++m)
                for (int i = 0; i <= m; ++i) {
                    for (int k = 0; k <= m - i; ++k) {
                        int j = m - i - k;

                        J0 = pow(cr[0], float(i)) * pow(cr[1], float(k)) * pow(cr[2], float(j));
                        J0 -= poly_constants(idx)[index];
                        J0 *= pow(bary_size(idx), (1.f - float(m)) / float(dim - 1));
                        result[index] = J0;
                        index++;
                    }
                }
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

        CUDA_CALLABLE fixed_array_t<float, dim> operator()(uint32_t idx, const point_t &coord,
                                                           const Vec &para) {
            fixed_array_t<fixed_array_t<float, n_unknown>, dim> aa;
            basis_function_gradient(idx, coord, aa);

            float temp = 0.0;
            fixed_array_t<float, dim> result;
            for (uint32_t i = 0; i < dim; ++i) {
                for (int t = 0; t < n_unknown; ++t) {
                    temp = para[t] * aa[i][t];
                    result[i] += temp;
                }
            }

            return result;
        }

        CUDA_CALLABLE void basis_function_gradient(uint32_t idx, const point_t &coord,
                                                   fixed_array_t<fixed_array_t<float, n_unknown>, dim> &result) {
            Grid3D::point_t cr = coord;
            cr -= bary_center(idx);

            // coordinate x
            int index = 0;
            double J0 = 0.0;
            for (int m = 1; m <= order; ++m) {
                for (int i = 0; i <= m; ++i) {
                    for (int k = 0; k <= m - i; ++k) {
                        int j = m - i - k;
                        if (i == 0) {
                            J0 = 0.0;
                        } else {
                            J0 = i * pow(cr[0], i - 1) * pow(cr[1], k) * pow(cr[2], j);
                        }

                        J0 *= pow(bary_size(idx), (1.f - float(m)) / float(dim - 1));
                        result[0][index] = J0;
                        index++;
                    }
                }
            }

            // coordinate y
            index = 0;
            J0 = 0.0;
            for (int m = 1; m <= order; ++m) {
                for (int i = 0; i <= m; ++i) {
                    for (int k = 0; k <= m - i; ++k) {
                        int j = m - i - k;
                        if (k == 0) {
                            J0 = 0.0;
                        } else {
                            J0 = pow(cr[0], i) * k * pow(cr[1], k - 1) * pow(cr[2], j);
                        }
                        J0 *= pow(bary_size(idx), (1.f - float(m)) / float(dim - 1));
                        result[1][index] = J0;
                        index++;
                    }
                }
            }

            // coordinate z
            index = 0;
            J0 = 0.0;
            for (int m = 1; m <= order; ++m) {
                for (int i = 0; i <= m; ++i) {
                    for (int k = 0; k <= m - i; ++k) {
                        int j = m - i - k;
                        if (j == 0) {
                            J0 = 0.0;
                        } else {
                            J0 = pow(cr[0], i) * pow(cr[1], k) * j * pow(cr[2], j - 1);
                        }
                        J0 *= pow(bary_size(idx), (1.f - float(m)) / float(dim - 1));
                        result[2][index] = J0;
                        index++;
                    }
                }
            }
        }

    private:
        array_t<point_t> bary_center;
        array_t<float> bary_size;
    };
};

}// namespace wp::fields