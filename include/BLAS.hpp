#ifndef ALGEFLOW_BLAS_HPP
#define ALGEFLOW_BLAS_HPP

#include <cstddef>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <span>
#include <mdspan>

#include "cscview.h"

namespace AlgeFlow {

using std::size_t;

// =============================================================================
// Matrix & Vector View Aliases
// =============================================================================

template<typename T>
using Mat2DColView = std::mdspan<T, std::dextents<std::size_t, 2>, std::layout_left>;

template<typename T>
using Mat2DColViewConst = std::mdspan<const T, std::dextents<std::size_t, 2>, std::layout_left>;

template<typename T>
using VecView = std::span<T>;

template<typename T>
using VecViewConst = std::span<const T>;

// =============================================================================
// BLAS Level 1: Vector-Vector Operations
// =============================================================================

// fill: x[:] = val
template<typename T>
constexpr void fill(VecView<T> x, T val) {
    std::fill(x.begin(), x.end(), val);
}

// copy: y = x
template<typename T>
constexpr void copy(VecViewConst<T> x, VecView<T> y) {
    assert(x.size() == y.size() && "Vector dimensions mismatch in copy");
    std::copy(x.begin(), x.end(), y.begin());
}

// scal: x = alpha * x
template<typename T>
constexpr void scal(T alpha, VecView<T> x) {
    if (alpha == T(0.0)) {
        std::fill(x.begin(), x.end(), T(0.0));
    } else if (alpha != T(1.0)) {
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] *= alpha;
        }
    }
}

// axpy: y = alpha * x + y
template<typename T>
constexpr void axpy(T alpha, VecViewConst<T> x, VecView<T> y) {
    assert(x.size() == y.size() && "Vector dimensions mismatch in axpy");
    if (alpha == T(0.0)) return;
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

// dot: result = x^T * y
template<typename T>
constexpr T dot(VecViewConst<T> x, VecViewConst<T> y) {
    assert(x.size() == y.size() && "Vector dimensions mismatch in dot");
    T sum = T(0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

// nrm2: Euclidean norm (L2 norm) = sqrt(x^T * x)
template<typename T>
inline T nrm2(VecViewConst<T> x) {
    T sum = T(0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}

// asum: L1 norm = sum(|x_i|)
template<typename T>
inline T asum(VecViewConst<T> x) {
    T sum = T(0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        sum += std::abs(x[i]);
    }
    return sum;
}

// iamax: index of element with maximum absolute value: argmax(|x_i|)
template<typename T>
inline size_t iamax(VecViewConst<T> x) {
    if (x.empty()) return 0;
    size_t max_idx = 0;
    T max_val = std::abs(x[0]);
    for (size_t i = 1; i < x.size(); ++i) {
        T abs_val = std::abs(x[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
            max_idx = i;
        }
    }
    return max_idx;
}

// =============================================================================
// BLAS Level 2: Matrix-Vector Operations
// =============================================================================

// Dense GEMV: General Matrix-Vector Multiplication
// y = alpha * A * x + beta * y
// Column-first optimized: AXPY mode, Column vector linear combination
template<typename T>
constexpr void gemv(Mat2DColViewConst<T> A, VecViewConst<T> x, VecView<T> y, 
                    T alpha = T(1.0), T beta = T(0.0)) {
    size_t Rows = A.extent(0);
    size_t Cols = A.extent(1);

    assert(x.size() == Cols && "Vector x dimension mismatch with matrix columns");
    assert(y.size() == Rows && "Vector y dimension mismatch with matrix rows");

    // beta * y
    if (beta == T(0.0)) {
        std::fill(y.begin(), y.end(), T(0.0));
    } else if (beta != T(1.0)) {
        for (size_t idx = 0; idx < Rows; ++idx) {
            y[idx] = beta * y[idx];
        }
    }
    
    if (alpha == T(0.0)) return;
    for (size_t idxC = 0; idxC < Cols; ++idxC) {
        T alphaXc = alpha * x[idxC];    
        for (size_t idxR = 0; idxR < Rows; ++idxR) {
            y[idxR] += alphaXc * A[idxR, idxC]; 
        }
    }
}

// Dense GEMV Transpose: General Matrix-Transpose-Vector Multiplication
// y = alpha * A^T * x + beta * y
// Dot product mode along columns of A
template<typename T>
constexpr void gemvT(Mat2DColViewConst<T> A, VecViewConst<T> x, VecView<T> y,
                     T alpha = T(1.0), T beta = T(0.0)) {
    size_t Rows = A.extent(0);
    size_t Cols = A.extent(1);

    assert(x.size() == Rows && "Vector x dimension mismatch with matrix rows");
    assert(y.size() == Cols && "Vector y dimension mismatch with matrix columns");

    // beta * y 
    if (beta == T(0.0)) {
        std::fill(y.begin(), y.end(), T(0.0));
    } else if (beta != T(1.0)) {
        for (size_t idx = 0; idx < Cols; ++idx) {
            y[idx] = beta * y[idx];
        }
    }

    if (alpha == T(0.0)) return;
    for (size_t idxC = 0; idxC < Cols; ++idxC) {
        T dot_val = T(0.0);
        for (size_t idxR = 0; idxR < Rows; ++idxR) {
            dot_val += x[idxR] * A[idxR, idxC];
        }
        y[idxC] += alpha * dot_val;
    }
}

// Sparse GEMV: General Sparse Matrix-Vector Multiplication (CSC format)
// y = alpha * A * x + beta * y
template<typename T>
constexpr void gemvSp(CSCViewConst<T> A, VecViewConst<T> x, VecView<T> y, 
                      T alpha = T(1.0), T beta = T(0.0)) {
    size_t Rows = A.rows;
    size_t Cols = A.cols;

    assert(x.size() == Cols && "Vector x dimension mismatch with matrix columns");
    assert(y.size() == Rows && "Vector y dimension mismatch with matrix rows");

    // beta * y
    if (beta == T(0.0)) {
        std::fill(y.begin(), y.end(), T(0.0));
    } else if (beta != T(1.0)) {
        for (size_t idx = 0; idx < Rows; ++idx) {
            y[idx] = beta * y[idx];
        }
    }
    
    if (alpha == T(0.0)) return;
    for (size_t idxC = 0; idxC < Cols; ++idxC) {
        size_t colStart = A.colPtr[idxC];
        size_t colEnd = A.colPtr[idxC + 1];

        T alphaXc = alpha * x[idxC];
        for (size_t idxR = colStart; idxR < colEnd; ++idxR) {
            size_t rLcl = A.rowIdx[idxR];
            y[rLcl] += alphaXc * A.values[idxR];
        }
    }
}

// Sparse GEMV Transpose: General Sparse Matrix-Transpose-Vector Multiplication (CSC format)
// y = alpha * A^T * x + beta * y
template<typename T>
constexpr void gemvTSp(CSCViewConst<T> A, VecViewConst<T> x, VecView<T> y, 
                       T alpha = T(1.0), T beta = T(0.0)) {
    size_t Rows = A.rows;
    size_t Cols = A.cols;

    assert(x.size() == Rows && "Vector x dimension mismatch with matrix rows");
    assert(y.size() == Cols && "Vector y dimension mismatch with matrix columns");

    // beta * y
    if (beta == T(0.0)) {
        std::fill(y.begin(), y.end(), T(0.0));
    } else if (beta != T(1.0)) {
        for (size_t idx = 0; idx < Cols; ++idx) {
            y[idx] = beta * y[idx];
        }
    }

    if (alpha == T(0.0)) return;
    for (size_t idxC = 0; idxC < Cols; ++idxC) {
        size_t colStart = A.colPtr[idxC];
        size_t colEnd = A.colPtr[idxC + 1];

        T dot_val = T(0.0);
        for (size_t idxR = colStart; idxR < colEnd; ++idxR) {
            size_t rLcl = A.rowIdx[idxR];
            dot_val += A.values[idxR] * x[rLcl];
        }
        y[idxC] += alpha * dot_val;
    }
}

// =============================================================================
// BLAS Level 3: Matrix-Matrix Operations
// =============================================================================

// Dense GEMM: General Matrix-Matrix Multiplication
// C = alpha * A * B + beta * C
// Optimized for column-major layout: circular loop order j (Cols of B) -> k (Cols of A) -> i (Rows of A)
template<typename T>
constexpr void gemm(Mat2DColViewConst<T> A, Mat2DColViewConst<T> B, Mat2DColView<T> C,
                    T alpha = T(1.0), T beta = T(0.0)) {
    size_t I = A.extent(0);
    size_t K = A.extent(1);
    size_t J = B.extent(1);

    assert(B.extent(0) == K && "B dimension mismatch with A");
    assert(C.extent(0) == I && C.extent(1) == J && "C dimension mismatch with A*B");

    // beta * C
    if (beta == T(0.0)) {
        for (size_t idx = 0; idx < C.size(); ++idx) {
            C.data_handle()[idx] = T(0.0);
        }
    } else if (beta != T(1.0)) {
        for (size_t idx = 0; idx < C.size(); ++idx) {
            C.data_handle()[idx] *= beta;
        }
    }

    if (alpha == T(0.0)) return;
    // column-major circular order: j(N) -> k(K) -> i(M)
    for (size_t idxj = 0; idxj < J; ++idxj) {
        for (size_t idxk = 0; idxk < K; ++idxk) {
            T alphaBkj = alpha * B[idxk, idxj];
            for (size_t idxi = 0; idxi < I; ++idxi) {
                C[idxi, idxj] += alphaBkj * A[idxi, idxk];
            }
        }
    }
}

} // namespace AlgeFlow

#endif // ALGEFLOW_BLAS_HPP
