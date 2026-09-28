#include <numeric>
#include <span>
#include <stdexcept>
#include <cassert>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>
#include <mdspan>
#include <print>      // C++23 格式化输出
#include <algorithm>

#include "DenseMatrix.hpp"
#include "SparseMatrix.hpp"

namespace AlgeFlow {

using std::size_t;

template<typename T>
using Mat2DColView = std::mdspan<T, std::dextents<std::size_t, 2>, std::layout_left>;
template<typename T>
using Mat2DColViewConst = std::mdspan<const T, std::dextents<std::size_t, 2>, std::layout_left>;
template <typename T>
using VecView = std::span<T>;
template <typename T>
using VecViewConst = std::span<const T>;

// BLAS Level 2: GEMV, General Matix-Vector Multiplication
// y = alpha * A*x + beta * y
// Column-fisrt optimized: AXPY mode, Column vector combination
// y = alpha*(x1.*A[:,1] + x2.*A[:,2]+...)
template<typename T>
constexpr void gemv(Mat2DColViewConst<T> A, VecViewConst<T> x, VecView<T> y, 
                    T alpha=T(1.0), T beta=T(0.0)) {

    size_t Rows = A.extent(0);
    size_t Cols = A.extent(1);

    assert(x.size() == Cols && "Vector x dimension mismatch with matrix columns");
    assert(y.size() == Rows && "Vector y dimension mismatch with matrix rows");

    // beta * y
    size_t idx=0;
    if(beta == T(0.0)){
        for(; idx != Rows; ++idx) y[idx] = T(0.0);
    }else{
        for(; idx != Rows; ++idx) y[idx] = beta * y[idx];
    }
    
    if(alpha==T(0.0)) return;
    for(size_t idxC=0; idxC!=Cols; ++idxC){
        T alphaXc = alpha * x[idxC];    
        for(size_t idxR=0; idxR!=Rows; ++idxR){
            y[idxR] = y[idxR] + A[idxR, idxC]; 
        }
    }
}

// BLAS Level 2: GEMV Transpose, General Matix-Transpose-Vector Multiplication
// y = alpha * A^T *x + beta * y
template<typename T>
constexpr void gemvT(Mat2DColViewConst<T> A, VecViewConst<T> x, VecView<T> y,
                    T alpha=T(1.0), T beta=T(0.0) )  {
    size_t Rows = A.extent(0);
    size_t Cols = A.extent(1);

    assert(x.size() == Rows && "Vector x dimension mismatch with matrix rows");
    assert(y.size() == Cols && "Vector y dimension mismatch with matrix columns");

    // beta * y 
    size_t idx=0;
    if(beta == T(0.0)){
        for(; idx != Cols; ++idx) y[idx] = T(0.0);
    }else{
        for(; idx != Cols; ++idx) y[idx] = beta * y[idx];
    }

    if(alpha==T(0.0)) return;
    for(size_t idxC=0; idxC!=Cols; ++idxC){
        T dot = T(0.0);
        for(size_t idxR; idxR!=Rows; ++idxR){
            dot += x[idxR] * A[idxR, idxC];
        }
        y[idxC] = y[idxC] + alpha*dot;
    }
}

// BLAS Level 3: GEMM, General Matrix-Matrix Multiplication
// no SIMD, block matrix, multi-core parrallel
// C = alpha * A * B + beta * C
template<typename T>
constexpr void gemm(Mat2DColViewConst<T> A, Mat2DColViewConst<T> B, Mat2DColView<T> C,
                    T alpha=T(1.0), T beta=T(0.0)) {
    size_t I = A.extent(0);
    size_t K = A.extent(1);
    size_t J = B.extent(1);

    assert(B.extent(0) == K && "B dimension mismatch with A");
    assert(C.extent(0)==I && C.extent(1)==J && "C dimension mismatch with A*B");

    // beta * C
    size_t idx=0;
    if(beta==T(0.0)){
        for(; idx != C.size(); ++idx) C.values_[idx] = T(0.0);
    }else{
        for(; idx != C.size(); ++idx)  C.values_[idx] = C.values_[idx] * beta;
    }

    if(alpha==T(0.0)) return;
    // column-major circular order: j(N) > k(K) > i(M)
    // inner loop: C(i, J) = A(i, K) .* b(K, J) 
    for(size_t idxj=0; idxj != J; ++idxj){
        for(size_t idxk=0; idxk != K; ++idxk){
            T alphaBkj = alpha * B[idxk, idxj];
            for(size_t idxi=0; idxi != I; ++idxi){
                C[idxi, idxj] = C[idxi, idxj] + alphaBkj * A[idxi, idxk];
            }
        }
    }

}

// BLAS Level 2: sparse GEMV, General Matix-Vector Multiplication
// y = alpha * A*x + beta * y
// Column-fisrt optimized: AXPY mode, Column vector combination
// y = alpha*(x1.*A[:,1] + x2.*A[:,2]+...)
template<typename T>
constexpr void gemvSp(CSCViewConst<T> A, VecViewConst<T> x, VecView<T> y, 
                    T alpha=T(1.0), T beta=T(0.0)) {

    size_t Rows = A.rows;
    size_t Cols = A.cols;

    assert(x.size() == Cols && "Vector x dimension mismatch with matrix columns");
    assert(y.size() == Rows && "Vector y dimension mismatch with matrix rows");

    // beta * y
    if(beta == T(0.0)){
        std::fill(y.begin(), y.end(), T(0.0));
    }else{
        for(size_t idx=0; idx != Rows; ++idx) y[idx] = beta * y[idx];
    }
    
    if(alpha==T(0.0)) return;
    for(size_t idxC=0; idxC!=Cols; ++idxC){
        size_t colStart = A.colPtr[idxC];
        size_t colEnd = A.colPtr[idxC+1];

        T alphaXc = alpha * x[idxC];
        for(size_t idxR=colStart; idxR!=colEnd; ++idxR){
            size_t rLcl = A.rowIdx[idxR];
            y[rLcl] = y[rLcl] + alphaXc * A.values[idxR];
        }
    }
}

// BLAS Level 2: GEMV Transpose, General Matix-Transpose-Vector Multiplication
// y = alpha * A^T *x + beta * y
template<typename T>
constexpr void gemvTSp(CSCViewConst<T> A, VecViewConst<T> x, VecView<T> y, 
                    T alpha=T(1.0), T beta=T(0.0)) {
    size_t Rows = A.rows;
    size_t Cols = A.cols;

    assert(x.size() == Rows && "Vector x dimension mismatch with matrix rows");
    assert(y.size() == Cols && "Vector y dimension mismatch with matrix columns");

    // beta * y
    if(beta == T(0.0)){
        std::fill(y.begin(), y.end(), T(0.0));
    }else{
        for(size_t idx=0; idx != Cols; ++idx) y[idx] = beta * y[idx];
    }

    if(alpha==T(0.0)) return;
    for(size_t idxC=0; idxC!=Cols; ++idxC){
        size_t colStart = A.colPtr[idxC];
        size_t colEnd = A.colPtr[idxC+1];

        T dot = T(0.0);
        for(size_t idxR=colStart; idxR!=colEnd; ++idxR){
            size_t rLcl = A.rowIdx[idxR];
            dot += A.values[idxR] * x[rLcl];
        }
        y[idxC] = y[idxC] + alpha * dot;
    }
}

}

