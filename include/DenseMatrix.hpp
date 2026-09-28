#ifndef ALGEFLOW_DENSE_MATRIX_HPP
#define ALGEFLOW_DENSE_MATRIX_HPP

#include <numeric>
#include <stdexcept>
#include <cassert>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>
#include <mdspan>
#include <print>      // C++23 格式化输出
#include <algorithm>


namespace AlgeFlow {

using std::size_t;
// Dense matrix utilities and declarations

template<typename T, size_t Rows, size_t Cols>
class DsMatSttc{
public:
    static_assert( Rows>0 && Cols>0, "Dimensions must be >0");
    std::array<T, Rows*Cols> data_{};

    constexpr DsMatSttc() = default;
    constexpr DsMatSttc(const std::array<T, Rows*Cols>& data) : data_(data) {}

    constexpr size_t rows() const {return Rows;};
    constexpr size_t cols() const {return Cols;};

    // mat[r, c] Operator reload
    template<typename Self>
    constexpr auto&& operator[](this Self&& self, size_t r, size_t c){
        assert(r<Rows && c<Cols && "Index out of bounds");
        return std::forward<Self>(self).data_[c*Rows + r];
    }

    // mdspan view with column-major (layout_left)
    constexpr auto view() {
        return std::mdspan<T, std::extents<size_t, Rows, Cols>, std::layout_left>(data_.data());
    }
    constexpr auto view() const {
        return std::mdspan<const T, std::extents<size_t, Rows, Cols>, std::layout_left>(data_.data());
    }

    // BLAS Level 2: GEMV, General Matix-Vector Multiplication
    // y = alpha * A*x + beta * y
    // Column-fisrt optimized: AXPY mode, Column vector combination
    // y = alpha*(x1.*A[:,1] + x2.*A[:,2]+...)
    constexpr void gemv(const std::array<T, Cols>& x, std::array<T, Rows> y, 
                        T alpha=T(1.0), T beta=T(0.0)) const {
        // beta * y
        for(size_t idx=0; idx != Rows; ++idx){
            if(beta == T(0.0)) y[idx] = T(0);
            else y[idx] = beta * y[idx];
        }
        
        if(alpha==T(0.0)) return;
        for(size_t idxC=0; idxC!=Cols; ++idxC){
            T alphaXc = alpha * x[idxC];    
            for(size_t idxR=0; idxR!=Rows; ++idxR){
                y[idxR] = y[idxR] + (*this)[idxR, idxC]; 
            }
        }
    }

    // BLAS Level 2: GEMV Transpose, General Matix-Transpose-Vector Multiplication
    // y = alpha * A^T *x + beta * y
    constexpr void gemvT(const std::array<T, Rows>& x, std::array<T, Cols>& y,
                            T alpha=T(1.0), T beta=T(0.0) ) const {
        // beta * y 
        for(size_t idx=0; idx !=Cols; ++idx){
            if(beta==T(0.0)) y[idx] = T(0.0);
            else y[idx] = beta * y[idx];
        }

        if(alpha==T(0.0)) return;
        for(size_t idxC=0; idxC!=Cols; ++idxC){
            T dot = T(0.0);
            for(size_t idxR; idxR!=Rows; ++idxR){
                dot += x[idxR] * (*this)[idxR, idxC];
            }
            y[idxC] = y[idxC] + alpha*dot;
        }
    }

    // BLAS Level 3: GEMM, General Matrix-Matrix Multiplication
    // no SIMD, block matrix, multi-core parrallel
    // C = alpha * A * B + beta * C
    template<size_t K>
    constexpr void gemm(const DsMatSttc<T, Cols, K>& B, DsMatSttc<T, Rows, K>& C,
                        T alpha=T(1.0), T beta=T(0.0)) const {
        // beta * C
        for(size_t idx=0; idx!=C.data_.size(); ++idx){
            if(beta==T(0.0)) C.data_[idx] = T(0.0);
            else C.data_[idx] = C.data_[idx] * beta;
        }

        if(alpha==T(0.0)) return;
        // column-major circular order: j(N) > k(K) > i(M)
        // inner loop: C(i, J) = A(i, K) .* b(K, J) 
        for(size_t idxj=0; idxj!=K; ++idxj){
            for(size_t idxk=0; idxk!=Cols; ++idxk){
                T alphaBkj = alpha * B[idxk, idxj];
                for(size_t idxi=0; idxi!=Rows; ++idxi){
                    C[idxi, idxj] = C[idxi, idxj] + alphaBkj * (*this)[idxi, idxk];
                }
            }
        }

    }
};

} // namespace AlgeFlow

#endif // ALGEFLOW_DENSE_MATRIX_HPP
