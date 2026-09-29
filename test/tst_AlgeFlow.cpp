#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <span>
#include <array>
#include <cmath>

#include "Eigen/Core"
#include "Eigen/Dense"

import AlgeFlow;

using namespace AlgeFlow;

// 辅助浮点对比函数
template<typename T>
bool approx_equal(T a, T b, T tol = static_cast<T>(1e-9)) {
    return std::abs(a - b) <= tol;
}

// -----------------------------------------------------------------------------
// Test 1: 稠密静态矩阵 DsMatSttc 及其 BLAS 运算 (gemv, gemvT, gemm)
// -----------------------------------------------------------------------------
void test_dense_matrix_and_blas() {
    std::cout << "\n>>> [1/3] Testing Dense Matrix (DsMatSttc) & BLAS Operations..." << std::endl;

    // 1. 初始化 3x2 列主序静态矩阵 A
    //    A = [ 1.0,  4.0 ]
    //        [ 2.0,  5.0 ]
    //        [ 3.0,  6.0 ]
    DsMatSttc<double, 3, 2> matA;
    matA[0, 0] = 1.0; matA[0, 1] = 4.0;
    matA[1, 0] = 2.0; matA[1, 1] = 5.0;
    matA[2, 0] = 3.0; matA[2, 1] = 6.0;

    assert(matA.rows() == 3 && "matA rows mismatch");
    assert(matA.cols() == 2 && "matA cols mismatch");
    assert(approx_equal(matA[1, 1], 5.0) && "Element access mismatch");

    // 2. 测试 gemv: y = 1.0 * A * x + 0.0 * y
    // x = [2.0, 3.0]^T -> y = [14.0, 19.0, 24.0]^T
    std::array<double, 2> x_vec{2.0, 3.0};
    std::array<double, 3> y_vec{0.0, 0.0, 0.0};
    gemv<double>(matA.view(), x_vec, y_vec, 1.0, 0.0);

    assert(approx_equal(y_vec[0], 14.0) && "gemv result[0] mismatch");
    assert(approx_equal(y_vec[1], 19.0) && "gemv result[1] mismatch");
    assert(approx_equal(y_vec[2], 24.0) && "gemv result[2] mismatch");
    std::cout << "  - Dense gemv passed: y = [" << y_vec[0] << ", " << y_vec[1] << ", " << y_vec[2] << "]^T" << std::endl;

    // 3. 测试 gemvT: y_t = 1.0 * A^T * x_t + 0.0 * y_t
    // x_t = [1.0, 1.0, 1.0]^T -> y_t = [6.0, 15.0]^T
    std::array<double, 3> x_t{1.0, 1.0, 1.0};
    std::array<double, 2> y_t{0.0, 0.0};
    gemvT<double>(matA.view(), x_t, y_t, 1.0, 0.0);

    assert(approx_equal(y_t[0], 6.0) && "gemvT result[0] mismatch");
    assert(approx_equal(y_t[1], 15.0) && "gemvT result[1] mismatch");
    std::cout << "  - Dense gemvT passed: y_t = [" << y_t[0] << ", " << y_t[1] << "]^T" << std::endl;

    // 4. 测试 gemm: C = 1.0 * A * B + 0.0 * C
    // B 为 2x2 矩阵: [ 1.0, 2.0 ]
    //               [ 0.0, 1.0 ]
    // C 应为 3x2:    [ 1.0,  6.0 ]
    //               [ 2.0,  9.0 ]
    //               [ 3.0, 12.0 ]
    DsMatSttc<double, 2, 2> matB;
    matB[0, 0] = 1.0; matB[0, 1] = 2.0;
    matB[1, 0] = 0.0; matB[1, 1] = 1.0;

    DsMatSttc<double, 3, 2> matC;
    gemm<double>(matA.view(), matB.view(), matC.view(), 1.0, 0.0);

    assert(approx_equal(matC[0, 0], 1.0) && approx_equal(matC[0, 1], 6.0));
    assert(approx_equal(matC[1, 0], 2.0) && approx_equal(matC[1, 1], 9.0));
    assert(approx_equal(matC[2, 0], 3.0) && approx_equal(matC[2, 1], 12.0));
    std::cout << "  - Dense gemm passed: C = A * B verified" << std::endl;
}

// -----------------------------------------------------------------------------
// Test 2: 可变稀疏矩阵 SpMatMtbl 拼接、Triplet 压缩与打印
// -----------------------------------------------------------------------------
void test_mutable_sparse_matrix(SpMatMtbl<double>& outMat) {
    std::cout << "\n>>> [2/3] Testing Mutable Sparse Matrix (SpMatMtbl) Assembly..." << std::endl;

    // 1. 初始化 2x2 可变稀疏矩阵
    SpMatMtbl<double> matA(2, 2);

    // 2. 插入稀疏 2x2 块 c1 (通过 CSCViewConst)
    //    c1 = [ 0.0, 0.1 ]
    //         [ 0.3, 0.0 ]
    std::vector<double> vals_c1{0.3, 0.1};
    std::vector<std::size_t> rowIdx_c1{1, 0};
    std::vector<std::size_t> colPtr_c1{0, 1, 2};
    CSCViewConst<double> c1{2, 2, 2, vals_c1.data(), rowIdx_c1.data(), colPtr_c1.data()};
    matA.addBlkMtrx(0, 0, c1);

    // 3. 右下角堆叠 c2 (2x2) -> 尺寸变更为 4x4
    Eigen::Matrix<double, 2, 2> c2;
    c2 << 1.0, 0.0,
          0.0, 4.0;
    matA.stack(Mat2DColViewConst<double>(c2.data(), 2, 2), SpMatMtbl<double>::ConcatMode::BttmRght, true);

    // 4. 右上角堆叠 6x6 单位阵 I -> 尺寸变更为 10x10
    std::vector<double> vals_I(6, 1.0);
    std::vector<std::size_t> rowIdx_I{0, 1, 2, 3, 4, 5};
    std::vector<std::size_t> colPtr_I{0, 1, 2, 3, 4, 5, 6};
    CSCViewConst<double> I{6, 6, 6, vals_I.data(), rowIdx_I.data(), colPtr_I.data()};
    matA.stack(I, SpMatMtbl<double>::ConcatMode::TopRght);

    // 5. 在 (6, 4) 插入稠密 4x6 块 d 并过滤 <= 0.5 的元素
    Eigen::Matrix<double, 4, 6> d;
    d <<  0.8, -0.2,  0.6,  0.1,  0.9, -0.4,
         -0.1,  0.7, -0.3,  0.5, -0.8,  0.2,
          0.9,  0.4, -0.6,  0.8, -0.1,  0.3,
         -0.5,  0.6,  0.2, -0.7,  0.4,  0.9;
    matA.addBlkMtrx(6, 4, Mat2DColViewConst<double>(d.data(), 4, 6), true, 0.5);

    // 6. 转换为 CSC 压缩格式
    matA.setFromTriplets();

    // 验证基本属性
    assert(matA.rows() == 10 && "rows check fails!");
    assert(matA.cols() == 10 && "cols check fails!");
    assert(matA.nonZeros() > 0 && "nonZeros check fails!");

    // 7. 打印稀疏矩阵结构与块映射
    print_spmatmtbl_info(matA);
    print_csc_aligned(matA);
    print_dense_view(matA);
    print_blkmap_mapping(matA);

    outMat = matA;
    std::cout << "  - Mutable Sparse Matrix assembly & CSC conversion passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 3: 稀疏矩阵 BLAS 运算 (gemvSp, gemvTSp) 及 CSCView
// -----------------------------------------------------------------------------
void test_sparse_blas(const SpMatMtbl<double>& matA) {
    std::cout << "\n>>> [3/3] Testing Sparse Matrix BLAS Operations (gemvSp & gemvTSp)..." << std::endl;

    CSCViewConst<double> cscView = matA.view();
    assert(cscView.is_valid() && "CSCView must be valid");

    // 构造测试向量 x = [1, 1, ..., 1]^T
    std::vector<double> x(matA.cols(), 1.0);
    std::vector<double> y(matA.rows(), 0.0);

    // 1. 稀疏乘向量: y = 1.0 * A * x + 0.0 * y
    gemvSp<double>(cscView, x, y, 1.0, 0.0);

    // 计算参考真值 y_expected = A * x
    std::vector<double> y_expected(matA.rows(), 0.0);
    for (std::size_t c = 0; c < cscView.cols; ++c) {
        for (std::size_t p = cscView.colPtr[c]; p < cscView.colPtr[c + 1]; ++p) {
            y_expected[cscView.rowIdx[p]] += cscView.values[p] * x[c];
        }
    }

    for (std::size_t i = 0; i < matA.rows(); ++i) {
        assert(approx_equal(y[i], y_expected[i]) && "gemvSp numerical mismatch");
    }
    std::cout << "  - Sparse gemvSp passed (numerical verification succeeded)." << std::endl;

    // 2. 稀疏转置乘向量: y_t = 1.0 * A^T * y + 0.0 * y_t
    std::vector<double> y_t(matA.cols(), 0.0);
    gemvTSp<double>(cscView, y, y_t, 1.0, 0.0);

    // 计算参考真值 yt_expected = A^T * y
    std::vector<double> yt_expected(matA.cols(), 0.0);
    for (std::size_t c = 0; c < cscView.cols; ++c) {
        for (std::size_t p = cscView.colPtr[c]; p < cscView.colPtr[c + 1]; ++p) {
            yt_expected[c] += cscView.values[p] * y[cscView.rowIdx[p]];
        }
    }

    for (std::size_t j = 0; j < matA.cols(); ++j) {
        assert(approx_equal(y_t[j], yt_expected[j]) && "gemvTSp numerical mismatch");
    }
    std::cout << "  - Sparse gemvTSp passed (numerical verification succeeded)." << std::endl;
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "        Running AlgeFlow Full Unit Tests         " << std::endl;
    std::cout << "=================================================" << std::endl;

    // 1. 稠密矩阵与稠密 BLAS
    test_dense_matrix_and_blas();

    // 2. 可变稀疏矩阵块拼接与 CSC 压缩
    SpMatMtbl<double> matA(1, 1);
    test_mutable_sparse_matrix(matA);

    // 3. 稀疏 BLAS 乘法与转置乘法验证
    test_sparse_blas(matA);

    std::cout << "\n=================================================" << std::endl;
    std::cout << "    All AlgeFlow tests completed successfully!   " << std::endl;
    std::cout << "=================================================" << std::endl;
    return 0;
}