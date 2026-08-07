#include <iostream>
#include <cassert>
#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "SparseMatrix.hpp"

using namespace AlgeFlow;

void test_addBlkMtrx_dense() {
    std::cout << "[Test 1] Testing addBlkMtrx with dense Eigen matrix..." << std::endl;
    SpMatMtbl<double> mat(4, 4);

    Eigen::Matrix2d A;
    A << 1.0, 2.0,
         3.0, 4.0;

    mat.addBlkMtrx(0, 0, A, false);
    mat.setFromTriplets();

    assert(mat.rows() == 4);
    assert(mat.cols() == 4);
    assert(mat.nonZeros() == 4);

    std::cout << "  Passed dense block test!" << std::endl;
}

void test_addBlkMtrx_sparse() {
    std::cout << "[Test 2] Testing addBlkMtrx with sparse Eigen matrix..." << std::endl;
    SpMatMtbl<double> mat(3, 3);

    Eigen::SparseMatrix<double> S(2, 2);
    S.insert(0, 0) = 5.0;
    S.insert(1, 1) = 10.0;
    S.makeCompressed();

    mat.addBlkMtrx(1, 1, S);
    mat.setFromTriplets();

    assert(mat.rows() == 3);
    assert(mat.cols() == 3);
    assert(mat.nonZeros() == 2);

    std::cout << "  Passed sparse block test!" << std::endl;
}

void test_stack() {
    std::cout << "[Test 3] Testing matrix stack / concatenation..." << std::endl;
    SpMatMtbl<double> matA(2, 2);

    Eigen::Matrix2d A;
    A << 1.0, 0.0,
         0.0, 2.0;
    matA.addBlkMtrx(0, 0, A, true);

    Eigen::Matrix2d B;
    B << 3.0, 4.0,
         0.0, 5.0;

    matA.stack(B, SpMatMtbl<double>::ConcatMode::Rght, false);
    matA.setFromTriplets();

    assert(matA.rows() == 2);
    assert(matA.cols() == 4);

    std::cout << "  Passed matrix stack test!" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " Running AlgeFlow SparseMatrix Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_addBlkMtrx_dense();
    test_addBlkMtrx_sparse();
    test_stack();

    std::cout << "\nAll SparseMatrix tests passed successfully!" << std::endl;
    return 0;
}