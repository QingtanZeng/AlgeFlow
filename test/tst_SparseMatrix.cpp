#include <iostream>
#include <cassert>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "SparseMatrix.hpp"
#include "PrintMatrix.hpp"

using namespace AlgeFlow;


int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " Running AlgeFlow SparseMatrix Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. Initialize 2x2 mutable sparse matrix
    SpMatMtbl<double> matA(2, 2);

    // 2. Add sparse 2x2 block c1 at (0, 0)
    Eigen::SparseMatrix<double> c1(2, 2);
    c1.insert(0, 1) = 0.1;
    c1.insert(1, 0) = 0.3;
    c1.makeCompressed();
    matA.addBlkMtrx(0, 0, c1);

    // 3. Stack c2 (2x2) at BttmRght -> Resulting size: 4x4
    Eigen::Matrix<double, 2, 2> c2;
    c2 << 1.0, 0.0,
          0.0, 4.0;
    matA.stack(c2, SpMatMtbl<double>::ConcatMode::BttmRght, true);

    // 4. Stack I (6x6 Identity) at TopRght -> Resulting size: 10x10
    Eigen::SparseMatrix<double> I(6, 6);
    I.setIdentity();
    I.makeCompressed();
    matA.stack(I, SpMatMtbl<double>::ConcatMode::TopRght);

    // 5. Add dense 4x6 block d at (6, 4) with threshold filtering (> 0.5)
    Eigen::Matrix<double, 4, 6> d;
    d <<  0.8, -0.2,  0.6,  0.1,  0.9, -0.4,
         -0.1,  0.7, -0.3,  0.5, -0.8,  0.2,
          0.9,  0.4, -0.6,  0.8, -0.1,  0.3,
         -0.5,  0.6,  0.2, -0.7,  0.4,  0.9;
    matA.addBlkMtrx(6, 4, d, true, 0.5);

    // 6. Compress COO triplets to CSC format
    matA.setFromTriplets();

    // 7. Print aligned CSC information, block layout view, and detailed mappings
    print_spmatmtbl_info(matA);
    print_csc_aligned(matA);
    print_dense_view(matA);
    print_blkmap_mapping(matA);

    // 8. Assertions
    assert(matA.rows() == 10 && "rows check fails!");
    assert(matA.cols() == 10 && "cols check fails!");
    assert(matA.nonZeros() > 0 && "nonZeros check fails!");

    std::cout << "\nAll SparseMatrix tests passed successfully!" << std::endl;
    return 0;
}