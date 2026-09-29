module;

#include "DenseMatrix.hpp"
#include "SparseMatrix.hpp"
#include "PrintMatrix.hpp"
#include "BLAS.cpp"

export module AlgeFlow;

export namespace AlgeFlow {
    // Dense Matrix
    using AlgeFlow::DsMatSttc;

    // Sparse Matrix
    using AlgeFlow::SpMatSttc;
    using AlgeFlow::SpMatMtbl;

    // Matrix & Vector Views
    using AlgeFlow::CSCView;
    using AlgeFlow::CSCViewConst;
    using AlgeFlow::Mat2DColView;
    using AlgeFlow::Mat2DColViewConst;
    using AlgeFlow::VecView;
    using AlgeFlow::VecViewConst;

    // BLAS Operations
    using AlgeFlow::gemv;
    using AlgeFlow::gemvT;
    using AlgeFlow::gemm;
    using AlgeFlow::gemvSp;
    using AlgeFlow::gemvTSp;

    // Matrix Printing Utilities
    using AlgeFlow::print_csc_aligned;
    using AlgeFlow::print_spmatmtbl_info;
    using AlgeFlow::print_blkmap_mapping;
    using AlgeFlow::print_dense_view;
}