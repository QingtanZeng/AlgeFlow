module;

#include "SparseMatrix.hpp"
#include "DenseMatrix.hpp"
#include "PrintMatrix.hpp"

export module AlgeFlow;

export namespace AlgeFlow {
    using AlgeFlow::SpMatSttc;
    using AlgeFlow::SpMatMtbl;
    using AlgeFlow::print_csc_aligned;
    using AlgeFlow::print_spmatmtbl_info;
    using AlgeFlow::print_blkmap_mapping;
    using AlgeFlow::print_dense_view;
}