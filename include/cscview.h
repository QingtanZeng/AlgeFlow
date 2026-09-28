#include <cstddef>
#include <cassert>

namespace AlgeFlow {
using std::size_t;

template<typename T>
struct CSCViewConst{
    size_t rows;
    size_t cols;
    size_t nnz;
    
    const T* values;
    const size_t* rowIdx;
    const size_t* colPtr;

    constexpr bool is_valid() const {
        return values!=nullptr && rowIdx!=nullptr && colPtr!=nullptr;
    }
};

template<typename T>
struct CSCView{
    size_t rows;
    size_t cols;
    size_t nnz;
    
    T* values;
    const size_t* rowIdx;
    const size_t* colPtr;
};

}