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
    static constexpr size_t m_ = Rows;     // Row
    static constexpr size_t n_ = Cols;     // Column
    std::array<T, Rows*Cols> values_{};

    constexpr DsMatSttc() = default;
    constexpr DsMatSttc(const std::array<T, Rows*Cols>& data) : values_(data) {}

    constexpr size_t rows() const {return m_;};
    constexpr size_t cols() const {return n_;};

    // mat[r, c] Operator reload
    template<typename Self>
    constexpr auto&& operator[](this Self&& self, size_t r, size_t c){
        assert(r<Rows && c<Cols && "Index out of bounds");
        return std::forward<Self>(self).values_[c*Rows + r];
    }

    // mdspan view with column-major (layout_left)
    constexpr auto view() {
        return std::mdspan<T, std::extents<size_t, Rows, Cols>, std::layout_left>(values_.data());
    }
    constexpr auto view() const {
        return std::mdspan<const T, std::extents<size_t, Rows, Cols>, std::layout_left>(values_.data());
    }

};

} // namespace AlgeFlow

#endif // ALGEFLOW_DENSE_MATRIX_HPP
