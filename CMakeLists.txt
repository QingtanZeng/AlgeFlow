# 1. 声明 CMake 最低版本要求 (C++20/23 模块支持推荐 CMake 3.28+)
cmake_minimum_required(VERSION 3.28)

# 指定使用 clang/clang++ 编译器
set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)


# 2. 声明项目名称 and 支持的语言
project(AlgeFlow VERSION 1.0 LANGUAGES C CXX)

# 设置默认构建类型为 Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# 强制 CMake 导出 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 3. 全局设置 C++ 标准 (C++23)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 4. 定义 AlgeFlow C++23 模块库
add_library(AlgeFlow)

target_sources(AlgeFlow
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/AlgeFlow.cppm
)

target_include_directories(AlgeFlow
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/libs/eigen>
)

# 5. 编译测试可执行文件
add_executable(tst_SparseMatrix test/tst_SparseMatrix.cpp)

target_link_libraries(tst_SparseMatrix PRIVATE AlgeFlow)

target_include_directories(tst_SparseMatrix PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/libs/eigen
)

target_compile_options(tst_SparseMatrix PRIVATE -g -O0)