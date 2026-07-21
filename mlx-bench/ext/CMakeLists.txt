# TEMPORAL-PATCH extension build (out-of-tree MLX extension; links the pip
# wheel's libmlx.dylib, pinned mlx==0.32.0 -- see ext/README note in repo docs)
cmake_minimum_required(VERSION 3.25)
project(_temporal_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

nanobind_add_module(_temporal_stream NB_STATIC STABLE_ABI NB_DOMAIN mlx
                    ${CMAKE_CURRENT_LIST_DIR}/temporal_stream.cpp)
target_link_libraries(_temporal_stream PRIVATE mlx)
target_compile_definitions(_temporal_stream PRIVATE _METAL_)
