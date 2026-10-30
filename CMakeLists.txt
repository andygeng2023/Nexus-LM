cmake_minimum_required(VERSION 3.20)
project(NEXUS_LM VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(NEXUS_BUILD_TESTS "Build NEXUS tests" ON)
option(NEXUS_BUILD_BENCHMARKS "Build architecture benchmark" ON)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

add_library(nexus_core
    src/tokenizer.cpp
    src/model.cpp
    src/trainer.cpp
)
target_include_directories(nexus_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nexus_core PUBLIC ${TORCH_LIBRARIES} Threads::Threads)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(nexus_core PUBLIC ${TORCH_CXX_FLAGS_LIST})

add_executable(nexus src/main.cpp)
target_link_libraries(nexus PRIVATE nexus_core)

add_executable(nexus_export_web src/export_web.cpp)
target_link_libraries(nexus_export_web PRIVATE nexus_core)

if(NEXUS_BUILD_TESTS)
    enable_testing()
    add_executable(nexus_tests tests/test_core.cpp)
    target_link_libraries(nexus_tests PRIVATE nexus_core)
    add_test(NAME nexus_core_tests COMMAND nexus_tests)
endif()

if(NEXUS_BUILD_BENCHMARKS)
    add_executable(nexus_bench benchmarks/benchmark.cpp)
    target_link_libraries(nexus_bench PRIVATE nexus_core)
endif()
