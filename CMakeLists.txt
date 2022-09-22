
cmake_minimum_required(VERSION 3.5)

project (BERT_CPP)

set(PROJECT_ROOT_PATH "${BERT_CPP_SOURCE_DIR}")
set(CMAKE_VERBOSE_MAKEFILE ON)

message(STATUS "BUILD INFO:")
message(STATUS "\tPrefix: ${CMAKE_INSTALL_PREFIX}")

add_subdirectory(src)
add_subdirectory(utils/tokenizers)
add_subdirectory(utils/ftokenizers)

