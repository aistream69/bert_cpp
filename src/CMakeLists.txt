
set(target sample)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(Torch REQUIRED)

include_directories(
    )

link_directories(
    "${PROJECT_ROOT_PATH}/build/utils/ftokenizers"
    "${PROJECT_ROOT_PATH}/build/utils/tokenizers/tokenizers/lib/unilib"
    "${PROJECT_ROOT_PATH}/build/utils/tokenizers/tokenizers/bert"
    "${PROJECT_ROOT_PATH}/build/utils/tokenizers/tokenizers/utils"
    "${PROJECT_ROOT_PATH}/build/utils/tokenizers/tokenizers/fundamental"
    "${PROJECT_ROOT_PATH}/build/utils/tokenizers/tokenizers/basic"
    )

add_executable(
    ${target} 
    main.cpp
    )

add_dependencies(${target} tokenizers ftokenizers)

set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

target_link_libraries(
    ${target}
    ${TORCH_LIBRARIES} 
    -lftokenizers
    -lbert_tokenizer
    -lbasic_tokenizers
    -lfundamental_tokenizer
    -lutils
    -lunilib
    -licuuc
    -licui18n
    -lpthread
    )

