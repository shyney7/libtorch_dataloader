cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dataloader_test)

list(APPEND CMAKE_PREFIX_PATH "~/projects/libs/libtorch1.8/libtorch")
set(CMAKE_MODULE_PATH "~/projects/libs/libtorch1.8/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(EXECUTABLE_NAME dataloader_test)

add_executable(${EXECUTABLE_NAME} main.cpp)

target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${EXECUTABLE_NAME}
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:${EXECUTABLE_NAME}>)
endif (MSVC)