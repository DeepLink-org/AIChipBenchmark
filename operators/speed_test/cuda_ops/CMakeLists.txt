# set the project name
project(cuda_gemm_conv)

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(CUDNN "7.0")

find_package(CUDA "9.0")

if (CUDA_FOUND)
    message(STATUS "CUDA Libraries: " ${CUDA_LIBRARIES})
    message(STATUS "CUDA Version: " ${CUDA_VERSION_STRING})
    message(STATUS "NVCC Path: ${CUDA_TOOLKIT_ROOT_DIR}/bin")
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
        # cmake 3.3 and later version may auto-set nvcc host compiler with
        # -std=c++11 option, which will cause std redefine error
        if (${CMAKE_VERSION} VERSION_LESS "3.3")
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
        endif()

        if (${CMAKE_VERSION} VERSION_GREATER "3.3" AND ${CMAKE_VERSION} VERSION_LESS "4")
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
        endif()

        if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER "4" AND ${CUDA_VERSION_STRING} VERSION_LESS "8")
            # nvcc in cuda 7.5 or lower cannot work well with g++ 5.0 or higher,
            # here is a hack to solve this problem
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
        endif()

        include(cmake/CUDAComputeArch.cmake)
        if (HALF)
            if (NOT (CUDA_VERSION VERSION_LESS "11.0"))
                cuda_select_nvcc_arch_flags(ARCH_FLAGS 6.0 6.1 7.0 7.5 8.0+PTX)
            else()
                if (NOT (CUDA_VERSION VERSION_LESS "10.0"))
                    cuda_select_nvcc_arch_flags(ARCH_FLAGS 6.0 6.1 7.0 7.5+PTX)
                else()
                    cuda_select_nvcc_arch_flags(ARCH_FLAGS 6.0 6.1 7.0+PTX)
                endif()
            endif()
        else()
            if (CUDA_ARCH_AUTO)
                cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
            else()
                cuda_select_nvcc_arch_flags(ARCH_FLAGS Common)
            endif()
        endif()
        message(STATUS "Selected CUDA archs : ${ARCH_FLAGS}")
        string(REPLACE ";" " " ARCH_FLAG_STR "${ARCH_FLAGS}")
        string(REPLACE ";" " " CUDA_LIBRARIES_STR "${CUDA_LIBRARIES}")
        set(APPENDIX_VERSION_INFO ${APPENDIX_VERSION_INFO}
            "NVCC_PATH ${CUDA_TOOLKIT_ROOT_DIR}/bin"
            "CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}"
            "CUDA_LIBRARIES ${CUDA_LIBRARIES_STR}"
            "CUDA_ARCH_FLAGS ${ARCH_FLAG_STR}"
            "CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}")
        list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS} --expt-extended-lambda)
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# add the executable
set(PROJECT_NAME "gemm")
add_executable(gemm cuda_gemm.cpp)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(conv cudnn_convforward.cpp)

target_link_libraries(conv ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES})

