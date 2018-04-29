cmake_minimum_required(VERSION 2.8)
project(AO LANGUAGES C CXX CUDA)

# check requirements
if (UNIX)
	find_package(PkgConfig REQUIRED)
	pkg_search_module(GLFW REQUIRED glfw3)
	find_package(glfw3  3.2 REQUIRED)
	find_package(OpenCV 3.0 REQUIRED)
endif (UNIX)
if (WIN32)
	find_package(glfw3  3.2 REQUIRED PATHS C:/Program\ Files/GLFW/lib/cmake/glfw3)
	find_package(OpenCV 3.0 REQUIRED PATHS C:/Users/Congli\ Wang/Documents/opencv-3.4.0/build)
endif (WIN32)
find_package(OpenGL 4.5 REQUIRED)
find_package(CUDA   8.0 REQUIRED)

# set include directories
if (UNIX)
	set(FLYCAPTURE_DIR /usr/include/flycapture)
	set(CUDA_TARGET_INC ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include)
	set(CUDA_SAMPLE_INC ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
endif (UNIX)
if (WIN32)
	set(FLYCAPTURE_DIR C:/Program\ Files/Point\ Grey\ Research/FlyCapture2/include)
	set(CUDA_TARGET_INC C:/Program\ Files/NVIDIA GPU\ Computing\ Toolkit/CUDA/v9.0/include)
	set(CUDA_SAMPLE_INC C:/ProgramData/NVIDIA\ Corporation/CUDA\ Samples/v9.0/common/inc)
	include_directories(C:/Program\ Files/GLFW/include)
	include_directories(C:/Users/Congli\ Wang/Documents/opencv-3.4.0/build/install/include)
endif (WIN32)
include_directories(deps utils src ${FLYCAPTURE_DIR} ${CUDA_TARGET_INC} ${CUDA_SAMPLE_INC})

# set link directories
if (UNIX)
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
endif (UNIX)
if (WIN32)
	link_directories(C:/Program\ Files/Point\ Grey\ Research/FlyCapture2/lib64)
	link_directories(C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.0/lib/x64)	
	link_directories(C:/Program\ Files/GLFW/lib)
endif (WIN32)

# define C++ 11
set(CMAKE_CXX_STANDARD 11)

# define the name for our static library 
set(STATIC_LIB_FLOWCUDA flowCUDA)

# build a static library for our customized CWS algorithm
add_library(${STATIC_LIB_FLOWCUDA} STATIC
  src/flowCUDA.cu
  src/prepare_precomputations.cu
  src/prepare_cufft_warmup.cu
)

# O3 optimization for GCC
if (CMAKE_COMPILER_IS_GNUXX)
	add_compile_options(-O3)
endif (CMAKE_COMPILER_IS_GNUXX)

# add sub directories of each project
add_subdirectory(projects/AO_closedloop)
add_subdirectory(projects/AO_xy_alignment)
add_subdirectory(projects/settings)
add_subdirectory(tests/background_SLM)
add_subdirectory(tests/shader)
add_subdirectory(tests/solver_accuracy)
add_subdirectory(tests/hardware_trigger)

# install
install(TARGETS ${STATIC_LIB_FLOWCUDA}
        ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/lib
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/utils ${CMAKE_SOURCE_DIR}/src
        DESTINATION ${CMAKE_SOURCE_DIR}/include
        FILES_MATCHING PATTERN "*.h")
