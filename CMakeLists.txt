# cmake_minimum_required(VERSION 3.18) # CMake最低版本号要求  
# project(DeepNetFramework) # 项目名字  
  
# # 设置C++标准  
# set(CMAKE_CXX_STANDARD 11)  
# set(CMAKE_CXX_STANDARD_REQUIRED ON)  

# set(CMAKE_CUDA_STANDARD 11)
# enable_language(CUDA)#这个命令告诉系统要使用CUDA语言，绝不可省略

# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# # 添加头文件搜索路径  
include_directories(include)  
  
# # 寻找./src下面所有.cpp为后缀的源文件，并且保存到SRC变量里面  
# file(GLOB_RECURSE SRC ./srcs/*.cu ./srcs/*.cpp ./test/*.cu ./test/*.cpp)  
  
# # 编译SRC变量存储的源文件，编译生成目标文件命名为hello  
# add_executable(target ${SRC})

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(glog REQUIRED libglog)
# include_directories(${glog_INCLUDE_DIRS})

# find_package(gflags REQUIRED)
# include_directories(${gflags_INCLUDE_DIR})

# find_package(OpenMP REQUIRED)
# if(OpenMP_FOUND)
#     add_compile_options(-fopenmp)
#     add_compile_options(-Wunknown-pragmas )
# endif()

# enable_testing()

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# set(MY_SYSTEM_INCLUDE_DIRS
#     /usr/include/c++/9
#     /usr/include/c++/10
# )
project(cuda-practice_v1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

if(CMAKE_BUILD_TYPE MATCHES "debug|Debug|DEBUG")
    add_definitions(-DDEBUG=1)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
endif()


find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)
include_directories(${glog_INCLUDE_DIRS})

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    add_compile_options(-fopenmp)
    add_compile_options(-Wunknown-pragmas )
endif()

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
