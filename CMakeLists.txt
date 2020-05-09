cmake_minimum_required(VERSION 3.15)
project(KokkosApplication)

set(CMAKE_CXX_STANDARD 17)

# Kokkos specific options
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
set(Kokkos_ARCH_PASCAL61 ON CACHE BOOL "")
set(Kokkos_ENABLE_CUDA_UVM ON CACHE BOOL "")
set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
set(Kokkos_ENABLE_DEPRECATED_CODE OFF CACHE BOOL "")
set(Kokkos_CUDA_DIR /usr/local/cuda CACHE PATH "")

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable C++ extansions for Clang++, or otherwise it may crash with -std=gnu+11 flag
set(CMAKE_CXX_EXTENSIONS OFF)

# Qt path
set(CMAKE_PREFIX_PATH ~/Qt/5.14.1/gcc_64/)

find_package(Qt5 COMPONENTS Widgets Gui Charts REQUIRED)

add_subdirectory(kokkos)

set(INCLUDE_FILES
        include/parallel/gpu_code.h
        include/gui/mainwindow.h
        include/gui/speedupgraph.h
        include/gui/optionswidget.h
        include/gui/matrixwidget.h
        include/gui/nussinovwidget.h)

set(SOURCE_FILES
        src/main.cpp
        src/parallel/gpu_code.cpp
        src/gui/mainwindow.cpp
        src/gui/speedupgraph.cpp
        src/gui/mainwindow.ui
        src/gui/optionswidget.cpp
        src/gui/optionswidget.ui
        src/gui/matrixwidget.cpp
        src/gui/matrixoptions.ui
        src/gui/nussinovwidget.cpp
        src/gui/nussinovoptions.ui)

add_executable(KokkosApplication ${INCLUDE_FILES} ${SOURCE_FILES})

target_link_libraries(KokkosApplication PRIVATE
        kokkos
        Qt5::Widgets
        Qt5::Gui
        Qt5::Charts)

target_include_directories(KokkosApplication PRIVATE ${Kokkos_INCLUDE_DIRS_RET} include)
