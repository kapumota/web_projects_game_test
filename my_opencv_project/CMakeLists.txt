cmake_minimum_required(VERSION 3.10)
project(my_opencv_project)

# Especifica el estándar de C++
set(CMAKE_CXX_STANDARD 11)

# Incluye directorios de encabezados
include_directories(include)

# Encuentra OpenCV
find_package(OpenCV REQUIRED)

# Archivos fuente
set(SOURCES
    src/main.cpp
    src/common.cpp
    src/image_processing.cpp
    src/video_processing.cpp
)

# Ejecutable
add_executable(my_opencv_project ${SOURCES})

# Vincula OpenCV al proyecto
target_link_libraries(my_opencv_project ${OpenCV_LIBS})

