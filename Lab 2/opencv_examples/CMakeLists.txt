project(opencv_examples)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE "Release")

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

#Add here your source files
set(TEST_3DP_SRCS src/main.cpp)

add_executable(${PROJECT_NAME} ${TEST_3DP_SRCS})
target_include_directories( ${PROJECT_NAME} PUBLIC 
                            ${OpenCV_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
set_target_properties(opencv_examples PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

