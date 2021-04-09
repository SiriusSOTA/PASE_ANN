project(ann_index)

set(CMAKE_CXX_STANDARD 17)

add_library(ann_index INTERFACE)
target_include_directories(ann_index INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
