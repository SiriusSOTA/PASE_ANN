project(ann_index)

set(CMAKE_CXX_STANDARD 20)

link_directories(/thread_pool)

add_library(ann_index utils.cpp)
target_include_directories(ann_index PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(ann_index INTERFACE thread_pool)
