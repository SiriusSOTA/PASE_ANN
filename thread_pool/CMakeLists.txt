project(thread_pool)

set(CMAKE_CXX_STANDARD 20)

add_library(thread_pool thread_pool.cpp thread_pool.h)
target_include_directories(thread_pool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Boost REQUIRED COMPONENTS system filesystem thread)
target_link_libraries(thread_pool ${Boost_LIBRARIES})
