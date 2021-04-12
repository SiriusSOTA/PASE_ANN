project(test_ann_index)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS system filesystem unit_test_framework)
link_directories(/src)

#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

add_executable(test_ann_index test_parser.cpp test_pase_build.cpp test_k_means.cpp test_search.cpp common.cpp)

target_link_libraries(test_ann_index ${Boost_LIBRARIES})
target_link_libraries(test_ann_index ann_index)
