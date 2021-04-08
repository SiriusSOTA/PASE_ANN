project(test_ann_index)

set(CMAKE_CXX_STANDARD 17)


find_package(Boost REQUIRED COMPONENTS system filesystem unit_test_framework)
link_directories(/src)

add_executable(test_ann_index test_parser.cpp test_pase_build.cpp common.cpp )

target_link_libraries(test_ann_index ${Boost_LIBRARIES})
target_link_libraries(test_ann_index ann_index)
