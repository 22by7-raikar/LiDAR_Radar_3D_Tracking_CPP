find_package(GTest REQUIRED)

add_executable(fusion_tests
  test_models.cpp
  test_filters.cpp
  test_tracking.cpp
  test_sim.cpp
  test_config.cpp
  test_regression.cpp
)
target_link_libraries(fusion_tests PRIVATE fusion_tracking GTest::gtest GTest::gtest_main)
target_compile_definitions(fusion_tests PRIVATE
  FUSION_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

include(GoogleTest)
gtest_discover_tests(fusion_tests)
