add_executable(${PROJECT_NAME}
  main.cpp
  Matrix.cpp
  Algebra.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)