# define variables
set(APP test_shader)

add_executable(${APP} shader.cpp ${CMAKE_SOURCE_DIR}/deps/glad/glad.c)

# link
target_compile_features(${APP} PUBLIC cxx_std_11)
if (UNIX)
    target_link_libraries(${APP} PRIVATE ${GLFW_STATIC_LIBRARIES} cudart)
endif (UNIX)
if (WIN32)
	target_link_libraries(${APP} PRIVATE glfw3 cudart)
endif (WIN32)

# install                                   
install(TARGETS ${APP} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
