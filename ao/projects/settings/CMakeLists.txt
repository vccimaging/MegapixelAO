# define variables
set(APP settings)

add_executable(${APP} settings.cpp ${CMAKE_SOURCE_DIR}/deps/glad/glad.c)

# link
target_compile_features(${APP} PUBLIC cxx_std_11)
set_target_properties(${APP} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if (UNIX)
	target_link_libraries(${APP} PRIVATE ${GLFW_STATIC_LIBRARIES} flycapture opencv_highgui)
endif (UNIX)
if (WIN32)
	target_link_libraries(${APP} PRIVATE glfw3 flycapture2 opencv_highgui)
endif (WIN32)

# install                                   
install(TARGETS ${APP} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
