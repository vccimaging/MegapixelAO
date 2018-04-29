# define variables
set(APP AO_xy_alignment)

add_executable(${APP} AO_xy_alignment.cpp ${CMAKE_SOURCE_DIR}/deps/glad/glad.c)

# link
target_compile_features(${APP} PUBLIC cxx_std_11)
if (UNIX)
	target_link_libraries(${APP} PRIVATE ${GLFW_STATIC_LIBRARIES} cudart flycapture
                                     opencv_imgcodecs opencv_imgproc opencv_core opencv_highgui)                                  
endif (UNIX)
if (WIN32)
	target_link_libraries(${APP} PRIVATE glfw3 cudart flycapture2
                                     opencv_imgcodecs opencv_imgproc opencv_core opencv_highgui)                                  
endif (WIN32)

# install                                   
install(TARGETS ${APP} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
