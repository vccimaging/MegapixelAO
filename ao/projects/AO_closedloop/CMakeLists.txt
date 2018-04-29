# define variables
set(APP AO_closedloop)

add_executable(${APP} AO_closedloop.cpp 
                    ${CMAKE_SOURCE_DIR}/deps/glad/glad.c
                    gen_SLM_images.cu)

# link
set_target_properties(${APP} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if (UNIX)
	target_link_libraries(${APP} PRIVATE ${STATIC_LIB_FLOWCUDA} ${GLFW_STATIC_LIBRARIES} ${OpenCV_LIBS}
										cufft flycapture)
endif (UNIX)
if (WIN32)
	target_link_libraries(${APP} PRIVATE ${STATIC_LIB_FLOWCUDA} glfw3 opencv_cudawarping opencv_cudaimgproc
							opencv_imgproc opencv_highgui cufft flycapture2)
endif (WIN32)

# install                                   
install(TARGETS ${APP} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
