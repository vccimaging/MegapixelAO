# define variables
set(APP test_background_SLM)

add_executable(${APP} background_SLM.cu ${CMAKE_SOURCE_DIR}/deps/glad/glad.c)

# link
target_compile_features(${APP} PUBLIC cxx_std_11)
# if (UNIX)
#     target_link_libraries(${APP} PRIVATE ${GLFW_STATIC_LIBRARIES} GL m X11 Xrandr Xinerama Xi Xxf86vm Xcursor pthread dl)
# endif (UNIX)
# if (WIN32)
#     target_link_libraries(${APP} PRIVATE glfw3)
# endif (WIN32)
if (UNIX)
    target_link_libraries(${APP} PRIVATE ${GLFW_STATIC_LIBRARIES} opencv_highgui cudart)
endif (UNIX)
if (WIN32)
	target_link_libraries(${APP} PRIVATE glfw3 opencv_highgui cudart)
endif (WIN32)

# install                                   
install(TARGETS ${APP} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
