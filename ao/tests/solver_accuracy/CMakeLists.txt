# define variables
set(APP  test_solver_accuracy)
set(APP2 test_fista_rof)

add_executable(${APP}  solver_accuracy.cpp)
add_executable(${APP2} fista_rof.cpp)

target_compile_features(${APP} PUBLIC cxx_std_11)
target_compile_features(${APP2} PUBLIC cxx_std_11)

# link
if (UNIX)
    target_link_libraries(${APP} PRIVATE ${STATIC_LIB_FLOWCUDA} ${OpenCV_LIBS} cufft)
    target_link_libraries(${APP2} PRIVATE ${STATIC_LIB_FLOWCUDA} ${OpenCV_LIBS} cufft)
endif (UNIX)
if (WIN32)
	target_link_libraries(${APP} PRIVATE ${STATIC_LIB_FLOWCUDA} cufft opencv_imgcodecs)
    target_link_libraries(${APP2} PRIVATE ${STATIC_LIB_FLOWCUDA} cufft opencv_imgcodecs)
endif (WIN32)

# install                                   
install(TARGETS ${APP}  RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
install(TARGETS ${APP2} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
