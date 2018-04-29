# define variables
set(APP test_hardware_trigger)
add_executable(${APP} hardware_trigger.cpp)

# link
target_compile_features(${APP} PUBLIC cxx_std_11)
if (UNIX)
    target_link_libraries(${APP} PRIVATE flycapture)
endif (UNIX)
if (WIN32)
    target_link_libraries(${APP} PRIVATE flycapture2)
endif (WIN32)

# install                                   
install(TARGETS ${APP} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
