project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( matching matching )
target_link_libraries( matching ${OpenCV_LIBS} )