project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( neuro neuro )
target_link_libraries( neuro ${OpenCV_LIBS} )