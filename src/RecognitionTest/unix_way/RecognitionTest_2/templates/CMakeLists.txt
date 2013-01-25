project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( templates templates )
target_link_libraries( templates ${OpenCV_LIBS} )