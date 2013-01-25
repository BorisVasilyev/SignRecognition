project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( svm svm )
target_link_libraries( svm ${OpenCV_LIBS} )