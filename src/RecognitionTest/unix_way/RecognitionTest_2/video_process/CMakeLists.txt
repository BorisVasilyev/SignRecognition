project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( video_process video_process )
target_link_libraries( video_process ${OpenCV_LIBS} )