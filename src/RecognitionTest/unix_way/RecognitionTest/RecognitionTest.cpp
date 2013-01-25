#include "VideoSequenceProcess.h"

int main( int argc, char** argv )
{
	VideoSequenceProcess vsProcess("/home/boris/src/test.avi", "TestRecognition", false);
	
	vsProcess.processFrame();
	vsProcess.show();
	cv::waitKey(0);

	return 0;
}