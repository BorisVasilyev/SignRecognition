// RecognitionTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "VideoSequenceProcess.h"


int _tmain(int argc, _TCHAR* argv[])
{
	VideoSequenceProcess vsProcess("E:/android/sign/sign00.jpg", "TestRecognition", true);
	
	vsProcess.processFrame();
	vsProcess.show();
	cv::waitKey(0);

	return 0;
}

