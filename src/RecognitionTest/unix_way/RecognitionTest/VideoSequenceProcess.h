#include "opencv/cv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include "ProhibitedSignFinder.h"

class VideoSequenceProcess
{
private:
	unsigned int deviceNum;//����� ����������
	std::string windowDesc;//������� ������� ����
	cv::VideoCapture capture;//������
	cv::Mat seqFrame;//������� ����

	ProhibitedSignFinder* sFinder;
public:

	VideoSequenceProcess (std::string singleImageFileName, std::string winName, bool singleImage) {
		try
		{
			if (singleImage)
			{
				seqFrame = cv::imread(singleImageFileName, 1);
				cv::namedWindow(windowDesc = winName, 1);
			}
			else VideoSequenceProcess(singleImageFileName, winName);//���������� ������ �������������.
		}
		catch (...)
		{
			std::cout << "No image exists";
		}
	}

	VideoSequenceProcess (std::string fileName, std::string winName) {
		try
		{
			capture.open(fileName);
			cv::namedWindow(windowDesc = winName, 1);
		}
		catch(...)
		{
			std::cout << "No videofile exists";
		}
	}

	VideoSequenceProcess (int device, std::string winName) {
		try
		{
			capture.open(deviceNum = device);
			cv::namedWindow(windowDesc = winName, 1);
		}
		catch (...)
		{
			std::cout << "No videodevice exists";
		}

	}

	~VideoSequenceProcess(){
		//������� ���������
		if (capture.isOpened()) capture.release();

		cv::destroyWindow(windowDesc);

		seqFrame.release();

		delete sFinder;
	}

	void processSequence()
	{
		try
		{
			videoGetFrame();
			processFrame();
		}
		catch (...)
		{
			std::cout << "No frame can be captured";
		}
	}


	bool processFrame()
	{
		sFinder = new ProhibitedSignFinder(seqFrame);
		try
		{
			sFinder->signDetect(seqFrame);

			return true;
		}
		catch(...)
		{
			std::cout << "Unknown problem with ObjectFinder has been occured";
		}
	}

void show()
{
	cv::imshow(windowDesc, seqFrame);
}

private:
	bool videoGetFrame()
	{
		return capture.read(seqFrame);
	}

};