#pragma once
#include <opencv\cv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>;

#include "ProhibitedSignFinder.h"

class VideoSequenceProcess
{
private:
	unsigned int deviceNum;//номер устройства
	std::string windowDesc;//главное рабочее окно
	cv::VideoCapture capture;//захват
	cv::Mat seqFrame;//рабочий кадр

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
			else VideoSequenceProcess(singleImageFileName, winName);//пользуемся вторым конструктором.
		}
		catch (...)
		{
			cout << "No image exists";
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
			cout << "No videofile exists";
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
			cout << "No videodevice exists";
		}

	}

	~VideoSequenceProcess(){
		//удаляем созданное
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
			cout << "No frame can be captured";
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
			cout << "Unknown problem with ObjectFinder has been occured";
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