#include "ProhibitedSignFinder.h"

#include <stdio.h>
#include <iostream>

#define MAXHUE 180
#define MAXSAT 255
#define MAXVAL 255


void ProhibitedSignFinder:: signDetect (cv::Mat &curFrame)
{
	try
	{
		std::vector <cv::Mat> channelSplit;
		std::vector<cv::Mat>::iterator it;

		//tempImage->set
		
		cv::cvtColor(sFrame, sFrame, CV_BGR2HSV);//32CU3 type
	
		cv::split(sFrame, channelSplit);//split into SINGLE channel Mats of 32 bit values

		if (channelSplit.empty()) 
		{
			std::cout << "Frame splitter empty";

			cv::waitKey(TIMETOWAIT);

			return;
		}

		for (it = channelSplit.begin(); it < channelSplit.end(); it++)
		{
			static int i = 0;

			
			switch (i)
			{
				case 0:
					cv::threshold(channelSplit.at(i), channelSplit.at(i),  0.8*MAXHUE, MAXHUE, cv::THRESH_BINARY_INV);

					tempImage1 = channelSplit.at(i).clone();//create same SINGLE channel Mat

					custShow(tempImage1);
					//tempImage2 = tempImage1.clone();

					break;
				case 1:
					cv::threshold(channelSplit.at(i), channelSplit.at(i),  0.7*MAXSAT, MAXSAT, cv::THRESH_BINARY_INV);

					cv::bitwise_and(tempImage1, channelSplit.at(i), tempImage1);

					custShow(tempImage1);

					break;
				case 2:
					cv::threshold(channelSplit.at(i), channelSplit.at(i),  0.5*MAXVAL, MAXVAL, cv::THRESH_BINARY_INV);

					cv::bitwise_and(tempImage1, channelSplit.at(i), tempImage1);

					break;
			}

			i++;
		}
		//
		cv::morphologyEx(tempImage1, tempImage1, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)), cv::Point(-1,-1), 2);

		custShow(tempImage1);

		tempImage1.release();
		//tempImage2.release();
		channelSplit.clear();
	}
	catch(...)
	{
		std::cout << "Binary finder failure. Unknown exception";

		cv::waitKey(TIMETOWAIT);

		return;
	}

	
	//cv::adaptiveThreshold(
	//cv::threshold(tmpImage, tmpImage
}

void ProhibitedSignFinder:: signClassify (cv::Mat &curFrame)
{
}