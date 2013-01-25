#pragma once
#include "SignFinderBase.h"

class ProhibitedSignFinder : public SignFinderBase
{
private:
	cv::Mat sFrame;//основная переменная для кадра

	cv::Mat tempImage1, tempImage2;

	std::string workWindowDesc;

public:
	ProhibitedSignFinder(const cv::Mat &curFrame) : SignFinderBase() {
		sFrame = curFrame.clone();

		cv::namedWindow(workWindowDesc = "workWindow", 1);
		//tempImage = &curFrame.clone();
	}

	~ProhibitedSignFinder () {

		cv::destroyWindow(workWindowDesc);
		sFrame.release();
		//tempImage->release();
	};

	virtual void signDetect (cv::Mat &curFrame);
	virtual void signClassify (cv::Mat &curFrame);

private:
	void custShow (const cv::Mat& showIm) const
	{
		cv::imshow(workWindowDesc, showIm);

		cv::waitKey(0);
	}
};