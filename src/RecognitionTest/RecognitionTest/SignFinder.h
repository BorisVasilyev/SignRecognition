#pragma once

#include "SignFinderBase.h"

class ProhibitedSignFinder : public SignFinderBase
{
public:
	void signDetect (cv::Mat sFrame);
	void signClassify (cv::Mat sFrame);
};