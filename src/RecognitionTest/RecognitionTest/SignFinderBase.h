#pragma once
#include "StdAfx.h"
#include <opencv\cv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

#include <iostream>

#define TIMETOWAIT 10000

class SignFinderBase
{
public:
	SignFinderBase () {};
	virtual ~SignFinderBase () {};
	
	virtual void signDetect (cv::Mat &sFrame) = 0;
	virtual void signClassify (cv::Mat &sFrame) = 0;
};