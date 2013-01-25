#pragma once
#include "StdAfx.h"
#include <iostream>
#include <vector>
#include <opencv\cv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>


class Sign
{
public:

	Sign () {};

	enum BaseColor {RED, BLUE, YELLOW};
	enum BaseShape {ROUND, SQUARE, RHOMB, TRIANGLE};

	virtual ~Sign () {};

	virtual BaseColor getBColor () {};

	virtual BaseShape getBShape () {};

	virtual cv::Mat* getTemplate () {};

};

class SingleSign : public Sign
{
private:
	BaseColor bColor;
	BaseShape bShape;

	cv::Mat* sTemplate;
public:
	SingleSign () {};
};

class SignList : Sign
{
public:
	vector <Sign*> sList;// список с шаблонами знаков
	int lastNumber; //текущее количество элементов списка
	int current;

	Sign* sSign;

	SignList () : lastNumber (0), sSign()
	{
		sSign = new SingleSign;
		sList.push_back(sSign);
	}

	~SignList ()
	{
		delete sSign;
	}
};

