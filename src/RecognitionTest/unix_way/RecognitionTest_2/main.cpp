#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

class imageProcessor
{
public:
	imageProcessor()
	{

	}

	void process_image(Mat image)
	{
		Mat output;

		cv::cvtColor(image, output, CV_BGR2HSV);

		cout << "Rows :" << output.rows << "\nColumns: " << output.cols << "\n";

		cout << "Data type: " << output.type() << "\n";

		cout << "Data: " << output.at<uchar>(10,10) << "\n";

		// cout << output;

		/*
		int in_channels = image.channels();
		int out_channels = output.channels();

		cout << in_channels << " " << out_channels << "/n";

		int channels = output.channels();

		int nRows = output.rows;
		int nCols = output.cols;

		if (output.isContinuous())
		{
			nCols *= nRows;
			nRows = 1;
		}

		int i,j;
		uchar* p;

		for( i = 0; i < nRows; ++i)
		{
			p = output.ptr<uchar>(i);

			for ( j = 0; j < nCols; ++j)
			{
				p[j] = p[j] / 2;
			}
		}
		*/

		namedWindow( "Transformed Image", CV_WINDOW_AUTOSIZE );
  		imshow( "Transformed Image", output );
	}

	void test_method(Mat image)
	{
		/*
		Mat m(3, 3, CV_8UC3, Scalar(0,0,255));

		cout << m << endl;

		imshow("Test image", m);
		*/

		int channels = image.channels();

		int nRows = image.rows;
		int nCols = image.cols;

		if (image.isContinuous())
		{
			nCols *= nRows;
			nRows = 1;
		}

		int i,j;
		uchar* p;

		for( i = 0; i < nRows; ++i)
		{
			p = image.ptr<uchar>(i);

			for ( j = 0; j < nCols; ++j)
			{
				p[j*3 + 2] = 0;
			}
		}

		namedWindow( "Transformed Image", CV_WINDOW_AUTOSIZE );
  		imshow( "Transformed Image", image );
	}

	~imageProcessor()
	{

	}
};

int main( int argc, char** argv )
{
  Mat image;
  image = imread( argv[1], 1 );

  namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
  imshow( "Display Image", image );

  imageProcessor* processor = new imageProcessor();

  processor->process_image(image);

  // processor->test_method(image);

  waitKey(0);

  return 0;
}