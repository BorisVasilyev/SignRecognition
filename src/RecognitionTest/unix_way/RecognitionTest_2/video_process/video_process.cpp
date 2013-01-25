#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <sstream>
#include <stdio.h>

using namespace cv;
using namespace std;

#define ERROR_STR "ERROR: " 
#define STATUS_STR "STATUS: "

/* Параметры для выбора красного цвета */

#define MINHUE 0
#define MINSAT 70
#define MINVAL 50
#define MAXHUE 15
#define MAXSAT 255
#define MAXVAL 255

int minhue = 0, maxhue = 15, minsat = 35, minval = 50;

/* Параметры для детектора кругов */

#define THRESHOLD1 40
#define THRESHOLD2 150
#define APERTURE_SIZE 3

int hi = 150, lo = 40;

/* Параметры для детектора особых точек */

#define SURF_THRESHOLD 10
#define SIGN_SIZE 150

/* Индекс для нумерации сохраняемых изображений */

int img_counter = 16;

class ImageProcessor
{
public:
	ImageProcessor(string train_img_dir)
	{
		_detector = SurfFeatureDetector(SURF_THRESHOLD);

		/* Загружаем изображения эталонов и вычисляем дескрипторы */

		cout << STATUS_STR << "Training dir: " << train_img_dir << endl;

		_train_img_dir = train_img_dir;

		addTrainImage("30");
		addTrainImage("40");
		addTrainImage("50");
		addTrainImage("60");
		addTrainImage("70");
	}

	Mat process_image(Mat image)
	{
		Mat im = image.clone();
		std::vector<cv::Mat> channelSplit(3);
		
		cvtColor(im, im, CV_BGR2HSV);

		Mat range1, range2;

		inRange(im, Scalar(160, minsat, minval), Scalar(180, MAXSAT, MAXVAL), range1);
		inRange(im, Scalar(0, minsat, minval), Scalar(5, MAXSAT, MAXVAL), range2);

		add(range1, range2, im);

		medianBlur(im, im, 5);

		//int dilation_type = MORPH_RECT; 
  		//int dilation_type = MORPH_CROSS;
  		int dilation_type = MORPH_ELLIPSE;

  		int dilation_size = 1;

		Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );

  		dilate( im, im, element);
	
		Canny(im, im, THRESHOLD1, THRESHOLD2, APERTURE_SIZE);
		
		vector<Vec3f> circles;

		HoughCircles(im, circles, CV_HOUGH_GRADIENT, 1, 50, hi > 0 ? hi : 1, lo > 0 ? lo : 1, 0, 100);

		if(circles.size() > 0)
		{
			cout << STATUS_STR << circles.size() << " circles found " << endl;

			int cornerX = 0;
			int cornerY = 0;
			int roiWidth = 0;
			int roiHeight = 0;

			for( size_t i = 0; i < circles.size(); i++ )
			{
				Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
				int radius = cvRound(circles[i][2]) + 5;

			  	/* Берем область из исходного изображения */

			  	cornerX = circles[i][0] - radius > 0 ? circles[i][0] - radius : 0;
				cornerY = circles[i][1] - radius > 0 ? circles[i][1] - radius : 0;
				roiWidth = circles[i][0] + 2*radius < image.cols ? 2*radius : 1;
				roiHeight = circles[i][1] + 2*radius < image.rows ? 2*radius : 1;

				/*
				cout << "cornerX = " << cornerX << endl;
				cout << "cornerY = " << cornerY << endl;
				cout << "width = " << roiWidth << endl;
				cout << "height = " << roiHeight << endl;

				cout << "image width = " << image.cols << endl;
				cout << "image height = " << image.rows << endl;
				*/

			  	Mat part(image, 
			  		Rect(cornerX, 
			  			cornerY, 
			  			roiWidth, 
			  			roiHeight)
			  		);

			  	resize(part, part, Size(SIGN_SIZE, SIGN_SIZE), 0, 0);

			  	Mat mask = Mat::zeros(part.size(), CV_8UC1);
			  	Mat partCopy(image.size(), 8, 3);

			  	circle(mask, Point(SIGN_SIZE / 2, SIGN_SIZE / 2), SIGN_SIZE / 2, Scalar(255), -1);

			  	part.copyTo(partCopy, mask);

			  	imshow("Found circles", mask);

			  	//tryFindImage_features(partCopy);
			  	//tryFindImage_templates(partCopy);

			  	save_image(partCopy, ++img_counter);

			  	circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );

			 	circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
			}
		}

		return im;
	}

	~ImageProcessor()
	{

	}
private:

	// vector<string> train_file_names;
	vector<Mat> _train_images;
	vector<string> _train_sign_names;
	vector< vector<KeyPoint> > _train_keypoints;
	vector<Mat> _train_descriptors;

	string _train_img_dir;

	SurfDescriptorExtractor _extractor;
    SurfFeatureDetector _detector;
    FlannBasedMatcher _matcher;

    void addTrainImage(string name)
    {
    	/* Добавление нового эталонного изображения и вычисление его дескриптора */

    	Mat train_img = imread(_train_img_dir + "template_" + name + ".jpg");

    	if(!train_img.empty())
    	{
			resize(train_img, train_img, Size(SIGN_SIZE, SIGN_SIZE), 0, 0);
			_train_images.push_back(train_img);
			_train_sign_names.push_back(name);

			vector<KeyPoint> points;
			_detector.detect( train_img, points );
			_train_keypoints.push_back(points);

			Mat descriptors;
			_extractor.compute( train_img, points, descriptors);
			_train_descriptors.push_back(descriptors);	
		}
		else
		{
			cout << ERROR_STR << "Could not load train image " << _train_img_dir << name << ".jpg" << endl;
		}
    }

    void tryFindImage_features(Mat input)
    {
    	/* Сравниваем входящее изрображение с набором эталонов и выбираем наиболее подходящее */

    	resize(input, input, Size(SIGN_SIZE, SIGN_SIZE), 0, 0);

    	vector<KeyPoint> keyPoints;
    	_detector.detect(input, keyPoints);

    	Mat descriptors;
    	_extractor.compute(input, keyPoints, descriptors);

    	int max_value = 0, max_position = 0; 

    	for(int i=0; i < 5; i++)
    	{
    		vector< vector<DMatch> > matches;

    		_matcher.knnMatch(descriptors, _train_descriptors[i], matches, 50);

    		int good_matches_count = 0;
		   
		    for (size_t j = 0; j < matches.size(); ++j)
		    { 
		        if (matches[j].size() < 2)
		                    continue;
		       
		        const DMatch &m1 = matches[j][0];
		        const DMatch &m2 = matches[j][1];
		            
		        if(m1.distance <= 0.7 * m2.distance)        
		            good_matches_count++;    
		    }

		    if(good_matches_count > max_value)
		    {
		    	max_value = good_matches_count;
		    	max_position = i;
		    }
    	}

    	cout << STATUS_STR << "Detected sign: " << _train_sign_names[max_position] << endl;
    }

    void tryFindImage_templates(Mat input)
    {
    	/* Сравниваем входящее изрображение с набором эталонов и выбираем наиболее подходящее */

    	resize(input, input, Size(SIGN_SIZE, SIGN_SIZE), 0, 0);

    	//save_image(input, ++img_counter);

    	int max_position = 0;
    	double max_diff = 0;

    	for(int i=0; i<_train_images.size(); i++)
    	{
    		Mat result;

    		resize(_train_images[i], _train_images[i], Size(80, 50),0,0);

    		stringstream ss;

    		ss << "./images/output/template_" << i << ".jpeg";

    		imwrite(ss.str(), _train_images[i]);

    		int result_cols =  input.cols - _train_images[i].cols + 1;
		  	int result_rows = input.rows - _train_images[i].rows + 1;

		  	result.create( result_cols, result_rows, CV_32FC1 );

		  	matchTemplate( input, _train_images[i], result, CV_TM_SQDIFF );

		  	double minVal; double maxVal; Point minLoc; Point maxLoc;
		  	Point matchLoc;

		  	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

		  	if(max_diff < maxVal)
		  	{
		  		max_diff = maxVal;
		  		max_position = i;
		  	}

		  	cout << "Max value for " << _train_sign_names[i] << " : " << (maxVal*0.00000001) << endl;
    	}

    	cout << STATUS_STR << "Detected sign: " << _train_sign_names[max_position] << endl;
    }

    void save_image(Mat img, int counter)
    {
    	/* Сохраняем изображение на диск под заданным номером */

    	stringstream ss;

    	ss << "./images/output/img_" << counter << ".jpeg";

    	imwrite(ss.str(), img);
    }

};

int main( int argc, char** argv )
{
	if(argc == 3)
	{
		ImageProcessor* proc = new ImageProcessor(argv[2]);

		VideoCapture cap;

		if(!cap.open(argv[1]))
		{
			cout << ERROR_STR << "Failed to open " << argv[1] << endl;
			return -1;
		}

		namedWindow("Video output", 1);
		namedWindow("Input", 1);

		namedWindow("Settings", 1);
		createTrackbar("hi", "Settings", &hi, 255);
	    createTrackbar("lo", "Settings", &lo, 255);

	    //createTrackbar("min_hue", "Settings", &minhue, 180);
	    //createTrackbar("max_hue", "Settings", &maxhue, 180);
	    createTrackbar("sat", "Settings", &minsat, 255);
	    createTrackbar("val", "Settings", &minval, 255);

	    namedWindow("Found circles", 1);

		for(;;)
		{
			Mat curFrame;

			cap >> curFrame;

			if (curFrame.empty())
			{
				cout << STATUS_STR << "Frame is empty" << endl;
				break;
			}
			else
			{
				//resize(curFrame, curFrame, Size(), 0.4, 0.4, INTER_AREA);

				Mat fr = proc->process_image(curFrame);

				imshow("Video output", fr);
				imshow("Input", curFrame);

				// cout << "hi = " << hi << " lo = " << lo << endl;

				if(fr.empty())
				{
					cout << STATUS_STR << "Processed frame is empty" << endl;
					break;
				}
			}

			cvWaitKey(5);

			if(waitKey(30) >= 0)
				break;
		}
	}
	else
	{
		cout << ERROR_STR << "Invalid input parameters. Need to specify video file and directory path to training images." << endl;

		return -1;
	}

	waitKey(0);

	return 0;
}
