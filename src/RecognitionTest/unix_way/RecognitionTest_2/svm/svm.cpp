#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <dirent.h>

using namespace cv;
using namespace std;

#define ERROR_STR "ERROR: " 
#define STATUS_STR "STATUS: "

// Стандартный размер картинки
#define IMG_SIZE 150
#define TRAIN_VECTOR_LENGTH 2*IMG_SIZE + 3

/* Параметры для выбора красного цвета */

#define MINHUE 0
#define MINSAT 70
#define MINVAL 50
#define MAXHUE 15
#define MAXSAT 255
#define MAXVAL 255

int minhue = 0, maxhue = 15, minsat = 35, minval = 0;

/* Параметры для детектора кругов */

#define THRESHOLD1 45
#define THRESHOLD2 150
#define APERTURE_SIZE 3

int hi = 150, lo = 45;

#define SIGN_SIZE 150

// Модель
SVM svm;

HOGDescriptor hog;

// Имена знаков в соответствии с классами для модели
string sign_names[] = {"20", "30", "40", "50", "70"};

/* Возвращает имена всех файлов в данной директории */
vector<string> get_file_names(string path)
{
	vector<string> res;

	DIR* dir;

	struct dirent *ent;
	dir = opendir(path.c_str());

	if (dir != NULL)
	{
		while ((ent = readdir(dir)) != NULL)
		{
			if(ent->d_name[0] != '.')
				res.push_back(path + ent->d_name);
		}

		closedir(dir);
	}
	else
	{
		cout << ERROR_STR << "Could not open dir " << path << endl;
	}

	return res;
}

/* Вычисление вектора признаков изображения */
vector<float> compute_vector(Mat img)
{
	vector<float> res;

	res.reserve(TRAIN_VECTOR_LENGTH);

	Mat input = img.clone();

	resize(input, input, Size(IMG_SIZE, IMG_SIZE), 0, 0, CV_INTER_LINEAR);

	Mat mask = Mat::zeros(input.size(), CV_8UC1);
  	Mat partCopy(input.size(), 8, 3);

  	circle(mask, Point(SIGN_SIZE / 2, SIGN_SIZE / 2), SIGN_SIZE / 2, Scalar(255), -1);

  	input.copyTo(partCopy, mask);

	cv::Scalar s = mean(partCopy);

	res.push_back(s[0] / 255);
	res.push_back(s[1] / 255);
	res.push_back(s[2] / 255);

	cvtColor(partCopy, partCopy, CV_BGR2GRAY);

	equalizeHist( partCopy, partCopy );

	for(int i=0; i<IMG_SIZE; i++)
	{
		Scalar r = mean(partCopy.row(i));

		res.push_back(r[0] / IMG_SIZE);
	}

	for(int i=0; i<IMG_SIZE; i++)
	{
		Scalar r = mean(partCopy.col(i));

		res.push_back(r[0] / IMG_SIZE);
	}

	return res;
}

/* Получение массива данных для обучения модели */
void get_training_data(Mat& trainData, Mat& responses, string imageDir)
{
	vector<string> file_names_20 = get_file_names(imageDir + "20/");
	vector<string> file_names_30 = get_file_names(imageDir + "30/");
	vector<string> file_names_40 = get_file_names(imageDir + "40/");
	vector<string> file_names_50 = get_file_names(imageDir + "50/");
	vector<string> file_names_70 = get_file_names(imageDir + "70/");

	cout << "No of 20 images: " << file_names_20.size() << endl;
	cout << "No of 30 images: " << file_names_30.size() << endl;
	cout << "No of 40 images: " << file_names_40.size() << endl;
	cout << "No of 50 images: " << file_names_50.size() << endl;
	cout << "No of 70 images: " << file_names_70.size() << endl;

	int train_img_count = file_names_20.size()
					+ file_names_30.size()
					+ file_names_40.size()
					+ file_names_50.size()
					+ file_names_70.size();

	cout << "Total file count: " << train_img_count << endl;

	trainData.create(train_img_count, TRAIN_VECTOR_LENGTH, CV_32FC1);
	responses.create(train_img_count, 1, CV_32FC1);

	cout << "Train data: rows: " << trainData.rows << " columns: " << trainData.cols << endl;
	cout << "Responses data: rows: " << responses.rows << " columns: " << responses.cols << endl;

	int cur_row = 0;

	for(int i=0; i<file_names_20.size(); i++)
	{
		vector<float> vec = compute_vector(imread(file_names_20[i]));

		for(int j=0; j<vec.size(); j++)
		{
			trainData.at<float>(cur_row, j) = vec[j];
		}

		responses.at<float>(cur_row, 0) = 1.0;

		cur_row++;
	}

	for(int i=0; i<file_names_30.size(); i++)
	{
		vector<float> vec = compute_vector(imread(file_names_30[i]));

		for(int j=0; j<vec.size(); j++)
		{
			trainData.at<float>(cur_row, j) = vec[j];
		}

		responses.at<float>(cur_row, 0) = 2.0;

		cur_row++;
	}

	for(int i=0; i<file_names_40.size(); i++)
	{
		vector<float> vec = compute_vector(imread(file_names_40[i]));

		for(int j=0; j<vec.size(); j++)
		{
			trainData.at<float>(cur_row, j) = vec[j];
		}

		responses.at<float>(cur_row, 0) = 3.0;

		cur_row++;
	}

	for(int i=0; i<file_names_50.size(); i++)
	{
		vector<float> vec = compute_vector(imread(file_names_50[i]));

		for(int j=0; j<vec.size(); j++)
		{
			trainData.at<float>(cur_row, j) = vec[j];
		}

		responses.at<float>(cur_row, 0) = 4.0;

		cur_row++;
	}

	for(int i=0; i<file_names_70.size(); i++)
	{
		vector<float> vec = compute_vector(imread(file_names_70[i]));

		for(int j=0; j<vec.size(); j++)
		{
			trainData.at<float>(cur_row, j) = vec[j];
		}

		responses.at<float>(cur_row, 0) = 5.0;

		cur_row++;
	}
}

/* Обучение модели */
void train_svm(string img_dir)
{
	Mat trainData;
	Mat responses;

	get_training_data(trainData, responses, img_dir);

	CvSVMParams params;

    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    params.gamma = 1;
    params.nu = 0.5;

	svm.train(trainData, responses, Mat(), Mat(), params);
}

string tryFindImage(Mat input)
{
	vector<float> test_vec = compute_vector(input);

	Mat test_data(1, TRAIN_VECTOR_LENGTH, CV_32FC1);

	for(int i=0; i<test_vec.size(); i++)
	{
		test_data.at<float>(0, i) = test_vec[i];
	}

	float res = svm.predict(test_data);

	cout << "Result: " << sign_names[(int)res - 1] << endl;

	return sign_names[(int)res - 1];
}

class ImageProcessor
{
public:
	ImageProcessor()
	{
	}

	Mat process_image(Mat image)
	{
		Mat im = image.clone();
		std::vector<cv::Mat> channelSplit(3);
		
		cvtColor(im, im, CV_BGR2HSV);

		Mat range1, range2;

		inRange(im, Scalar(160, minsat, minval), Scalar(180, MAXSAT, MAXVAL), range1);
		inRange(im, Scalar(0, minsat, minval), Scalar(10, MAXSAT, MAXVAL), range2);

		add(range1, range2, im);

		//medianBlur(im, im, 5);

		//int dilation_type = MORPH_RECT; 
  		//int dilation_type = MORPH_CROSS;
  		int dilation_type = MORPH_ELLIPSE;

  		int dilation_size = 1;

		Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );

  		//dilate( im, im, element);
	
		Canny(im, im, THRESHOLD1, THRESHOLD2, APERTURE_SIZE);
		
		vector<Vec3f> circles;

		HoughCircles(im, circles, CV_HOUGH_GRADIENT, 1, 50, hi > 0 ? hi : 1, lo > 0 ? lo : 1, 50, 100);

		if(circles.size() > 0)//&& circles.size() < 3)
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
				
			  	cornerX = circles[i][0] - radius > 0 ? circles[i][0] - radius : 0;
				cornerY = circles[i][1] - radius > 0 ? circles[i][1] - radius : 0;
				
				roiWidth = circles[i][0] + 2*radius < image.cols ? 2*radius : 1;
				roiHeight = circles[i][1] + 2*radius < image.rows ? 2*radius : 1;

			  	Mat part(image, 
			  		Rect(cornerX, 
			  			cornerY, 
			  			roiWidth, 
			  			roiHeight)
			  		);

			  	resize(part, part, Size(SIGN_SIZE, SIGN_SIZE), 0, 0);

			  	string sign = tryFindImage(part);
			  		
			  	imshow("Found circles", part);
			  	
			  	circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );

			 	circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );

			 	putText(image, sign, cvPoint(center.x + radius, center.y - radius), 
			 			CV_FONT_HERSHEY_SIMPLEX, 1.2, cvScalar(0,0,250), 2, CV_AA);
			 	
			}
		}

		return im;
	}

	~ImageProcessor()
	{
	}

};

int main( int argc, char** argv )
{
	if(argc == 2)
	{
		ImageProcessor* proc = new ImageProcessor();

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

		train_svm("/home/boris/src/driveAssist_2/res/images/train_2/");

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
				// resize(curFrame, curFrame, Size(), 0.5, 0.5, INTER_AREA);

				Mat fr = proc->process_image(curFrame);

				imshow("Video output", fr);
				imshow("Input", curFrame);

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
		cout << ERROR_STR << "Invalid input parameters. Need to specify video file." << endl;

		return -1;
	}

	waitKey(0);

	return 0;
}
