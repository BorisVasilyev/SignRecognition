#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    Mat r_img1, r_img2;

    resize(img_1, r_img1, Size(150, 150), 0, 0);
    resize(img_2, r_img2, Size(150, 150), 0, 0);

    int minHessian = 500;

    std::vector<KeyPoint> keypoints_1, keypoints_2;

    FAST(r_img1, keypoints_1, 10);
    FAST(r_img2, keypoints_2, 10);

    //detector.detect( r_img1, keypoints_1 );
    //detector.detect( r_img2, keypoints_2 );

    SurfDescriptorExtractor extractor;
    Mat descriptors_1, descriptors_2;

    extractor.compute( r_img1, keypoints_1, descriptors_1 );
    extractor.compute( r_img2, keypoints_2, descriptors_2 );

    FlannBasedMatcher matcher;

    vector< vector<DMatch> > matches;

    matcher.knnMatch(descriptors_1, descriptors_2, matches, 50);

    vector< DMatch > good_matches;
    good_matches.reserve(matches.size());  
   
    for (size_t i = 0; i < matches.size(); ++i)
    { 
        if (matches[i].size() < 2)
                    continue;
       
        const DMatch &m1 = matches[i][0];
        const DMatch &m2 = matches[i][1];
            
        if(m1.distance <= 0.7 * m2.distance)        
            good_matches.push_back(m1);     
    }

    /*
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors_1.rows; i++ )
    { 
        double dist = matches[i].distance;

        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    { 
        if( matches[i].distance <= 2*min_dist )
        { 
            good_matches.push_back( matches[i]); 
        }
    }

    */

    Mat img_matches, good_img_matches;
    // drawMatches( r_img1, keypoints_1, r_img2, keypoints_2, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    drawMatches( r_img1, keypoints_1, r_img2, keypoints_2, good_matches, good_img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cout << "Total matches count: " << matches.size() << endl; 
    cout << "Good matches count: " << good_matches.size() << endl;

    // imshow( "All Matches", img_matches );
    imshow( "Good Matches", good_img_matches );

    waitKey(0);

    return 0;
}