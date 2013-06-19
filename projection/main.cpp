#include <iostream>
#include <cv.h>
#include <opencv2/highgui/highgui_c.h>
#include <string>
#include <fstream>
#include "Window.h"
#include "verticalProjection.h"
#include "HorizontalProjection.h"
#include "DIPController.h"
#include "MyImage.h"
#include "DataBaseController.h"
#include "CarLogoDetector.h"
#include "dir.h"
#include "Saver.h"

#include <boost/smart_ptr.hpp>
#include <boost/timer.hpp>

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
using namespace cv;

using namespace std;
using namespace BasicCvApi;
using namespace boost;

//////////////////////////////////////////////////////////////////////////
// if define DETECT, output the logo class
//otherwise, it will save the coarse car logo to the directory
// for the test
//////////////////////////////////////////////////////////////////////////

#define  DETECT

int main(int argc,char *argv[])
{

	if (argc != 4)
	{
		printf("error input string\n");
		system("PAUSE");
		exit(0);
	}

	string filesLocation(argv[1]),saveDirectory(argv[2]),templateLocation(argv[3]);
	// cout << "filesLocation:" << filesLocation << endl;
	// cout << "saveDirectory:" << saveDirectory << endl;
	// cout << "templateLocation:" << templateLocation << endl;
	 
	CarLogoDetector detector;
	DataBaseController* databaseController = new DataBaseController(templateLocation);

	


	//load in the locations
	vector<string> files = dir::readImages(filesLocation.c_str());
	filesLocation.append("\\");
	saveDirectory.append("\\");
	string fullpath;
	string saveLoc;
	
	vector<string>::iterator it;

	string fileOutput = filesLocation + saveDirectory + "z.out";
	ofstream fout(fileOutput);

	//load in the data

	boost::timer timer;
#ifdef DETECT	
	detector.initDataBase(databaseController);	
	fout<<"init database success"<<endl<<"time: "<<timer.elapsed()<<"s"<<endl;
	cout<<"init database success"<<endl<<"time: "<<timer.elapsed()<<"s"<<endl;
#endif
	

	fout<<"there are totally "<<files.size()<<" images"<<endl;

	string className;
	Saver* saver = new Saver(string(filesLocation + saveDirectory));

	for ( it = files.begin(); it< files.end(); it++)
	{
		timer.restart();
		fout<<"//////////////////////////////////////////////////////////////////////////"<<endl;
		fout<<"current file: "<<*it <<endl;
		cout<<"current file: "<<*it <<endl;

		fullpath = filesLocation + *it;
		saveLoc = filesLocation + saveDirectory + *it;
		IplImage* image = cvLoadImage(fullpath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		BasicCvApi::MyImagePtr imagePtr(image);

		// get the coarse logo 
		IplImage* logo = detector.getCoarseLogoAreaFromImage(image);
		cvSaveImage("hihi.jpg",logo);

		IplImage* image1 = cvLoadImage("C:\\logo\\tlogo.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//BasicCvApi::MyImagePtr imagePtr1(image1);
// get the coarse logo
//IplImage* fetch_result = detector.getCoarseLogoAreaFromImage(image);
//cvSaveImage("fetch_result.jpg",fetch_result);
cv::Mat fetch_result_MAT(logo,0); 
//show surf matching.
// vector of keypoints
cv::Mat outImg1, outImg2;
vector<cv::KeyPoint> keypoints1, keypoints2;

// Read input images
cv::Mat logoImg_MAT(image1,0);
cv::namedWindow("logoImg_MAT");
imshow("logoImg_MAT", logoImg_MAT);
cv::SurfFeatureDetector surf(2500);

surf.detect(fetch_result_MAT, keypoints1);
surf.detect(logoImg_MAT, keypoints2);
drawKeypoints(fetch_result_MAT, keypoints1, outImg1, Scalar(255,255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
drawKeypoints(logoImg_MAT, keypoints2, outImg2, Scalar(255,255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

//cv::namedWindow("SURF detector img1");
//imshow("SURF detector img1", outImg1);

//cv::namedWindow("SURF detector img2");
//imshow("SURF detector img2", outImg2);

SurfDescriptorExtractor surfDesc;
Mat descriptors1, descriptors2;
surfDesc.compute(fetch_result_MAT, keypoints1, descriptors1);
surfDesc.compute(logoImg_MAT, keypoints2, descriptors2);

BruteForceMatcher< L2<float> > matcher;
vector<DMatch> matches;
matcher.match(descriptors1,descriptors2, matches);

nth_element(matches.begin(), matches.begin()+24, matches.end());
matches.erase(matches.begin()+25, matches.end());

Mat imageMatches;
drawMatches(fetch_result_MAT, keypoints1, logoImg_MAT , keypoints2, matches, imageMatches, Scalar(255,255,255));

namedWindow("Matched");
imshow("Matched", imageMatches);

cv::waitKey();

	#ifdef DETECT
	//show coarse result
		Window m_window(*it);
		m_window.show(logo,0);

		// detect the coarse logo
		className = detector.detectLogoImage(logo);
		if(className != "no")
		{
			detector.outputTextResult(fout);
			detector.saveResultsImgs(saver);
		}
		


		fout<<"final className: "<<className<<endl
			<<"timer "<<timer.elapsed()<<"s"<<endl;

		
#endif
		cout<<"still "<<files.end() - it <<" files..."<<endl;

#ifndef DETECT
	cvSaveImage(saveLoc.c_str(), logo,0);
#endif
	}
	
	cout<<"OK"<<endl;
	fout.close();





 	 return 0;
}