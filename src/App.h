#pragma once
// system
#include <iostream>
#include <string>
#include <stdlib.h>
#include <thread>
// cv
#include "opencv2/core.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
// user
#include "BlobModel.h"
#include "KalmanTracker.h"
#include "Hungarian.h"

#define CNUM 20

using namespace cv;
using namespace std;

typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;


class App
{
private:
	VideoCapture * video;
	BlobModel * blobmodel;

	// store path
	map<int, vector<cv::Point>> ObjectPaths;
	map<int, Scalar> * ObjectRandomColors;

public:
	App(std::string);
	App(int);
	~App() {
		//delete blobmodel;
		//delete video;
	}
	
	const int timerCycle = 15; // seconds
	void RunTracking(float);
	vector<TrackingBox> toBox(std::vector<KeyPoint> , float );
	cv::Point toCenter(Rect_<float>);

	void registerNewObjects(int id, cv::Point);

	void  deregisterObjects(int id);

	Scalar randomlyColor();

	double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);

	void onTick(long&, long&, long&, long&);
};

