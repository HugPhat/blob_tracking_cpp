#pragma once
// system
#include <iostream>
#include <string>
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
public:
	App(std::string);
	~App() {
		//delete blobmodel;
		//delete video;
	}
	
	void RunTracking(float);
	vector<TrackingBox> toBox(std::vector<KeyPoint> , float );
	cv::Point toCenter(Rect_<float>);

	double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);
};

