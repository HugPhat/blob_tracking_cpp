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

using namespace cv;


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
	void toBox(std::vector<KeyPoint> &);

};

