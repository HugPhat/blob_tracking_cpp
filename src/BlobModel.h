#pragma once
#include "opencv2/opencv.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

class BlobModel
{
private:
	Ptr<SimpleBlobDetector> detector;
public:
	BlobModel();
	~BlobModel() {};
	Mat Detect(Mat, std::vector<KeyPoint> &);
	
};

