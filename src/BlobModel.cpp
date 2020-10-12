#include "BlobModel.h"

BlobModel::BlobModel() {
	SimpleBlobDetector::Params params;

    //Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 189;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 14;
    params.maxArea = 7800;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.1;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.88;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
	
    
    // Set up detector with params
    detector = SimpleBlobDetector::create(params);

}

Mat BlobModel::Detect(Mat im, std::vector<KeyPoint> &kps) {
    Mat gray;
    cvtColor(im, gray, CV_BGR2GRAY);
    detector->detect(gray, kps);
    if (kps.size() > 1) {
        Mat im_with_keypoints;
        drawKeypoints(im , kps, im_with_keypoints, Scalar(0, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        return im_with_keypoints;
    }
    else {
        return im;
    }
}