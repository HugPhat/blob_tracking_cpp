#include "App.h"


App::App(std::string path) {
	//video.open(path);
	video = new VideoCapture(path);
    blobmodel = new BlobModel();
}

void App::RunTracking(float ratio) {
    Mat result;
    Mat videoFrame;
    Mat detectImg;
    std::vector<KeyPoint> kps;
    std::string windowname = "Fish Food tracking";
	//namedWindow("Fish Food tracking", WINDOW_AUTOSIZE);
    int frame_width = int((float)video->get(CAP_PROP_FRAME_WIDTH) / ratio);
    int frame_height = int((float)video->get(CAP_PROP_FRAME_HEIGHT) / ratio);
    video->set(CAP_PROP_FRAME_WIDTH, frame_width);
    video->set(CAP_PROP_FRAME_HEIGHT, frame_height);
    //video->set(CV_CAP_PROP_FPS, 65);
    //video->set(CAP_PROP_BUFFERSIZE, 2);
    if (!video->isOpened()) {
        std::cout << "Can't open camera" << std::endl;
    }
    else {
        while (true) {
            video->read( videoFrame);
            resize(videoFrame.clone(), videoFrame, Size(frame_width, frame_height));
            detectImg = blobmodel->Detect(videoFrame, kps);

            for (auto point : kps) {
                std::cout << point.pt << std::endl;

            }

            hconcat(videoFrame, detectImg, result);
            imshow(windowname, result);
            if (waitKey(10) == 27){
               break;
               video->release();
               destroyWindow(windowname);
            }

        }
    }
}

void App::toBox(std::vector<KeyPoint>&)
{

}
