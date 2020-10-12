#include "App.h"


App::App(std::string path) {
	//video.open(path);
	video = new VideoCapture(path);
    blobmodel = new BlobModel();
    //ObjectPaths = new map<int, vector<cv::Point>* >();
    ObjectRandomColors = new map<int, Scalar>();
}

App::App(int path)
{
    //video.open(path);
    video = new VideoCapture(path);
    blobmodel = new BlobModel();
    //ObjectPaths = new map<int, vector<cv::Point>* >();
    ObjectRandomColors = new map<int, Scalar>();
}

void App::RunTracking(float ratio) {
    Mat result;
    Mat videoFrame;
    Mat detectImg;
    std::vector<KeyPoint> kps;
    std::string windowname = "Fish Food tracking";
	namedWindow(windowname, WINDOW_NORMAL);
    int frame_width = int((float)video->get(CAP_PROP_FRAME_WIDTH) / ratio);
    int frame_height = int((float)video->get(CAP_PROP_FRAME_HEIGHT) / ratio);
    video->set(CAP_PROP_FRAME_WIDTH, frame_width);
    video->set(CAP_PROP_FRAME_HEIGHT, frame_height);


    Scalar_<int> randColor[CNUM];
    // Kalman Config
    int max_age = 360;
    int min_hits = 1;
    double iouThreshold = 0.25;
    vector<KalmanTracker> trackers;
    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
    // variables used in the for-loop
    int frame_count = 0;
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    double cycle_time = 0.0;
    int64 start_time = 0;

    // Update 12-10

    unsigned long long totalObjects = 0;
    long newobjects_inCycle = 0;
    long newobjects_inCycle_updating = 0;
    long totalObject_inCycle = 0;
    long totalObject_inCycle_updating = 0;

    auto update_func = [&totalObjects, &newobjects_inCycle_updating, &totalObject_inCycle_updating]() {
        totalObjects++;
        newobjects_inCycle_updating++;
        totalObject_inCycle_updating++;
    };
    onTick(newobjects_inCycle, newobjects_inCycle_updating, totalObject_inCycle, totalObject_inCycle_updating);

    // End of 12-10

    if (!video->isOpened())
    {
        std::cout << "Can't open camera" << std::endl;
    }
    else {
        while (true) {
            // stream the video

            frame_count++;
            // read everyframe
            video->read( videoFrame);
            // resize for faster inference
            resize(videoFrame.clone(), videoFrame, Size(frame_width, frame_height));
            // detect
            detectImg = blobmodel->Detect(videoFrame, kps);
            // Convert center to Rect

        // Update new Object
            if (kps.size()) {

                vector<TrackingBox> detFrameData = toBox(kps, 5.);
                //
                if (trackers.size() == 0) // the first frame met
                {
                    // initialize kalman trackers using first detections.
                    for (unsigned int i = 0; i < detFrameData.size(); i++)
                    {
                        KalmanTracker trk = KalmanTracker(detFrameData[i].box);
                        trackers.push_back(trk);
                        // Here
                        registerNewObjects((int)(trk.m_id + 1), toCenter(detFrameData[i].box));
                        update_func();
                        
                    }

                    continue;
                }


                ///////////////////////////////////////
            // 3.1. get predicted locations from existing trackers.
                predictedBoxes.clear();

                for (auto it = trackers.begin(); it != trackers.end();)
                {
                    Rect_<float> pBox = (*it).predict();
                    if (pBox.x >= 0 && pBox.y >= 0)
                    {
                        predictedBoxes.push_back(pBox);
                        it++;
                    }
                    else
                    {
                        it = trackers.erase(it);
                        deregisterObjects(((*it).m_id));
                        //cerr << "Box invalid at frame: " << frame_count << endl;
                    }
                }
                ///////////////////////////////////////
            // 3.2. associate detections to tracked object (both represented as bounding boxes)
            // dets : detFrameData
                trkNum = predictedBoxes.size(); // preds
                detNum = detFrameData.size();   // reals 

                iouMatrix.clear();
                iouMatrix.resize(trkNum, vector<double>(detNum, 0));

                for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
                {
                    for (unsigned int j = 0; j < detNum; j++)
                    {
                        // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                        iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[j].box);
                    }
                }

                // solve the assignment problem using hungarian algorithm.
                // the resulting assignment is [track(prediction) : detection], with len=preNum
                HungarianAlgorithm HungAlgo;
                assignment.clear();
                HungAlgo.Solve(iouMatrix, assignment);

                // find matches, unmatched_detections and unmatched_predictions
                unmatchedTrajectories.clear();
                unmatchedDetections.clear();
                allItems.clear();
                matchedItems.clear();

                if (detNum > trkNum) //	there are unmatched detections
                {
                    for (unsigned int n = 0; n < detNum; n++)
                        allItems.insert(n);

                    for (unsigned int i = 0; i < trkNum; ++i)
                    {

                        matchedItems.insert(assignment[i]);
                    }

                    set_difference(allItems.begin(), allItems.end(),
                        matchedItems.begin(), matchedItems.end(),
                        insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
                }
                else if (detNum < trkNum) // there are unmatched trajectory/predictions
                {
                    for (unsigned int i = 0; i < trkNum; ++i)
                        if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                            unmatchedTrajectories.insert(i);
                }
                else
                    ;
                // filter out matched with low IOU
                matchedPairs.clear();
                for (unsigned int i = 0; i < trkNum; ++i)
                {
                    if (assignment[i] == -1) // pass over invalid values
                        continue;
                    if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
                    {
                        unmatchedTrajectories.insert(i);
                        unmatchedDetections.insert(assignment[i]);
                    }
                    else
                        matchedPairs.push_back(cv::Point(i, assignment[i]));
                }

                ///////////////////////////////////////
        //   // 3.3. updating trackers

            // update matched trackers with assigned detections.
            // each prediction is corresponding to a tracker
                int detIdx, trkIdx;
                for (unsigned int i = 0; i < matchedPairs.size(); i++)
                {
                    trkIdx = matchedPairs[i].x;
                    detIdx = matchedPairs[i].y;
                    trackers[trkIdx].update(detFrameData[detIdx].box);
                    //(ObjectPaths)[trkIdx ].push_back(toCenter(detFrameData[detIdx].box));
                }

                // create and initialise new trackers for unmatched detections
                for (auto umd : unmatchedDetections)
                {
                    KalmanTracker tracker = KalmanTracker(detFrameData[umd].box);
                    // here
                    registerNewObjects((int)(tracker.m_id + 1), toCenter(detFrameData[umd].box));
                    //cout << "add " << umd << endl;
                    trackers.push_back(tracker);
                    update_func();
                }

                // get trackers' output
                frameTrackingResult.clear();
                for (auto it = trackers.begin(); it != trackers.end();)
                {
                    if (((*it).m_time_since_update < 1) &&
                        ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
                    {
                        TrackingBox res;
                        res.box = (*it).get_state();
                        res.id = (*it).m_id + 1;
                        res.frame = frame_count;
                        frameTrackingResult.push_back(res);
                        //
                        (ObjectPaths)[res.id].push_back(toCenter(res.box));

                        it++;
                    }
                    else
                        it++;

                    // remove dead tracklet
                    if (it != trackers.end() && (*it).m_time_since_update > max_age) {
                        deregisterObjects((*it).m_id);
                        it = trackers.erase(it);
                    }
                }
                // End. Display result
                for (auto tb : frameTrackingResult) {
                    //cv::rectangle(videoFrame, tb.box, randColor[tb.id % CNUM], 2, 8, 0);

                    if ((ObjectPaths)[tb.id].size() > 2) {

                        for (int i = 0; i < (ObjectPaths)[tb.id].size() - 1; i++) {
                            cv::line(videoFrame,
                                ((ObjectPaths)[tb.id])[i],
                                ((ObjectPaths)[tb.id])[i + 1],
                                ((*ObjectRandomColors)[tb.id]),
                                2,   // thickness of line
                                CV_AA     // anti aliased line type
                            );

                        }

                    }

                    cv::circle(videoFrame,
                        toCenter(tb.box),
                        5,
                        (*ObjectRandomColors)[tb.id],
                        7);
                    cv::putText(videoFrame,
                        to_string(tb.id),
                        toCenter(tb.box),
                        cv::FONT_HERSHEY_DUPLEX,
                        1.0,
                        (*ObjectRandomColors)[tb.id], //font color
                        2);
                    
                    /// update 12-10
                    // Draw total objects
                    int posx = 0;
                    cv::putText(detectImg,
                                 "total objects: " + to_string(totalObjects),
                                Point(posx, 100),
                                cv::FONT_HERSHEY_DUPLEX,
                                1.0,
                                Scalar(255, 0, 0), //font color
                                2);
                    // new obj in the last 15s
                    cv::putText(detectImg,
                                "new objects in the last " + to_string(timerCycle) + "s: "
                                     + to_string(newobjects_inCycle),
                                Point(posx, 150),
                                cv::FONT_HERSHEY_DUPLEX,
                                1.0,
                                Scalar(180, 0, 0), //font color
                                2);
                    // total object in the last 15s
                    cv::putText(detectImg,
                                "total objects in the last " + to_string(timerCycle) + "s: " 
                                     + to_string(totalObject_inCycle),
                                Point(posx, 200),
                                cv::FONT_HERSHEY_DUPLEX,
                                1.0,
                                Scalar(180, 0, 0), //font color
                                2);
                }
            }
            /// 
            cv::hconcat(videoFrame, detectImg, result);
            cv::resizeWindow(windowname, 1300, 450);
            cv::imshow(windowname, result);
            if (waitKey(10) == 27){
               video->release();
               destroyWindow(windowname);
               break;
            }

        }
    }
}

void App::onTick(long &newobj, long &newobj_up, long &total, long &total_up)
{
    std::thread([&] {
        while (true)
        {
            // Wait timeCyle Seconds
            std::this_thread::sleep_for(std::chrono::seconds(timerCycle));
            // Call our method
            newobj = newobj_up;
            newobj_up = 0;
            total += total_up;
            total_up = 0;
        }
    }).detach();

}

vector<TrackingBox> App::toBox(std::vector<KeyPoint> kps, float calib)
{
    vector<TrackingBox> * tempKps = new vector<TrackingBox>();
    float x, y, r;
    for (auto kp : kps) {
        TrackingBox Box;
        Rect_<float> pBox;
        // center
        x = kp.pt.x;
        y = kp.pt.y;
        // size
        r = kp.size / 2 + calib ;
        //
        pBox.x = abs(x - r);
        pBox.y = abs(y - r);
        pBox.height = r + calib;
        pBox.width = r + calib;

        Box.box = pBox;
        //
        tempKps->push_back(Box);
    }

    return *tempKps;
}

cv::Point App::toCenter(Rect_<float> box)
{
    int x = box.x + box.width  * 0.5;
    int y = box.y + box.height * 0.5;

    return cv::Point(x, y);
}

void App::registerNewObjects(int id, cv::Point center)
{
    //(ObjectPaths)[id] = new vector<cv::Point>();
    //cout << "register id " << to_string(id) << endl;
    (ObjectPaths)[id].push_back(center);
    (*ObjectRandomColors)[id] = randomlyColor();
}

void App::deregisterObjects(int ind)
{
    ObjectPaths.erase(ind);
    ObjectRandomColors->erase( ind);
}

Scalar App::randomlyColor()
{
    return Scalar((rand() % 255 + 1) , (rand() % 255 + 1), (rand() % 255 + 1));
}

double App::GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}
