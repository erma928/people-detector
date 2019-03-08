//
// Created by Feng Jimin on 12/7/18.
//

#ifndef LIFTDOOR_DETECT_H
#define LIFTDOOR_DETECT_H

#include <opencv2/opencv.hpp>

using namespace cv;

namespace detect {

    struct RectDetect {
        Rect rect;
        int seqCount;
    };

    /**
     * returns true if the given 3 channel image is B = G = R
     */
    bool isGrayImage(Mat& img);

    class ContourDetector {

        RectDetect loadRecentDetect();

        void emptyRecentDetect();

        void storeRecentDetect(RectDetect detect);



        void detect(Mat& img, Mat& grayBackground);
    };

};




#endif //LIFTDOOR_DETECT_H
