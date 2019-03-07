//
// Created by Feng Jimin on 3/7/19.
//

#ifndef LIFTDOOR_DARKNETYOLODETECTOR_H
#define LIFTDOOR_DARKNETYOLODETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace detect {

    class DnnPeopleDetector {
        bool swapRB;
        float scale, confThreshold, nmsThreshold;
        int inpWidth, inpHeight;
        cv::Scalar mean;
        std::vector <std::string> classes;
        cv::dnn::Net net;

    public:
        DnnPeopleDetector(std::string config, std::string model,
                            std::string classesFile, float scale, cv::Scalar mean,
                            float confThreshold, float nmsThreshold, bool swapRB,
                            int inpWidth, int inpHeight, int backend, int target);

        void detect(cv::Mat& img, cv::Mat& blob);

        void setConfThreshold(float confThreshold);

    protected:
        void postprocess(cv::Mat &frame, const std::vector <cv::Mat> &out);

        void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame);
    };

}
#endif //LIFTDOOR_DARKNETYOLODETECTOR_H
