//
// Created by Feng Jimin on 12/7/18.
//
#include <opencv2/opencv.hpp>
#include "ContourDetecter.h"

using namespace cv;
using namespace std;

namespace detect {
    const char* DETECT_DATA_FILE = "non_human_obj_detect.yml";
    const double PEOPLE_CONTOUR_AREA_LOW = 400;
    const double PEOPLE_CONTOUR_AREA_HIGH = 16000;
    const float OVERLAPPING_THRESH = 0.96;
    const int SEQ_COUNT_THRESH = 6;

    bool isGrayImage(Mat& img) {
        Mat imgResize, dst;
        Mat bgr[3];

        resize(img, imgResize, Size(img.cols/4, img.rows/4));
        split( imgResize, bgr );
        absdiff( bgr[0], bgr[1], dst );

        if(countNonZero( dst ))
            return false;

        absdiff( bgr[0], bgr[2], dst );
        return !countNonZero( dst );
    }

    RectDetect ContourDetector::loadRecentDetect() {
        FileStorage fs(DETECT_DATA_FILE, FileStorage::READ);

        Rect rect; int count = 0;
        fs["rect"] >> rect;
        fs["seqCount"] >> count;

        RectDetect result;
        result.rect = rect;
        result.seqCount = count;

        fs.release();

        return result;

    }

    void ContourDetector::emptyRecentDetect() {
        FileStorage fs(DETECT_DATA_FILE, FileStorage::WRITE);
        fs << "seqCount" << 0;
        fs << "rect" << Rect();
        fs.release();
    }

    void ContourDetector::storeRecentDetect(RectDetect detect) {
        FileStorage fs(DETECT_DATA_FILE, FileStorage::WRITE);

        fs << "seqCount" << detect.seqCount;
        fs << "rect" << detect.rect;

        fs.release();
    }

    void ContourDetector::detect(Mat& inputImage, Mat& grayBackground) {
        Mat grayImage, threshImageLow, threshImageHigh;
        cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
        Mat diffImage, threshDiffImage;

        absdiff(grayBackground, grayImage, diffImage);
        threshold(diffImage, threshDiffImage, 127, 255, THRESH_BINARY | THRESH_OTSU);
        erode(threshDiffImage, threshDiffImage, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)), Point(-1, -1), 2);
        dilate(threshDiffImage, threshDiffImage, getStructuringElement(MORPH_ELLIPSE, Size(8, 3)), Point(-1, -1), 2);

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(threshDiffImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<vector<Point> > contoursPoly(contours.size());
        vector<Rect> boundRects;
        for (size_t i = 0; i < contours.size(); i++) {
            approxPolyDP(contours[i], contoursPoly[i], 3, true);
            double area = contourArea(contoursPoly[i]);
            if (area > PEOPLE_CONTOUR_AREA_LOW && area < PEOPLE_CONTOUR_AREA_HIGH) {
                boundRects.push_back(boundingRect(contours[i]));
            }
        }

        // if only one detection exists, then it possibly indicates that it is not human
        if (boundRects.size()==1) {
            RectDetect detect = loadRecentDetect();
            Rect& r1 = boundRects.at(0);
            if (detect.seqCount>0) {
                Rect o1 = r1 & detect.rect;
                Rect o2 = r1 | detect.rect;

                if (o1.area()>o2.area()*OVERLAPPING_THRESH) {
                    detect.seqCount += 1;
                    storeRecentDetect(detect);
                    if (detect.seqCount>SEQ_COUNT_THRESH) { //means non-human object detected
                    }
                } else { // means it is not an static object
                    RectDetect newDetect;
                    newDetect.rect = r1;
                    newDetect.seqCount = 1;
                    storeRecentDetect(newDetect);
                }
            } else { // means it is the first object detection
                RectDetect newDetect;
                newDetect.rect = r1;
                newDetect.seqCount = 1;
                storeRecentDetect(newDetect);
            }
        } else { // no non-human object detected
            emptyRecentDetect();
        }
    }
}