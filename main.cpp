#include <opencv2/opencv.hpp>

#include "common.hpp"
#include "common/ContourDetecter.h"
#include "dnn/DnnPeopleDetector.h"

using namespace cv;
using namespace dnn;
using namespace detect;
using namespace std;

string keys =
        "{ help  h     | | Print help message. }"
        "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
        "{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
        "{ device      |  0 | camera device number. }"
        "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
        "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
        "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
        "{ thr         | .5 | Confidence threshold. }"
        "{ nms         | .4 | Non-maximum suppression threshold. }"
        "{ backend     |  0 | Choose one of computation backends: "
        "0: automatically (by default), "
        "1: Halide language (http://halide-lang.org/), "
        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
        "3: OpenCV implementation }"
        "{ target      | 0 | Choose one of target computation devices: "
        "0: CPU target (by default), "
        "1: OpenCL, "
        "2: OpenCL fp16 (half-float precision), "
        "3: VPU }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    const string modelName = parser.get<String>("@alias");
    const string zooFile = parser.get<String>("zoo");
    keys += genPreprocArguments(modelName, zooFile);
    parser = CommandLineParser(argc, argv, keys);
    parser.about("people detection using deep learning networks or common contour detection using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    float confThreshold = parser.get<float>("thr");
    float nmsThreshold = parser.get<float>("nms");
    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");
    CV_Assert(parser.has("model"));
    string modelPath = findFile(parser.get<String>("model"));
    string configPath = findFile(parser.get<String>("config"));
    string classes = findFile(parser.get<String>("classes"));

    DnnPeopleDetector dnnPeopleDetector(configPath, modelPath, classes, scale, mean,
                                            confThreshold, nmsThreshold, swapRB,
                                            inpWidth, inpHeight,
                                            backend, target);
    // Create a window
    static const string kWinName = "Deep learning people detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    int initialConf = (int)(confThreshold * 100);
    createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback, &dnnPeopleDetector);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(parser.get<int>("device"));
    // Process frames.
    Mat frame, blob;
    while (waitKey(1) < 0) {
        cap >> frame;
        if (frame.empty()) {
            waitKey();
            break;
        }
        dnnPeopleDetector.detect(frame, blob);
        imshow(kWinName, frame);
    }
    return 0;
}
