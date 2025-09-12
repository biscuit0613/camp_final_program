#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include <fstream>
#include "json.hpp"  // 使用 nlohmann::json

using json = nlohmann::json;

void saveCalibrationToJson(const Mat& cameraMatrix, const Mat& distCoeffs, const string& filename) {
    json j;

    // 相机内参矩阵
    j["camera_matrix"] = {
        {cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(0, 1), cameraMatrix.at<double>(0, 2)},
        {cameraMatrix.at<double>(1, 0), cameraMatrix.at<double>(1, 1), cameraMatrix.at<double>(1, 2)},
        {cameraMatrix.at<double>(2, 0), cameraMatrix.at<double>(2, 1), cameraMatrix.at<double>(2, 2)}
    };

    // 畸变系数
    j["distortion_coefficients"] = {};
    for (int i = 0; i < distCoeffs.cols; ++i) {
        j["distortion_coefficients"].push_back(distCoeffs.at<double>(0, i));
    }

    // 保存到文件
    std::ofstream ofs(filename);
    ofs << j.dump(4);  // 缩进为4，便于阅读
    ofs.close();

    std::cout << "标定结果已保存为 JSON 文件：" << filename << std::endl;
}

int main() {
    int boardWidth = 11;  // 横向内角点数量
    int boardHeight = 8;  // 纵向内角点数量
    float squareSize = 1.f;
    Size boardSize(boardWidth, boardHeight);

    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;
    vector<Point2f> corners;

    Mat frame, gray;
    namedWindow("image", WINDOW_NORMAL);

    VideoCapture cap("../calibration/calibration.mp4",cv::CAP_FFMPEG);  // 或使用 0 打开摄像头
    if (!cap.isOpened()) {
        cerr << "无法打开视频文件" << endl;
        return -1;
    }

    int frameCount = 0;
    Size imageSize;
    bool imageSizeSet = false;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "跳过空帧..." << endl;
            break;  // 自动跳过空帧
        }

        
        if (!imageSizeSet) {
            imageSize = frame.size();
            imageSizeSet = true;
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        bool found = findChessboardCorners(gray, boardSize, corners,
                                           CALIB_CB_ADAPTIVE_THRESH +
                                           CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            drawChessboardCorners(frame, boardSize, corners, found);
        }

        imshow("image", frame);
        char key = (char)waitKey(30);

        if (key == 27) break;  // ESC退出
        if (found) {
            vector<Point3f> objectCorners;
            for (int j = 0; j < boardHeight; j++) {
                for (int k = 0; k < boardWidth; k++) {
                    objectCorners.push_back(Point3f(k * squareSize, j * squareSize, 0));
                }
            }
            objectPoints.push_back(objectCorners);
            imagePoints.push_back(corners);
            frameCount++;
            cout << "自动保存帧 #" << frameCount << endl;
        }

    }

    if (!imageSizeSet||imagePoints.size() < 5) {
        cerr << "采集帧数量不足，无法进行标定" << endl;
        return -1;
    }

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                    distCoeffs, rvecs, tvecs);

    cout << "Camera matrix:" << endl << cameraMatrix << endl;
    cout << "Distortion coefficients:" << endl << distCoeffs << endl;
    saveCalibrationToJson(cameraMatrix, distCoeffs, "camera_calibration.json");

    return 0;
}
