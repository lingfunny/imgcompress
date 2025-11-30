#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

struct ImageData {
    std::string magic;
    int width = 0;
    int height = 0;
    int maxValue = 255;
    cv::Mat image;
};

class ImageLoader {
public:
    static ImageData load(const std::string& path);
    static void save(const std::string& path, const cv::Mat& image, int maxValue = 255);
    static void compress(const std::string& path, const cv::Mat& image, int maxValue = 255);
    static ImageData decompress(const std::string& path);
};
