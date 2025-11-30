#include "ImageLoader.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>
#include <opencv2/opencv.hpp>

void printHelp(const char* programName) {
    std::cout << "用法: " << programName << " [选项] <输入> <输出>\n"
              << "示例: " << programName << " -c data/lena.bmp out/lena.hfm\n"
              << "      " << programName << " -x out/lena.hfm out/lena_restored.bmp\n\n"
              << "  -h, --help       显示本帮助并退出\n"
              << "  -c, --compress   压缩图像\n"
              << "  -x, --extract    解压图像\n";
}

int main(int argc, char* argv[]) {
    bool doCompress = false;
    bool doExtract = false;

    struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"compress", no_argument, 0, 'c'},
        {"extract", no_argument, 0, 'x'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "hcx", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'h':
                printHelp(argv[0]);
                return 0;
            case 'c':
                doCompress = true;
                break;
            case 'x':
                doExtract = true;
                break;
            default:
                printHelp(argv[0]);
                return 1;
        }
    }

    if (optind + 2 > argc) {
        std::cerr << "错误: 缺少输入或输出文件参数\n";
        printHelp(argv[0]);
        return 1;
    }

    if (!doCompress && !doExtract) {
        std::cerr << "错误: 请指定操作模式 (-c 压缩 或 -x 解压)\n";
        printHelp(argv[0]);
        return 1;
    }

    if (doCompress && doExtract) {
        std::cerr << "错误: 不能同时指定压缩和解压\n";
        return 1;
    }

    std::string inputFile = argv[optind];
    std::string outputFile = argv[optind + 1];

    try {
        if (doCompress) {
            std::cout << "正在读取: " << inputFile << " ..." << std::endl;
            ImageData data = ImageLoader::load(inputFile);
            
            std::cout << "正在压缩至: " << outputFile << " ..." << std::endl;
            ImageLoader::compress(outputFile, data.image);
        } else {
            std::cout << "正在解压: " << inputFile << " ..." << std::endl;
            ImageData data = ImageLoader::decompress(inputFile);
            
            std::cout << "保存图像至: " << outputFile << " ..." << std::endl;
            ImageLoader::save(outputFile, data.image);
        }

        std::cout << "完成。" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
