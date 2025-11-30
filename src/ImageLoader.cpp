#include "ImageLoader.hpp"

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

namespace {

constexpr char kCompressedMagic[] = "HFM";
constexpr std::size_t kCompressedMagicSize = sizeof(kCompressedMagic) - 1;

// Write and read integers in binary
void writeUint32(std::ostream& os, std::uint32_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeUint16(std::ostream& os, std::uint16_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeUint8(std::ostream& os, std::uint8_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::uint32_t readUint32(std::istream& is) {
    std::uint32_t value = 0;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!is) {
        throw std::runtime_error("无法读取压缩文件中的 32 位整数");
    }
    return value;
}

std::uint16_t readUint16(std::istream& is) {
    std::uint16_t value = 0;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!is) {
        throw std::runtime_error("无法读取压缩文件中的 16 位整数");
    }
    return value;
}

std::uint8_t readUint8(std::istream& is) {
    std::uint8_t value = 0;
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!is) {
        throw std::runtime_error("无法读取压缩文件中的 8 位整数");
    }
    return value;
}

struct HuffmanTable {
    std::array<std::uint8_t, 256> lengths{};
    std::array<std::uint32_t, 256> codes{};
    std::uint8_t maxLength = 0; // maximum code length
};

class BitWriter {
public:
    void writeBits(std::uint32_t code, std::uint8_t length) {
        // write bits to the buffer
        for (int bit = length - 1; bit >= 0; --bit) {
            const std::uint8_t value = static_cast<std::uint8_t>((code >> bit) & 0x1U);
            current_ = static_cast<std::uint8_t>((current_ << 1) | value);
            ++bitCount_;
            if (bitCount_ == 8) {
                data_.push_back(current_);
                bitCount_ = 0;
                current_ = 0;
            }
        }
    }

    std::vector<std::uint8_t> takeData() {
        // finalize and return the data buffer
        if (bitCount_ > 0) {
            current_ <<= static_cast<std::uint8_t>(8 - bitCount_);
            data_.push_back(current_);
            bitCount_ = 0;
            current_ = 0;
        }
        return std::move(data_);
    }

private:
    std::vector<std::uint8_t> data_;
    std::uint8_t current_ = 0;
    std::uint8_t bitCount_ = 0;
};

class BitReader {
public:
    BitReader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

    std::uint8_t readBit() {
        if (bitCount_ == 0) {
            if (index_ >= size_) {
                throw std::runtime_error("压缩数据在解码过程中意外结束");
            }
            current_ = data_[index_++];
            bitCount_ = 8;
        }
        const std::uint8_t bit = static_cast<std::uint8_t>(current_ >> 7);
        current_ <<= 1;
        --bitCount_;
        return bit;
    }

private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t index_ = 0;
    std::uint8_t current_ = 0;
    std::uint8_t bitCount_ = 0;
};

std::array<std::uint64_t, 256> buildHistogram(const std::vector<std::uint8_t>& data) {
    std::array<std::uint64_t, 256> histogram{};
    for (std::uint8_t value : data) {
        ++histogram[value];
    }
    return histogram;
}

std::array<std::uint8_t, 256> buildCodeLengths(const std::array<std::uint64_t, 256>& frequencies) {
    struct Node {
        std::uint64_t freq = 0;
        int symbol = -1;
        Node* left = nullptr;
        Node* right = nullptr;
    };

    struct NodeCompare {
        // Compare nodes by frequency, then by symbol
        bool operator()(const Node* lhs, const Node* rhs) const {
            if (lhs->freq == rhs->freq) {
                return lhs->symbol > rhs->symbol;
            }
            return lhs->freq > rhs->freq;
        }
    };

    std::priority_queue<Node*, std::vector<Node*>, NodeCompare> queue;
    std::vector<std::unique_ptr<Node>> storage; // memory management

    // Build initial nodes
    for (int symbol = 0; symbol < 256; ++symbol) {
        if (frequencies[symbol] == 0) {
            continue;
        }
        auto node = std::make_unique<Node>();
        node->freq = frequencies[symbol];
        node->symbol = symbol;
        queue.push(node.get());
        storage.push_back(std::move(node));
    }

    if (queue.empty()) {
        auto node = std::make_unique<Node>();
        node->freq = 1;
        node->symbol = 0;
        queue.push(node.get());
        storage.push_back(std::move(node));
    }

    while (queue.size() > 1) {
        Node* a = queue.top();
        queue.pop();
        Node* b = queue.top();
        queue.pop();

        auto parent = std::make_unique<Node>();
        parent->freq = a->freq + b->freq;
        parent->symbol = -1;
        parent->left = a;
        parent->right = b;
        queue.push(parent.get());
        storage.push_back(std::move(parent));
    }

    Node* root = queue.top();
    std::array<std::uint8_t, 256> lengths{};

    auto assignLengths = [&](auto&& self, Node* node, std::uint8_t depth) -> void {
        // assign code lengths to the Huffman tree recursively
        if (!node) {
            return;
        }
        if (node->symbol >= 0) {
            lengths[node->symbol] = depth == 0 ? 1 : depth;
            return;
        }
        self(self, node->left, static_cast<std::uint8_t>(depth + 1));
        self(self, node->right, static_cast<std::uint8_t>(depth + 1));
    };

    assignLengths(assignLengths, root, 0);
    return lengths;
}

HuffmanTable buildCanonicalTable(const std::array<std::uint8_t, 256>& lengths) {
    HuffmanTable table;
    table.lengths = lengths;

    std::array<std::uint32_t, 32> count{};
    std::uint8_t maxLength = 0;
    for (int symbol = 0; symbol < 256; ++symbol) {
        const std::uint8_t length = lengths[symbol];
        if (length == 0) {
            continue;
        }
        ++count[length];
        maxLength = std::max(maxLength, length);
    }
    table.maxLength = maxLength;

    std::array<std::uint32_t, 32> nextCode{};
    std::uint32_t code = 0;
    for (std::uint8_t length = 1; length <= maxLength; ++length) {
        code = (code + count[length - 1]) << 1;
        nextCode[length] = code;
    }

    std::vector<int> symbols;
    symbols.reserve(256);
    for (int symbol = 0; symbol < 256; ++symbol) {
        if (lengths[symbol] > 0) {
            symbols.push_back(symbol);
        }
    }

    std::sort(symbols.begin(), symbols.end(), [&](int lhs, int rhs) {
        if (lengths[lhs] == lengths[rhs]) {
            return lhs < rhs;
        }
        return lengths[lhs] < lengths[rhs];
    });

    for (int symbol : symbols) {
        const std::uint8_t length = lengths[symbol];
        table.codes[symbol] = nextCode[length]++;
    }

    return table;
}

class HuffmanDecoder {
public:
    explicit HuffmanDecoder(const HuffmanTable& table) {
        // build Trie from the Huffman table
        nodes_.push_back(Node{});
        for (int symbol = 0; symbol < 256; ++symbol) {
            const std::uint8_t length = table.lengths[symbol];
            if (length == 0) {
                continue;
            }
            std::uint32_t code = table.codes[symbol];
            int current = 0;
            for (int bit = length - 1; bit >= 0; --bit) {
                const int direction = static_cast<int>((code >> bit) & 0x1U);
                if (nodes_[current].child[direction] == -1) {
                    nodes_[current].child[direction] = static_cast<int>(nodes_.size());
                    nodes_.push_back(Node{});
                }
                current = nodes_[current].child[direction];
            }
            nodes_[current].symbol = symbol;
        }
    }

    std::uint8_t decodeSymbol(BitReader& reader) const {
        int current = 0;
        while (nodes_[current].symbol < 0) {
            const int bit = reader.readBit();
            current = nodes_[current].child[bit];
            if (current == -1) {
                throw std::runtime_error("哈夫曼解码过程中遇到非法路径");
            }
        }
        return static_cast<std::uint8_t>(nodes_[current].symbol);
    }

private:
    struct Node {
        int child[2] = {-1, -1};
        int symbol = -1;
    };

    std::vector<Node> nodes_;
};

std::vector<std::uint8_t> encode(const std::vector<std::uint8_t>& data, const HuffmanTable& table) {
    // encode an array of data using the provided Huffman table
    BitWriter writer;
    for (std::uint8_t value : data) {
        const std::uint8_t length = table.lengths[value];
        const std::uint32_t code = table.codes[value];
        writer.writeBits(code, length);
    }
    return writer.takeData();
}

std::vector<std::uint8_t> decode(const std::vector<std::uint8_t>& data, const HuffmanTable& table, std::size_t expectedCount) {
    // decode binary data to an array of bytes using the provided Huffman table
    BitReader reader(data.data(), data.size());
    HuffmanDecoder decoder(table);
    std::vector<std::uint8_t> result(expectedCount);
    for (std::size_t i = 0; i < expectedCount; ++i) {
        result[i] = decoder.decodeSymbol(reader);
    }
    return result;
}

std::vector<std::uint8_t> buildResidualChannel(const cv::Mat& image, int channel) {
    // build residuals for a single channel
    const int width = image.cols;
    const int height = image.rows;
    std::vector<std::uint8_t> residuals(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    for (int row = 0; row < height; ++row) {
        if (image.channels() == 1) {
            const auto* rowPtr = image.ptr<std::uint8_t>(row);
            for (int col = 0; col < width; ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * width + col;
                const std::uint8_t current = rowPtr[col];
                if (col == 0) {
                    residuals[index] = current;
                } else {
                    const std::uint8_t left = rowPtr[col - 1];
                    const int diff = static_cast<int>(current) - static_cast<int>(left);
                    residuals[index] = static_cast<std::uint8_t>(diff & 0xFF);
                }
            }
        } else {
            const auto* rowPtr = image.ptr<cv::Vec3b>(row);
            for (int col = 0; col < width; ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * width + col;
                const std::uint8_t current = rowPtr[col][channel];
                if (col == 0) {
                    residuals[index] = current;
                } else {
                    const std::uint8_t left = rowPtr[col - 1][channel];
                    const int diff = static_cast<int>(current) - static_cast<int>(left);
                    residuals[index] = static_cast<std::uint8_t>(diff & 0xFF);
                }
            }
        }
    }

    return residuals;
}

std::vector<std::uint8_t> reconstruct(const std::vector<std::uint8_t>& residuals, int width, int height) {
    // reconstruct original channel values from residuals
    std::vector<std::uint8_t> values(residuals.size());
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * width + col;
            const std::uint8_t encoded = residuals[index];
            if (col == 0) {
                values[index] = encoded;
            } else {
                const std::uint8_t previous = values[index - 1];
                const std::uint16_t sum = static_cast<std::uint16_t>(previous) + static_cast<std::uint16_t>(encoded);
                values[index] = static_cast<std::uint8_t>(sum & 0xFFU);
            }
        }
    }
    return values;
}

} // namespace

/*
 * Compression format:
 * [magic "HFM" (3 bytes)]
 * [width (4 bytes)]
 * [height (4 bytes)]
 * [maxValue (2 bytes)]
 * [channels (1 byte)]
 * [Huffman tables (256 bytes each channel)]
 * [dataBitCount (4 bytes)] [encoded data (variable)]
 */

void ImageLoader::compress(const std::string& path, const cv::Mat& image, int maxValue) {
    if (image.empty()) {
        throw std::runtime_error("无法压缩空图像");
    }
    if (image.depth() != CV_8U) {
        throw std::runtime_error("当前压缩仅支持 8 位");
    }
    const int channels = image.channels();
    if (channels != 1 && channels != 3) {
        throw std::runtime_error("当前压缩仅支持单通道或三通道图像");
    }

    const int width = image.cols;
    const int height = image.rows;

    std::vector<HuffmanTable> tables(static_cast<std::size_t>(channels));
    std::vector<std::vector<std::uint8_t>> encodedChannels(static_cast<std::size_t>(channels));

    for (int ch = 0; ch < channels; ++ch) {
        // Build residuals, histogram, code lengths, and Huffman table for each channel
        const auto residuals = buildResidualChannel(image, ch);
        const auto histogram = buildHistogram(residuals);
        const auto lengths = buildCodeLengths(histogram);
        tables[static_cast<std::size_t>(ch)] = buildCanonicalTable(lengths);
        encodedChannels[static_cast<std::size_t>(ch)] = encode(residuals, tables[static_cast<std::size_t>(ch)]);
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("无法写入压缩文件: " + path);
    }

    ofs.write(kCompressedMagic, static_cast<std::streamsize>(kCompressedMagicSize));
    writeUint32(ofs, static_cast<std::uint32_t>(width));
    writeUint32(ofs, static_cast<std::uint32_t>(height));
    writeUint16(ofs, static_cast<std::uint16_t>(maxValue));
    writeUint8(ofs, static_cast<std::uint8_t>(channels));

    for (int ch = 0; ch < channels; ++ch) {
        const auto& table = tables[static_cast<std::size_t>(ch)];
        ofs.write(reinterpret_cast<const char*>(table.lengths.data()), static_cast<std::streamsize>(table.lengths.size()));
        const auto& data = encodedChannels[static_cast<std::size_t>(ch)];
        writeUint32(ofs, static_cast<std::uint32_t>(data.size()));
        if (!data.empty()) {
            ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        }
        if (!ofs) {
            throw std::runtime_error("写入压缩数据失败");
        }
    }

    std::streampos fileSize = ofs.tellp();
    ofs.close();

    double originalSize = static_cast<double>(width) * height * channels;
    double compressedSize = static_cast<double>(fileSize);
    double ratio = (compressedSize / originalSize) * 100.0;

    std::cout << "=== 压缩性能分析 ===" << std::endl;
    std::cout << "原始图像尺寸: " << width << "x" << height << " (" << channels << " 通道)" << std::endl;

    std::cout.precision(2);
    std::cout << std::fixed;

    std::cout << "原始数据大小: ";
    if (originalSize >= 1048576.0) std::cout << (originalSize / 1048576.0) << " MB";
    else if (originalSize >= 1024.0) std::cout << (originalSize / 1024.0) << " KB";
    else std::cout << originalSize << " Bytes";
    std::cout << std::endl;

    std::cout << "压缩文件大小: ";
    if (compressedSize >= 1048576.0) std::cout << (compressedSize / 1048576.0) << " MB";
    else if (compressedSize >= 1024.0) std::cout << (compressedSize / 1024.0) << " KB";
    else std::cout << compressedSize << " Bytes";
    std::cout << std::endl;

    std::cout << "压缩率 (Compressed/Original): " << ratio << "%" << std::endl;
    std::cout << "====================" << std::endl;
}

ImageData ImageLoader::decompress(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("无法打开压缩文件: " + path);
    }

    char magicBuffer[kCompressedMagicSize];
    ifs.read(magicBuffer, static_cast<std::streamsize>(kCompressedMagicSize));
    if (!ifs || std::memcmp(magicBuffer, kCompressedMagic, kCompressedMagicSize) != 0) {
        throw std::runtime_error("压缩文件魔术字不匹配或文件损坏");
    }

    const std::uint32_t width = readUint32(ifs);
    const std::uint32_t height = readUint32(ifs);
    const std::uint16_t maxValue = readUint16(ifs);
    const std::uint8_t channels = readUint8(ifs);

    if (width == 0 || height == 0) {
        throw std::runtime_error("压缩文件的图像尺寸非法");
    }
    if (channels != 1 && channels != 3) {
        throw std::runtime_error("压缩文件包含不受支持的通道数");
    }

    const std::size_t pixelCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    std::vector<HuffmanTable> tables(static_cast<std::size_t>(channels));
    std::vector<std::vector<std::uint8_t>> channelValues(static_cast<std::size_t>(channels));

    for (int ch = 0; ch < channels; ++ch) {
        std::array<std::uint8_t, 256> lengths{};
        ifs.read(reinterpret_cast<char*>(lengths.data()), static_cast<std::streamsize>(lengths.size()));
        if (!ifs) {
            throw std::runtime_error("读取哈夫曼码长度失败");
        }
        tables[static_cast<std::size_t>(ch)] = buildCanonicalTable(lengths);

        const std::uint32_t byteCount = readUint32(ifs);
        std::vector<std::uint8_t> buffer(byteCount);
        if (byteCount > 0) {
            ifs.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(byteCount));
            if (!ifs) {
                throw std::runtime_error("读取压缩数据正文失败");
            }
        }
        auto residual = decode(buffer, tables[static_cast<std::size_t>(ch)], pixelCount);
        auto reconstructed = reconstruct(residual, static_cast<int>(width), static_cast<int>(height));
        channelValues[static_cast<std::size_t>(ch)] = std::move(reconstructed);
    }

    cv::Mat image(static_cast<int>(height), static_cast<int>(width), channels == 3 ? CV_8UC3 : CV_8UC1);
    for (std::uint32_t row = 0; row < height; ++row) {
        if (channels == 1) {
            auto* rowPtr = image.ptr<std::uint8_t>(static_cast<int>(row));
            for (std::uint32_t col = 0; col < width; ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + col;
                rowPtr[col] = channelValues[0][index];
            }
        } else {
            auto* rowPtr = image.ptr<cv::Vec3b>(static_cast<int>(row));
            for (std::uint32_t col = 0; col < width; ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + col;
                rowPtr[col][0] = channelValues[0][index];
                rowPtr[col][1] = channelValues[1][index];
                rowPtr[col][2] = channelValues[2][index];
            }
        }
    }

    ImageData data;
    data.magic = (channels == 3) ? "P6" : "P2";
    data.width = static_cast<int>(width);
    data.height = static_cast<int>(height);
    data.maxValue = maxValue;
    data.image = std::move(image);

    return data;
}

ImageData ImageLoader::load(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error("无法读取图像文件: " + path);
    }

    if (image.depth() != CV_8U) {
        throw std::runtime_error("仅支持 8 位深度的图像");
    }

    ImageData data;
    data.width = image.cols;
    data.height = image.rows;
    data.maxValue = 255;
    data.image = image;
    
    if (image.channels() == 3) {
        data.magic = "P6";
    } else if (image.channels() == 1) {
        data.magic = "P5";
    } else {
        throw std::runtime_error("仅支持 1 通道(灰度) 或 3 通道(彩色) 图像");
    }

    return data;
}

void ImageLoader::save(const std::string& path, const cv::Mat& image, int maxValue) {
    if (!cv::imwrite(path, image)) {
        throw std::runtime_error("无法保存图像到: " + path);
    }
}