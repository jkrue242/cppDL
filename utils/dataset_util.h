//==============================================
// joseph krueger, 2025
//==============================================
#ifndef DATASET_UTIL_H
#define DATASET_UTIL_H
#include <fstream>
#include <iterator>
#include <string>

//==============================================
// Dataset util class
//==============================================
class DatasetUtil {
public:
    static bool download(const std::string& url, const std::string& filepath);
    static std::string file_to_buffer(const std::string& filepath);
};

#endif // DATASET_UTIL_H

