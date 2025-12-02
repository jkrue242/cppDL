//==============================================
// joseph krueger, 2025
//==============================================
#include "dataset_util.h"
#include <curl/curl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#include <boost/filesystem.hpp>

//==============================================
// creates a directory
//==============================================
static void create_directory(const std::string& filepath) {
    boost::filesystem::path file_path(filepath);
    boost::filesystem::path dir = file_path.parent_path();

    if (!dir.empty()) {
        try {
            boost::filesystem::create_directories(dir);
        } catch (const boost::filesystem::filesystem_error& e) {
            std::cerr << "[Downloader] Error creating directory: " << dir << " - " << e.what() << std::endl;
        }
    }
}

//==============================================
// callback for writing to file
//==============================================
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    size_t total_size = size * nmemb;
    file->write(static_cast<char*>(contents), total_size);
    return total_size;
}

//==============================================
// Downloads a file from the given URL and saves it to the specified path
//==============================================
bool DatasetUtil::download(const std::string& url, const std::string& filepath) {
    create_directory(filepath);

    // open file to write
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Downloader] Error: Could not open file for writing: " << filepath << std::endl;
        return false;
    }

    // curl setup
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[Downloader] Error: Failed to initialize curl" << std::endl;
        file.close();
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback); // uses the callback to write 
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    // download  the file
    CURLcode res = curl_easy_perform(curl);
    bool success = (res == CURLE_OK);
    if (!success) {
        std::cerr << "[Downloader] Error: curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
    }

    curl_easy_cleanup(curl);
    file.close();
    return success;
}


//==============================================
// reads a file to a string buffer 
//==============================================
std::string DatasetUtil::file_to_buffer(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::string error = "Error opening file: " + filepath;
        throw std::runtime_error(error);
    }
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    ); // i might need to add some sort of max size to this
}