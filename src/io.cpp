#include "io.hpp"

uint8_t* IO_LoadRaw(const std::string& fileName, uint32_t imageWidth, uint32_t imageHeight)
{
    size_t imageSize = static_cast<size_t>(imageWidth)*imageHeight;
    size_t alignedImageSize = (imageSize+63) & ~63;     // (Optional) To make imageSize aligned with 64

    uint8_t* imageBuffer = static_cast<uint8_t*>(std::aligned_alloc(64, alignedImageSize));
    if(!imageBuffer)
    {
        std::cerr << "[IO FAILURE] Memory Allocation error while loading: " << fileName << std::endl;
        return nullptr;
    }

    std::string filePath = std::string("./assets/") + fileName;
    std::ifstream file(filePath, std::ios::binary);
    if(!file)
    {
        std::cerr << "[IO FAILURE] File access error while loading: " << fileName << std::endl;
        std::free(imageBuffer);
        return nullptr;
    }

    file.read(reinterpret_cast<char*>(imageBuffer), imageSize);
    file.close();

    return imageBuffer;
}


void IO_SaveRaw(const std::string& fileName, uint8_t* imageBuffer,uint32_t imageWidth, uint32_t imageHeight)
{
    if(!imageBuffer)
    {
        std::cerr << "[IO FAILURE] Save access error while saving: " << fileName << std::endl;
        return;
    }
    size_t imageSize = static_cast<size_t>(imageWidth)*imageHeight;

    std::string filePath = std::string("./assets/") + fileName;
    std::ofstream file(filePath, std::ios::binary);
    if(!file)
    {
        std::cerr << "[IO FAILURE] File access error while saving: " << fileName << std::endl;
        return;
    }

    file.write(reinterpret_cast<char*>(imageBuffer), imageSize);
    file.close();
}