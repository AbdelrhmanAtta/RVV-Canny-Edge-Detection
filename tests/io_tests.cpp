#include "io.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

struct rawImage_t
{
    std::string fileName;
    std::string destName;
    uint32_t imageWidth;
    uint32_t imageHeight;
};

int main()
{
    const uint32_t dim = 512;
    std::vector<rawImage_t> images = 
    {
        {"rect.raw", "rect_copy.raw", dim, dim},
        {"circ.raw", "circ_copy.raw", dim, dim},
        {"diag.raw", "diag_copy.raw", dim, dim},
    };

    for(const auto& [src, dest, w, h] : images)
    {
        uint8_t* imageBuffer = io_loadRaw(src, w, h);
        if (imageBuffer != nullptr) 
        {
            io_saveRaw(dest, imageBuffer, w, h);
            std::free(imageBuffer);
        }
        else 
        {
            std::cerr << "Skipping save for " << src << " due to load failure." << std::endl;
        }
    }

    return 0;
}
