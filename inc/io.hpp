/**
 * @file io.hpp
 * @brief Utilities for loading and saving raw grayscale image data.
 */

#pragma once

#include <string>
#include <cstdint>

/**
 * @brief Loads a raw binary file into an aligned memory buffer.
 * * This function reads a raw file from the `./assets/` directory. The resulting 
 * buffer is 64-byte aligned to facilitate vector operations.
 * * @param fileName    The name of the file (searched within ./assets/).
 * @param imageWidth  The width of the image in pixels.
 * @param imageHeight The height of the image in pixels.
 * @return uint8_t* Pointer to the allocated image buffer, or nullptr on failure.
 * @note The caller is responsible for freeing the returned memory using std::free().
 */
uint8_t* io_loadRaw(const std::string& fileName, uint32_t imageWidth, uint32_t imageHeight);

/**
 * @brief Saves a buffer of image data to a raw binary file.
 * * Writes the raw bytes to a file within the `./assets/` directory.
 * * @param fileName    The name of the file to create/overwrite.
 * @param imageBuffer Pointer to the pixel data to be saved.
 * @param imageWidth  The width of the image in pixels.
 * @param imageHeight The height of the image in pixels.
 */
void io_saveRaw(const std::string& fileName, uint8_t* imageBuffer, uint32_t imageWidth, uint32_t imageHeight);