#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

uint8_t* IO_LoadRaw(const std::string& fileName, uint32_t imageWidth, uint32_t imageHeight);

void IO_SaveRaw(const std::string& fileName, uint8_t* imageBuffer,uint32_t imageWidth, uint32_t imageHeight);
