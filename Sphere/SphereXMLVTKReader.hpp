/**
 * @file SphereXMLVTKReader.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief Read PVTP & PVTU files and reconstruct a std::vector<Sphere>
 * @version 0.1
 * @date 2019-02-20
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef SPHEREXMLVTKREADER_HPP_
#define SPHEREXMLVTKREADER_HPP_

#include "Shexp.hpp"
#include "Sphere.hpp"

#include <string>
#include <vector>

class SphereXMLVTKReader {

  public:
    SphereXMLVTKReader() = default;
    ~SphereXMLVTKReader() = default;

    void setFileName(const std::string &pointDataName_, const std::vector<std::string> &gridDataName_);
    void readFrameToDest(const int &frameNumber, std::vector<Sphere> &sphereRead);

  private:
    std::string pointDataName;
    std::vector<std::string> gridDataName;

    void readPVTP(const std::string &filename, std::vector<Sphere> &sphereRead);
    void readPVTU(const std::string &filename, std::vector<Sphere> &sphereRead);
};

#endif