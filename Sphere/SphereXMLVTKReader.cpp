#include "SphereXMLVTKReader.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLPPolyDataReader.h>         // XML PVTP files
#include <vtkXMLPUnstructuredDataReader.h> // XML PVTU files
#include <vtkXMLPolyDataWriter.h>

void SphereXMLVTKReader::setFileName(const std::string &pointDataName_, const std::vector<std::string> &gridDataName_) {
    pointDataName = pointDataName_;
    gridDataName = gridDataName_;
}

void SphereXMLVTKReader::readFrameToDest(const int &frameNumber, std::vector<Sphere> &sphereRead) {
    auto pvtpFileName = pointDataName + std::string("_") + std::to_string(frameNumber) + std::string(".pvtp");
    readPVTP(pvtpFileName, sphereRead);
}

void SphereXMLVTKReader::readPVTP(const std::string &filename, std::vector<Sphere> &sphereRead) {
    // VTK reference: https://lorensen.github.io/VTKExamples/site/Cxx/IO/FindAllArrayNames/
    // Read all the data from the file
    vtkSmartPointer<vtkXMLPPolyDataReader> reader = vtkSmartPointer<vtkXMLPPolyDataReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata = reader->GetOutput();
    int numberOfPointArrays = polydata->GetPointData()->GetNumberOfArrays();
    int numberOfPoints = polydata->GetPoints()->GetNumberOfPoints();
    std::cout << "read in " << numberOfPoints << " points from " << filename << ", containing " << numberOfPointArrays
              << " data arrays" << std::endl;
}

void SphereXMLVTKReader::readPVTU(const std::string &filename, std::vector<Sphere> &sphereRead) {}