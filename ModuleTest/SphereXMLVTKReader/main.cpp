#include "../../Sphere/SphereXMLVTKReader.hpp"

int main(){

    SphereXMLVTKReader reader;
    std::vector<std::string> gridNames;
    reader.setFileName("Sphere",gridNames);

    std::vector<Sphere> sphereContainer;

    reader.readFrameToDest(0,sphereContainer);

    return 0;
}