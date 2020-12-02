#include "SpecialQuadWeights.hpp"
#include <iostream>

bool test(const int numQuadPt, const double lineHalfLength, const double *lineCenterCoord, const double *targetCoord, const double *lineDirection) {

    SpecialQuadWeights<16> sqw(numQuadPt);
    sqw.calcWeights(lineHalfLength, lineCenterCoord, targetCoord, lineDirection);
    const double *w1 = sqw.getWeights1();
    const double *w3 = sqw.getWeights3();
    const double *w5 = sqw.getWeights5();
    const double *wgl = sqw.getGLWeights();
    sqw.print();

    double er1;
    double er3;
    double er5;
    for (int i=0; i<numQuadPt; i++) {
        er1 = (wgl[i] - w1[i])/wgl[i]; 
        er3 = (wgl[i] - w3[i])/wgl[i]; 
        er5 = (wgl[i] - w5[i])/wgl[i]; 
        std::cout << "er1 " << er1 << "| er3 " << er3 << "| er5 " << er5 << std::endl; 
    }

    return true;
}

int main(int argc, char **argv) {
    
    const double lineHalfLength = 1.0;
    const double lineCenterCoord[3] = {0,0,0};
    const double targetCoord[3] = {1.1,0.0,0.0};
    const double lineDirection[3] = {1,0,0};
    
    bool status = test(16, lineHalfLength, lineCenterCoord, targetCoord, lineDirection);
    
    printf("Test finished\n");
    return 0;
}

// g++ -g  -I<path_to_eigen> -I<path_to_utility_folder> SpecialQuadWeights_test.cpp
