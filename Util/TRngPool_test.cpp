#include "TRngPool.hpp"

#include <random>
#include <vector>

#include <omp.h>

constexpr int nSample = 10000000;

std::pair<double, double> checkSample(const std::vector<double> &sample) {
    double mean = 0, var = 0;
    const int n = sample.size();
    for (int i = 0; i < n; i++)
        mean += sample[i];
    mean /= n;

    for (int i = 0; i < n; i++)
        var += (sample[i] - mean) * (sample[i] - mean);
    var /= n;

    return std::pair<double, double>(mean, var);
}

bool testU01() {
    std::random_device rd;
    TRngPool rngPool(rd());
    std::vector<double> sample(nSample);
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nSample; i++) {
            sample[i] = rngPool.getU01(tid);
        }
    }

    auto check = checkSample(sample);
    printf("sample mean  :%g\n", check.first);
    printf("sample var   :%g\n", check.second);
    if (fabs(check.first - 0.5) < 0.01 && fabs(check.second - 1.0 / 12) < 0.01) {
        return true;
    } else {
        return false;
    }
}

bool testN01() {
    std::random_device rd;
    TRngPool rngPool(rd());
    std::vector<double> sample(nSample);
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nSample; i++) {
            sample[i] = rngPool.getN01(tid);
        }
    }

    auto check = checkSample(sample);
    printf("sample mean  :%g\n", check.first);
    printf("sample var   :%g\n", check.second);
    if (fabs(check.first - 0.0) < 0.01 && fabs(check.second - 1.0) < 0.01) {
        return true;
    } else {
        return false;
    }
}

bool testLogNormal() {
    std::random_device rd;
    TRngPool rngPool(rd());
    const double mu = 1.2;
    const double sigma = 0.3;
    rngPool.setLogNormalParameters(mu, sigma);
    std::vector<double> sample(nSample);
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nSample; i++) {
            sample[i] = rngPool.getLN(tid);
        }
    }

    auto check = checkSample(sample);
    printf("sample mean  :%g\n", check.first);
    printf("sample var   :%g\n", check.second);
    const double sigma2 = sigma * sigma;
    if (fabs(check.first - exp(mu + sigma2 / 2)) < 0.05 &&
        fabs(check.second - (exp(sigma2) - 1) * exp(2 * mu + sigma2)) < 0.05) {
        return true;
    } else {
        return false;
    }
}

int main() {
    if (testU01() && testN01() && testLogNormal()) {
        printf("TestPassed\n");
    } else {
        printf("Error\n");
    }

    return 0;
}