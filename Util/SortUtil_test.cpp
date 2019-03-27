#include "SortUtil.hpp"

#include <cstdio>
#include <random>

bool test(const int length) {
    std::vector<int> tags(length);
    std::vector<int> data(length);
    // prepare initial data
    for (int i = 0; i < length; i++) {
        tags[i] = i; // unique
        data[i] = i * 10;
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(tags.begin(), tags.end(), g);

    std::vector<std::pair<int, int>> combined(length);
    for (int i = 0; i < length; i++) {
        combined[i].first = tags[i];
        combined[i].second = data[i];
    }

    // sort
    sortDataWithTag(tags, data);
    std::sort(combined.begin(), combined.end());

    // compare
    bool pass = true;
    for (int i = 0; i < length; i++) {
        if (data[i] != combined[i].second) {
            printf("sort mismatch at %d, data: %d, combined: %d\n", i, data[i], combined[i].second);
            pass = false;
        }
    }

    return pass;
}

int main() {

    bool pass = test(1000);

    if (pass) {
        printf("TestPassed\n");
    }

    return 0;
}
