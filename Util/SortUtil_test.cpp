#include "SortUtil.hpp"

#include <cstdio>

int main() {
    const int length = 100;
    std::vector<int> tags(length);
    std::vector<int> data(length);
    // prepare initial data
    for (int i = 0; i < length; i++) {
        tags[i] = i; // unique
        data[i] = i * 10;
    }
    std::random_shuffle(tags.begin(), tags.end());
    std::vector<std::pair<int, int>> combined(length);
    for (int i = 0; i < length; i++) {
        combined[i].first = tags[i];
        combined[i].second = data[i];
    }

    // sort
    sortDataWithTag(tags, data);
    std::sort(combined.begin(), combined.end());

    // compare
    for (int i = 0; i < length; i++) {
        if (data[i] != combined[i].second) {
            printf("sort mismatch at %d, data: %d, combined: %d\n", i, data[i], combined[i].second);
        } else {
            printf("sort result data: %d, combined: %d\n", data[i], combined[i].second);
        }
    }

    return 0;
}
