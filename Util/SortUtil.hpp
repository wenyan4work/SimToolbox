/**
 * @file SortUtil.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Utility for more sorting functions
 * @version 0.1
 * @date 2019-01-09
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef SORTUTIL_HPP_
#define SORTUTIL_HPP_

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

#include <omp.h>

/**
 * @brief compare index with tag
 *
 * @tparam Tag
 */
template <class Tag>
class sort_indices {
  private:
    const std::vector<Tag> *tagPtr;

  public:
    sort_indices(std::vector<Tag> *ptr) : tagPtr(ptr) {}

    sort_indices(const sort_indices &other) = default;

    sort_indices operator=(sort_indices<Tag> other) = delete;

    bool operator()(const int &i, const int &j) const {
        assert(i >= 0);
        assert(i < tagPtr->size());
        assert(j >= 0);
        assert(j < tagPtr->size());
        return (*tagPtr)[i] < (*tagPtr)[j];
    }
};

/**
 * @brief Sort both tag and data according to the comparison of tags
 *
 * @tparam Tag
 * @tparam TagContainer
 * @tparam Data
 * @tparam DataContainer
 * @param tags
 * @param data
 */
template <class Tag, class Data>
void sortDataWithTag(std::vector<Tag> &tags, std::vector<Data> &data) {
    const int length = tags.size();
    assert(tags.size() == data.size());
    std::vector<int> indices(length);
#pragma omp parallel for
    for (int i = 0; i < length; i++) {
        indices[i] = i;
    }

    // sort indices
    auto comp = sort_indices<Tag>(&tags);
    // This triggers bug? in intel openmp runtime?
    // std::sort(indices.begin(), indices.end(), comp);
    std::stable_sort(indices.begin(), indices.end(), comp);

    // create new
    std::vector<Tag> tagsSorted(length);
    std::vector<Data> dataSorted(length);
#pragma omp parallel for
    for (int i = 0; i < length; i++) {
        tagsSorted[i] = tags[indices[i]];
    }

#pragma omp parallel for
    for (int i = 0; i < length; i++) {
        dataSorted[i] = data[indices[i]];
    }

    std::swap(tagsSorted, tags);
    std::swap(dataSorted, data);
};

#endif