#ifndef MULTITYPE_HPP_
#define MULTITYPE_HPP_

#include <array>
#include <tuple>
#include <utility>
#include <vector>

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type //
for_each(std::tuple<Tp...> &, FuncT) // Unused arguments are given no names.
{}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if < I<sizeof...(Tp), void>::type //
                                     for_each(std::tuple<Tp...> &t, FuncT f) {
  f(std::get<I>(t));
  for_each<I + 1, FuncT, Tp...>(t, f);
}

/**
 * @brief Container for multiple types of particles
 *
 * @tparam ParType
 */
template <class... ParType>
struct MultiTypeContainer {
  std::tuple<std::vector<ParType>...> particles;

  std::vector<int> buildOffset() {
    std::vector<int> offset(1, 0);
    // iterate over particle types

    auto getsize = [&](const auto &container) {
      offset.push_back(container.size() + offset.back());
    };

    for_each(particles, getsize);
    return offset;
  }
};

#endif