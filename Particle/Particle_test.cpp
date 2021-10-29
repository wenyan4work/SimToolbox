#include "Particle.hpp"

#include <iostream>
#include <memory>
#include <random>
#include <vector>

struct Sph {
  double radius = 5;
  MSGPACK_DEFINE(radius);

  void echo() const { printf("radius %g\n", radius); }

  Emat6 getMobMat() const { return Emat6::Identity() / radius; };

  /**
   * @brief Get AABB for neighbor search
   *
   * @return std::pair<std::array<double,3>, std::array<double,3>>
   * boxLow,boxHigh
   */
  std::pair<std::array<double, 3>, std::array<double, 3>>
  getBox(const double pos[3], const double orientation[4]) const {
    using Point = std::array<double, 3>;
    return std::make_pair<Point, Point>(
        Point{pos[0] - radius, pos[1] - radius, pos[2] - radius}, //
        Point{pos[0] + radius, pos[1] + radius, pos[2] + radius});
  };
};

int main() {
  using Par = Particle<Sph>;

  std::vector<Par> particles;
  constexpr int npar = 100;
  for (int i = 0; i < npar; i++) {
    particles.emplace_back();
  }

  std::mt19937 gen(0);
  std::uniform_int_distribution<long> udis(0, npar - 1);

  // fill random data
  for (auto &p : particles) {
    p.gid = udis(gen);
    p.globalIndex = udis(gen);
    p.rank = 0;
    p.group = udis(gen);
    p.isImmovable = false;
  }

  // pack
  msgpack::sbuffer sbuf;
  for (auto &p : particles) {
    msgpack::pack(sbuf, p);
  }

  printf("packed buffer size: %u\n", sbuf.size());

  // unpack and verify
  {
    std::vector<Par> particles_verify;
    std::size_t len = sbuf.size();
    std::size_t off = 0;
    while (off < len) {
      auto result = msgpack::unpack(sbuf.data(), len, off);
      particles_verify.emplace_back(result.get().as<Par>());
    }
    if (particles.size() != particles_verify.size()) {
      printf("Error size\n");
      exit(1);
    }
    for (int i = 0; i < npar; i++) {
      const auto &p = particles[i];
      const auto &pv = particles_verify[i];
      p.echo();
      pv.echo();
      std::cout << pv.getMobMat() << std::endl;
      if ((p.gid != pv.gid) || (p.globalIndex != pv.globalIndex) ||
          (p.group != pv.group) || (p.rank != pv.rank)) {
        printf("Error data\n");
      }
    }
  }

  return 0;
}