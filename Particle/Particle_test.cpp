#include "Particle.hpp"

#include <memory>
#include <random>
#include <vector>

int main() {
  using ParPtr = std::shared_ptr<Particle>;

  std::vector<ParPtr> particles;
  constexpr int npar = 100;
  for (int i = 0; i < npar; i++) {
    particles.emplace_back(std::make_shared<Particle>());
  }

  std::mt19937 gen(0);
  std::uniform_int_distribution<long> udis(0, npar - 1);

  // fill random data
  for (auto &p : particles) {
    p->gid = udis(gen);
    p->globalIndex = udis(gen);
    p->rank = 0;
    p->group = udis(gen);
  }

  // pack
  msgpack::sbuffer sbuf;
  for (auto &p : particles) {
    msgpack::pack(sbuf, *p);
  }

  printf("packed buffer size: %u\n", sbuf.size());

  // unpack and verify
  {
    std::vector<ParPtr> particles_verify;
    std::size_t len = sbuf.size();
    std::size_t off = 0;
    while (off < len) {
      auto result = msgpack::unpack(sbuf.data(), len, off);
      particles_verify.emplace_back(
          std::make_shared<Particle>(result.get().as<Particle>()));
    }
    if (particles.size() != particles_verify.size()) {
      printf("Error size\n");
      exit(1);
    }
    for (int i = 0; i < npar; i++) {
      const auto &p = particles[i];
      const auto &pv = particles_verify[i];
      p->echo();
      pv->echo();
      if ((p->gid != pv->gid) || (p->globalIndex != pv->globalIndex) ||
          (p->group != pv->group) || (p->rank != pv->rank)) {
        printf("Error data\n");
      }
    }
  }

  return 0;
}