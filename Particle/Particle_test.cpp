#include "MultiTypeContainer.hpp"
#include "Particle.hpp"
#include "Sphere.hpp"
#include "Spherocylinder.hpp"

#include <iostream>
#include <memory>
#include <random>
#include <vector>

template <class Par>
void testPackUnpack() {

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
    p.immovable = false;
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
}

void testMultiContainer() {

  using MyContainer = MultiTypeContainer<Sylinder, Sphere>;
  MyContainer myContainer;

  std::get<0>(myContainer.particles).resize(20);
  std::get<1>(myContainer.particles).resize(40);

  const auto &offset = myContainer.buildOffset();
  for (auto &v : offset) {
    printf("%d\n", v);
  }
  printf("Total number particles: %d\n", offset.back());

  return;
}

int main() {
  testPackUnpack<Sylinder>();
  testPackUnpack<Sphere>();

  testMultiContainer();

  return 0;
}