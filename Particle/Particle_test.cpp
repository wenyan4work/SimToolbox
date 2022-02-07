#include "MultiTypeContainer.hpp"
#include "Particle.hpp"
#include "Sphere.hpp"
#include "Spherocylinder.hpp"

#include <iostream>
#include <memory>
#include <random>
#include <vector>

template <class Par>
void verify(const std::vector<Par> &A, const std::vector<Par> &B) {
  if (A.size() != B.size()) {
    printf("Error size\n");
    exit(1);
  }
  const int npar = A.size();
  for (int i = 0; i < npar; i++) {
    const auto &p = A[i];
    const auto &pv = B[i];
    if ((p.gid != pv.gid) || (p.globalIndex != pv.globalIndex) ||
        (p.group != pv.group) || (p.rank != pv.rank)) {
      p.echo();
      pv.echo();
      std::cout << pv.getMobMat() << std::endl;
      printf("Error data\n");
    }
  }
}

/**
 * @brief pack particle one by one
 *
 * @tparam Par
 */
template <class Par>
void testPackUnpack() {
  msgpack::sbuffer sbuf;
  std::vector<Par> particles;
  std::vector<Par> particles_verify;
  constexpr int npar = 100;
  {
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
    for (auto &p : particles) {
      msgpack::pack(sbuf, p);
    }
  }

  printf("packed buffer size: %u\n", sbuf.size());

  // unpack and verify
  {
    std::size_t len = sbuf.size();
    std::size_t off = 0;
    while (off < len) {
      auto result = msgpack::unpack(sbuf.data(), len, off);
      particles_verify.emplace_back(result.get().as<Par>());
    }
  }
  verify(particles, particles_verify);
}

/**
 * @brief pack std::vector<Par> as a single entity
 *
 * @tparam Par
 * @param fname
 */
template <class Par>
void testWriteReadData(const std::string &fname) {
  std::vector<Par> particles;
  std::vector<Par> particles_verify;
  constexpr int npar = 100;
  { // write
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

    std::ofstream ofs(fname);
    msgpack::pack(ofs, particles);
    ofs.close();
  }

  { // read
    std::ifstream ifs(fname, std::ifstream::binary);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    const std::string &sbuf = buffer.str();
    std::size_t len = sbuf.size();
    std::size_t off = 0;
    std::cout << len << " " << off << std::endl;
    auto result = msgpack::unpack(sbuf.data(), len, off);
    particles_verify = result.get().as<std::vector<Par>>();
    ifs.close();
  }

  verify(particles, particles_verify);
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

  testWriteReadData<Sylinder>("sylinder.msgpack");
  testWriteReadData<Sphere>("sphere.msgpack");

  testMultiContainer();

  return 0;
}