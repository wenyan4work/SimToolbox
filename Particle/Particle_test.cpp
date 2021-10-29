#include "Particle.hpp"

#include <memory>
#include <random>
#include <vector>

struct Sph : public Particle {
  double radius = 5;
  MSGPACK_DEFINE(radius, MSGPACK_BASE(Particle));

  virtual ~Sph() = default;

  virtual void echo() const {
    printf("gid %d, globalIndex %d, radius %g, pos %g, %g, %g\n", //
           gid, globalIndex,
           radius, //
           pos[0], pos[1], pos[2]);
    printf("vel %g, %g, %g; omega %g, %g, %g\n", //
           vel[0], vel[1], vel[2],               //
           vel[3], vel[4], vel[5]);
    printf("orient %g, %g, %g, %g\n", //
           orientation[0], orientation[1], orientation[2], orientation[3]);
  }
};

template <class T>
void testpack() {
  using ParPtr = std::shared_ptr<T>;

  std::vector<ParPtr> particles;
  constexpr int npar = 100;
  for (int i = 0; i < npar; i++) {
    particles.emplace_back(std::make_shared<T>());
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
      particles_verify.emplace_back(std::make_shared<T>(result.get().as<T>()));
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
        exit(1);
      }
    }
  }
}

int main() {
  testpack<Particle>();
  testpack<Sph>();
  return 0;
}