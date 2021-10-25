#include "ConstraintBlock.hpp"

int main(int argc, char **argv) {

  ConstraintBlockPool cPool(5);
  for (auto &que : cPool) {
    que.resize(10);
  }

  writeConstraintBlockPool("data.msgpack", cPool, true);

  return 0;
}