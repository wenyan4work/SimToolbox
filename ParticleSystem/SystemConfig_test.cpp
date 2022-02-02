#include "SystemConfig.hpp"

int main() {
  SystemConfig config("config_test.toml");
  config.echo();

  return 0;
}