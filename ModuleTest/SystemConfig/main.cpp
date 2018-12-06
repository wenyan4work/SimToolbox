#include "../../Util/SystemConfig.hpp"

int main() {
    SystemConfig config("runConfig.yaml");
    config.dump();
    return 0;
}