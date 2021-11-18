/**
 * @file TomlHelper.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2021-11-17
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef TOMLHELPER_HPP_
#define TOMLHELPER_HPP_

#include "Logger.hpp"

#include <toml.hpp>

/**
 * @brief convert variable name to string
 *
 */
#define VARNAME(a) #a

template <class T>
void err_message(bool optional, T &name, T &err) {
  if (optional) {
    spdlog::warn("Optional " + name +
                 " in input toml file not found, using default");
  } else {
    spdlog::critical("Required parameter " + name +
                     " in input toml file not found");
    std::exit(1);
  }
}

/**
 * @brief read a single value
 *
 * @tparam T
 * @param config
 * @param name
 * @param result
 * @param optional
 * @param err
 */
template <typename T>
void readConfig(const toml::value &config, const std::string &name, T &result,
                const bool optional = false, const std::string &err = "") {
  if (config.contains(name)) {
    result = toml::find<T>(config, name);
  } else {
    err_message(optional, name, err);
  }
}

/**
 * @brief read dim values
 *
 * @tparam T
 * @param config
 * @param name
 * @param result
 * @param dim
 * @param optional
 * @param err
 */
template <typename T>
void readConfig(const toml::value &config, const std::string &name, T result[],
                const int dim, const bool optional = false,
                const std::string &err = "") {
  if (config.contains(name)) {
    const auto arr = toml::find(config, name);
    const auto value = toml::get<std::vector<T>>(arr);
    if (value.size() != dim) {
      spdlog::critical("Expecting {} elements in {} in input toml file {}.",
                       dim, name, err);
      std::exit(1);
    }
    for (int i = 0; i < dim; i++) {
      result[i] = value[i];
    }
  } else {
    err_message(optional, name, err);
  }
}

/**
 * @brief read an unknown number of values
 *
 * @tparam T
 * @param config
 * @param name
 * @param result
 * @param optional
 * @param err
 */
template <typename T>
void readConfig(const toml::value &config, const std::string &name,
                std::vector<T> &result, const bool optional = false,
                const std::string &err = "") {
  if (config.contains(name)) {
    const auto arr = toml::find(config, name);
    result = toml::get<std::vector<T>>(arr);
  } else {
    err_message(optional, name, err);
  }
}

#endif