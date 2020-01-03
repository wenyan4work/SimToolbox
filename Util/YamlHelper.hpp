/**
 * @file YamlHelper.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief  type deduction and error message
 * @version 0.1
 * @date 2020-01-03
 *
 * @copyright Copyright (c) 2020
 *
 */
#ifndef YAMLHELPER_HPP_
#define YAMLHELPER_HPP_

#include <iostream>
#include <yaml-cpp/yaml.h>

/**
 * @brief convert variable name to string
 *
 */
#define VARNAME(a) #a

/**
 * @brief Read one variable from
 *
 * @tparam T  type of variable
 * @param config yaml config file
 * @param name name to look in yaml
 * @param variable [out] value found in yaml
 * @param err extra error message
 * @param optional if this variable is optional
 */
template <typename T>
void readConfig(const YAML::Node &config, const std::string &name, T &variable, const std::string &err,
                const bool optional = false) {
    if (config[name]) {
        variable = config[name].as<T>();
    } else {
        std::cout << "Expecting " << name << " in input yaml file not found. " << err << std::endl;
        if (!optional) {
            std::exit(1);
        }
    }
}

/**
 * @brief fixed-size array version of readConfig
 *
 * @tparam T
 * @param config
 * @param name
 * @param variable
 * @param err
 * @param optional
 */
template <typename T>
void readConfig(const YAML::Node &config, const std::string &name, T variable[], int dim, const std::string &err,
                const bool optional = false) {
    if (config[name]) {
        YAML::Node seq = config[name];
        if (seq.size() != dim) {
            std::cout << "Expecting " << dim << " elements in " << name << " in input yaml file. " << err << std::endl;
        }
        for (int i = 0; i < dim; i++) {
            variable[i] = seq[i].as<T>();
        }
    } else {
        std::cout << "Expecting " << name << " in input yaml file not found. " << err << std::endl;
        if (!optional) {
            std::exit(1);
        }
    }
}

/**
 * @brief std::vector version of
 *
 * @tparam T
 * @param config
 * @param name
 * @param variable
 * @param err
 * @param optional
 */
template <typename T>
void readConfig(const YAML::Node &config, const std::string &name, std::vector<T> &variable, const std::string &err,
                const bool optional = false) {
    if (config[name]) {
        YAML::Node seq = config[name];
        const int dim = seq.size();
        variable.resize(dim);
        for (int i = 0; i < dim; i++) {
            variable[i] = seq[i].as<T>();
        }
    } else {
        std::cout << "Expecting " << name << " in input yaml file not found. " << err << std::endl;
        if (!optional) {
            std::exit(1);
        }
    }
}

#endif