/**
 * @file Logger.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2021-04-28
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include "spdlog/cfg/env.h"
#include "spdlog/sinks/null_sink.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/spdlog.h"

#include <memory>

#include "mpi.h"

namespace logger {
void setup_mpi_spdlog(enum spdlog::level::level_enum level = spdlog::level::info) {

    /**
     * trace = SPDLOG_LEVEL_TRACE,
     * debug = SPDLOG_LEVEL_DEBUG,
     * info = SPDLOG_LEVEL_INFO,
     * warn = SPDLOG_LEVEL_WARN,
     * err = SPDLOG_LEVEL_ERROR,
     * critical = SPDLOG_LEVEL_CRITICAL,
     * off = SPDLOG_LEVEL_OFF,
     *
     */

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    spdlog::set_level(level);
    spdlog::logger sink = rank ? spdlog::logger("status", std::make_shared<spdlog::sinks::stdout_sink_st>())
                               : spdlog::logger("status", std::make_shared<spdlog::sinks::null_sink_st>());

    spdlog::set_default_logger(std::make_shared<spdlog::logger>(sink));
    spdlog::cfg::load_env_levels();
}
} // namespace logger

#endif