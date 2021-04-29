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

/**
 * @brief utility class
 *
 */
class Logger {

    /**
     * @brief initialize the spdlog
     *
     * @param level default level set to warn, i.e., print warn, err and critical
     */
  public:
    static void setup_mpi_spdlog(const int level_rank0 = spdlog::level::info) {

        /**
         *  spdlog levels and the what each level means in this code
         *  spdlog::level is different on mpi ranks
         *  on rank 0 level is set by level_rank0
         *  on other ranks level is set to spdlog::level::err
         *  environtment variable SPDLOG_LEVEL is ignored
         *
         * trace = SPDLOG_LEVEL_TRACE,       -> other least important messages
         * debug = SPDLOG_LEVEL_DEBUG,       -> showing the current
         * info = SPDLOG_LEVEL_INFO,         -> optional information
         * warn = SPDLOG_LEVEL_WARN,         -> messages must always be shown when code runs
         * err = SPDLOG_LEVEL_ERROR,         -> error but code continues
         * critical = SPDLOG_LEVEL_CRITICAL, -> messages when code crashes
         * off = SPDLOG_LEVEL_OFF,
         *
         */

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // spdlog::logger sink = rank ? spdlog::logger("log", std::make_shared<spdlog::sinks::stdout_sink_st>())
        //                            : spdlog::logger("log", std::make_shared<spdlog::sinks::null_sink_st>());
        // sink.set_level(spdlog::level::err);
        // spdlog::cfg::load_env_levels();

        spdlog::logger sink =
            spdlog::logger("rank " + std::to_string(rank), std::make_shared<spdlog::sinks::stdout_sink_st>());

        spdlog::set_default_logger(std::make_shared<spdlog::logger>(sink));

        if (rank == 0)
            spdlog::set_level(static_cast<spdlog::level::level_enum>(level_rank0));
        else
            spdlog::set_level(spdlog::level::err);
    }

    static void set_level(const int level_rank0 = spdlog::level::info) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            spdlog::set_level(static_cast<spdlog::level::level_enum>(level_rank0));
        else
            spdlog::set_level(spdlog::level::err);
    }
};

#endif