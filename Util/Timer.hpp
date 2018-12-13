/**
 * @file Timer.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief a simple timer and logger
 * @version 1.0
 * @date 2017-11-17
 *
 *
 * Reference: https://gist.github.com/jtilly/a423be999929d70406489a4103e67453
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>

/**
 * @brief a simple timer and logger class
 *
 */
class Timer {
  private:
    std::chrono::high_resolution_clock::time_point startTime; ///< tick
    std::chrono::high_resolution_clock::time_point stopTime;  ///< tock
    std::stringstream logfile;                                ///< log to a string
    bool work = true;                                         ///< flag for enabling/disabling this timer

  public:
    explicit Timer() = default;

    explicit Timer(bool work_) : Timer() { work = work_; }

    ~Timer() = default;

    /**
     * @brief record tick time
     *
     */
    void start() {
        if (work)
            this->startTime = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief record tock time and add a message for the recorded time interval
     *
     * @param s
     */
    void stop(const std::string &s) {
        if (work) {
            this->stopTime = std::chrono::high_resolution_clock::now();
            logfile << s.c_str() << " Time elapsed = "
                    << std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime).count() / 1e6
                    << std::endl;
        }
    }

    /**
     * @brief get time interval in seconds
     *
     * @return double
     */
    double getTime() {
        return std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime).count() / 1e6;
    }

    /**
     * @brief display all recorded time interval with messages
     *
     */
    void dump() {
        if (work)
            std::cout << logfile.rdbuf();
    }
};

#endif
