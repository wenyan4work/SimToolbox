/**
 * @file TRngPool.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Random Number Generator based on TRNG library
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef TRNGPOOL_HPP_
#define TRNGPOOL_HPP_

#include <iostream>
#include <memory>

#include <mpi.h>
#include <omp.h>

// #include <trng/config.hpp>
#include <trng/lcg64_shift.hpp>
#include <trng/lognormal_dist.hpp>
#include <trng/mrg5.hpp>
#include <trng/normal_dist.hpp>
#include <trng/uniform01_dist.hpp>

/**
 * @brief rng based on TRNG library
 *
 */
class TRngPool {
    using myEngineType = trng::mrg5; ///< rng engine

  private:
    int myRank;   ///< my MPI rank
    int nProcs;   ///< total number of MPI ranks
    int nThreads; ///< number of threads on each rank

    std::vector<std::unique_ptr<myEngineType>> rngEngineThreadsPtr; /// one rng for each local thread

    trng::uniform01_dist<double> u01; ///< \f$U(0,1)\f$ transformer
    trng::normal_dist<double> n01;    ///< \f$N(0,1)\f$ transformer

    std::unique_ptr<trng::lognormal_dist<double>> lnDistPtr; ///< \f$logNormal(\mu,\sigma)\f$ transformer

  public:
    /**
     * @brief Set the Log Normal Parameters \f$\mu,\sigma\f$
     * sigma2 = sigma*sigma
     * mean = exp(mu + sigma2 / 2))
     * var  = (exp(sigma2) - 1) * exp(2 * mu + sigma2)) 
     * @param mu
     * @param sigma
     */
    void setLogNormalParameters(double mu, double sigma) {
        lnDistPtr.reset(new trng::lognormal_dist<double>(mu, sigma));
    }

    /**
     * @brief Construct a new TRngPool object with seed
     *
     * @param seed
     */
    explicit TRngPool(int seed = 0) : n01(0, 1) {

        myRank = 0;
        nProcs = 1;
        nThreads = 1;
        int mpiInitFlag;
        MPI_Initialized(&mpiInitFlag);
        if (mpiInitFlag > 0) {
            MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
            MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
        } else {
            myRank = 0;
            nProcs = 1;
        }
        nThreads = omp_get_max_threads();
        myEngineType rngEngine;
        rngEngine.seed(static_cast<unsigned long>(seed));

        if (nProcs > 1) {
            rngEngine.split(nProcs, myRank);
        }

        rngEngineThreadsPtr.resize(nThreads);
#pragma omp parallel for num_threads(nThreads)
        for (int i = 0; i < nThreads; i++) {
            // a copy of engine for each thread
            rngEngineThreadsPtr[i].reset(new myEngineType());
            *rngEngineThreadsPtr[i] = rngEngine;
            // split
            rngEngineThreadsPtr[i]->split(nThreads, i);
        }
        setLogNormalParameters(1.0, 1.0);
    };

    ~TRngPool() = default;

    TRngPool(const TRngPool &) = delete;
    TRngPool &operator=(const TRngPool &) = delete;

    /**
     * @brief get a U01 random number on thread Id
     *
     * @param threadId
     * @return double
     */
    inline double getU01(int threadId) { return u01(*rngEngineThreadsPtr[threadId]); }
    inline double getN01(int threadId) { return n01(*rngEngineThreadsPtr[threadId]); }
    inline double getLN(int threadId) { return (*lnDistPtr)(*rngEngineThreadsPtr[threadId]); }

    /**
     * @brief check thread ID and get a U01 random number
     *
     * @return double
     */
    inline double getU01() {
        const int threadId = omp_get_thread_num();
        return getU01(threadId);
    }

    inline double getN01() {
        const int threadId = omp_get_thread_num();
        return getN01(threadId);
    }

    inline double getLN() {
        const int threadId = omp_get_thread_num();
        return getLN(threadId);
    }
};

#endif