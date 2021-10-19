/**
 * @file ZDD2.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief a wrapper for Zoltan2_Directory
 *
 * WARNING:
 * 1. Zoltan2_Directory requires user_t defines += operator
 * 2. Zoltan2_Directory does not have explicit template initiation
 * for custom user_t types.
 *
 * @version 0.1
 * @date 2021-10-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef ZDD2_HPP_
#define ZDD2_HPP_

#include "TpetraUtil.hpp"
#include "Util/Logger.hpp"

#include <vector>

#include <mpi.h>

/**
 * @brief A wrapper for Zoltan2_Directory
 *
 * @tparam T the data type associated with each gid
 */
template <class T>
class ZDD2 {
  using DataDirectory = Zoltan2::Zoltan2_Directory_Simple<TGO, TLO, T>;

  int nProcs;                           ///< mpi rank size
  int rank;                             ///< local mpi rank id
  std::unique_ptr<DataDirectory> ddPtr; ///< Zoltan2_Directory object

public:
  std::vector<TGO> gidToFind;  ///< a list of Global ID to find
  std::vector<T> dataToFind;   ///< data associated with ID to find
  std::vector<TGO> gidOnLocal; ///< ID located on local mpi rank
  std::vector<T> dataOnLocal;  ///< data associated with ID on each proc

  ZDD2(const ZDD2 &) = delete;
  ZDD2 &operator=(const ZDD2 &other) = delete;

  /**
   * @brief Construct a new ZDD object with an estimate of buffer list size
   *
   * @param nEst estimate of buffer list size
   */
  ZDD2(TGO nEst = 100) {
    const auto &commRcp = getMPIWORLDTCOMM();
    rank = commRcp->getRank();
    nProcs = commRcp->getSize();
    ddPtr = std::make_unique<DataDirectory>(commRcp, false, 0);

    gidToFind.reserve(nEst);
    dataToFind.reserve(nEst);
    gidOnLocal.reserve(nEst);
    dataOnLocal.reserve(nEst);

    static_assert(std::is_trivially_copyable<T>::value, "");
    static_assert(std::is_default_constructible<T>::value, "");
  }

  /**
   * @brief Destroy the ZDD2 object
   *
   */
  ~ZDD2() = default;

  /**
   * @brief build the Zoltan data directory
   *
   * must have an effective nblocal data list
   * @return int the error code returned by Zoltan_DD.Update()
   */
  int build() {
    const auto length = gidOnLocal.size();
    dataOnLocal.resize(length);

    int error =
        ddPtr->update(length, gidOnLocal.data(), nullptr, dataOnLocal.data(),
                      nullptr, DataDirectory::Update_Mode::Replace);
    spdlog::debug("ZDD2 build error code {}", error);

    return error;
  }

  /**
   * @brief locate and put data to findData as requested by findID
   *
   * @return int the error code returned by Zoltan_DD.Find()
   */
  int find() {
    const auto length = gidToFind.size();
    dataToFind.clear();
    dataToFind.resize(length);

    int error = ddPtr->find(length, gidToFind.data(), nullptr,
                            dataToFind.data(), nullptr, nullptr, false);
    spdlog::debug("ZDD2 find error code {}", error);
    return error;
  }
};

#endif /* ZDD_HPP_ */
