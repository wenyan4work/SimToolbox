/**
 * @file SylinderSystem.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief System for sylinders
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef SYLINDERSYSTEM_HPP_
#define SYLINDERSYSTEM_HPP_

#include "Sylinder.hpp"
#include "SylinderConfig.hpp"
#include "SylinderNear.hpp"

#include "Boundary/Boundary.hpp"
// #include "Constraint/ConstraintSolver.hpp"
#include "FDPS/particle_simulator.hpp"
#include "Trilinos/TpetraUtil.hpp"
#include "Trilinos/ZDD.hpp"
#include "Util/TRngPool.hpp"

#include <unordered_map>

/**
 * @brief A collection of sylinders distributed to multiple MPI ranks.
 *
 */
class SylinderSystem {
    // FDPS stuff
    PS::DomainInfo dinfo; ///< domain size, boundary condition, and decomposition info
    void setDomainInfo();

    PS::ParticleSystem<Sylinder> sylinderContainer;        ///< sylinders
    std::unique_ptr<TreeSylinderNear> treeSylinderNearPtr; ///< short range interaction of sylinders
    int treeSylinderNumber;                                ///< the current max_glb number of treeSylinderNear
    void setTreeSylinder();

    std::unordered_multimap<int, int> linkMap;        ///< links prev,next
    std::unordered_multimap<int, int> linkReverseMap; ///< links next, prev

    // constraint stuff
    std::shared_ptr<ConstraintCollector> conCollectorPtr; ///<  pointer to ConstraintCollector

    // MPI stuff
    std::shared_ptr<TRngPool> rngPoolPtr;      ///< TRngPool object for thread-safe random number generation
    const Teuchos::RCP<const TCOMM> commRcp;   ///< TCOMM, set as a Teuchos::MpiComm object in constructor
    Teuchos::RCP<TMAP> sylinderMapRcp;         ///< TMAP, contiguous and sequentially ordered 1 dof per sylinder
    Teuchos::RCP<TMAP> sylinderMobilityMapRcp; ///< TMAP, contiguous and sequentially ordered 6 dofs per sylinder
    Teuchos::RCP<TCMAT> mobilityMatrixRcp;     ///< block-diagonal mobility matrix
    Teuchos::RCP<TOP> mobilityOperatorRcp;     ///< full mobility operator (matrix-free), to be implemented

    // Data directory
    std::shared_ptr<ZDD<SylinderNearEP>> sylinderNearDataDirectoryPtr; ///< distributed data directory for sylinder data

    // internal utility functions
    /**
     * @brief generate initial configuration on rank 0 according to runConfig
     *
     */
    void setInitialFromConfig();

    /**
     * @brief set initial configuration as given in the (.dat) file
     *
     * The simBox and BC settings in runConfig are still used
     * @param filename
     */
    void setInitialFromFile(const std::string &filename);

    /**
     * @brief set linkMap from the .dat file
     *
     * Every mpi rank run this simultaneously to set linkMap from the same file
     * @param filename
     */
    void setLinkMapFromFile(const std::string &filename);

    /**
     * @brief set initial configuration as given in the (.dat) file
     *
     * The simBox and BC settings in runConfig are still used
     * @param pvtpFileName
     */
    void setInitialFromVTKFile(const std::string &pvtpFileName);

    /**
     * @brief set initial configuration if runConfig.initCircularX is set
     *
     * This function move the position of all sylinders into a cylindrical tube fit in initBox
     */
    void setInitialCircularCrossSection();

    /**
     * @brief update the sylinderMap and sylinderMobilityMap
     *
     * This function is called in prepareStep(), and no adding/removing/exchanging is allowed before runStep()
     */
    void updateSylinderMap(); ///< update sylindermap and sylinderMobilityMap

    /**
     * @brief write VTK parallel XML file into baseFolder
     *
     * @param baseFolder
     */
    void writeVTK(const std::string &baseFolder, const std::string &postfix);

    /**
     * @brief write Ascii file controlled by FDPS into baseFolder
     *
     * @param baseFolder
     */
    void writeAscii(const int stepCount, const std::string &baseFolder, const std::string &postfix);

    /**
     * @brief write a simple legacy VTK file for simBox
     *
     */
    void writeBox();

    /**
     * @brief Get orientation quaternion with givne px,py,pz
     *
     * component in [px,py,pz] out of range [-1,1] will be randomly generated
     * if all out of range [-1,1], a uniformly random orientation on sphere is generated
     * @param orient
     * @param px
     * @param py
     * @param pz
     * @param threadId openmp thread id for random number generation
     */
    void getOrient(Equatn &orient, const double px, const double py, const double pz, const int threadId);

    /**
     * @brief update the rank data field of sylinder
     *
     */
    void updateSylinderRank();

  public:
    SylinderConfig runConfig; ///< system configuration. Be careful if this is modified on the fly

    /**
     * @brief Construct a new SylinderSystem object
     *
     * initialize() should be called after this constructor
     */
    SylinderSystem() = default;

    /**
     * @brief Construct a new SylinderSystem object
     *
     * This constructor calls initialize() internally
     * @param config SylinderConfig object
     * @param argc command line argument
     * @param argv command line argument
     */
    SylinderSystem(const SylinderConfig &config, const Teuchos::RCP<const TCOMM> &commRcp_, 
                   std::shared_ptr<TRngPool> rngPoolPtr_, std::shared_ptr<ConstraintCollector> conCollectorPtr_);

    ~SylinderSystem() = default;

    // forbid copy
    SylinderSystem(const SylinderSystem &) = delete;
    SylinderSystem &operator=(const SylinderSystem &) = delete;

    /**
     * @brief initialize after an empty constructor
     *
     * @param posFile initial configuration. use empty string ("") for no such file
     * @param argc command line argument
     * @param argv command line argument
     */
    void initialize(const std::string &posFile);

    /**
     * @brief reinitialize from vtk files
     *
     * @param restartFile txt file containing timestep and most recent pvtp file names
     * @param argc command line argument
     * @param argv command line argument
     */
    void reinitialize(const std::string &pvtpFileName_, bool eulerStep = true);

    /**
     * @brief compute axis-aligned bounding box of sylinders
     *
     * @param localLow
     * @param localHigh
     * @param globalLow
     * @param globalHigh
     */
    void calcBoundingBox(double localLow[3], double localHigh[3], double globalLow[3], double globalHigh[3]);

    /**
     * @brief compute domain decomposition by sampling sylinder distribution
     *
     * domain decomposition must be triggered when particle distribution significantly changes
     */
    void decomposeDomain();

    /**
     * @brief exchange between mpi ranks according to domain decomposition
     *
     * particle exchange must be triggered every timestep:
     */
    void exchangeSylinder();

    /**
     * one-step high level API
     */
    // get information
    /**
     * @brief Get sylinderContainer
     *
     * @return PS::ParticleSystem<Sylinder>&
     */
    const PS::ParticleSystem<Sylinder> &getContainer() { return sylinderContainer; }
    PS::ParticleSystem<Sylinder> &getContainerNonConst() { return sylinderContainer; }

    /**
     * @brief Get the DomainInfo object
     *
     * @return PS::DomainInfo&
     */
    const PS::DomainInfo &getDomainInfo() { return dinfo; }
    PS::DomainInfo &getDomainInfoNonConst() { return dinfo; }

    const std::unordered_multimap<int, int> &getLinkMap() { return linkMap; }
    const std::unordered_multimap<int, int> &getLinkReverseMap() { return linkReverseMap; }

    /**
     * @brief prepare a step
     *
     * apply simBox boundary condition
     * decomposeDomain() for every 50 steps
     * exchangeSylinder() at every step
     * clear velocity
     * rebuild map
     * compute mobility matrix&operator
     * between prepareStep() and runStep(), sylinders should not be moved, added, or removed
     */
    void prepareStep(const int stepCount);

    // These should run after runStep()
    /**
     * @brief add new Sylinders into the system from all ranks
     *
     * add new sylinders
     * 1. new gids will be randomly generated and assigned to each new sylinder
     * 2. added new sylinders will be appended to the local rank
     *
     * @param newSylinder list of new sylinders.
     * @return the generated new gids of the added new sylinders
     */
    std::vector<int> addNewSylinder(const std::vector<Sylinder> &newSylinder);

    /**
     * @brief add new links into the system from all ranks
     *
     * the newLink will be gathered from all ranks, remove duplication, and synchronized to the linkMap on all ranks
     *
     * @param newLink
     */
    void addNewLink(const std::vector<Link> &newLink);

    /**
     * @brief calculate both Col and Bi stress
     *
     */
    void calcConStress();

    /**
     * @brief calculate polar and nematic order parameter
     *
     * The result is shown on screen
     */
    void calcOrderParameter();

    /**
     * @brief calculate volume fraction
     *
     */
    void calcVolFrac();

    /**
     * detailed low level API
     */
    /**
     * @brief apply periodic boundary condition
     *
     */
    void applyBoxBC();

    /**
     * @brief apply periodic boundary condition
     *
     */
    void applyMonolayer();

    // compute non-collision velocity and mobility, before collision resolution
    /**
     * @brief calculate translational and rotational Brownian motion as specified in runConfig
     *
     * write back to sylinder.velBrown/omegaBrown
     */
    void calcVelocityBrown();

    /**
     * @brief sum vel = velNonB + velB + velCol + velBi
     *
     */
    void sumForceVelocity();

    /**
     * @brief calculate the mobility matrix (block diagonal)
     *
     */
    void calcMobMatrix();

    /**
     * @brief calculate the mobility operator (full-dense, matrix-free)
     *
     * TODO: to be implemented
     */
    void calcMobOperator();

    /**
     * @brief build the ZDD<SylinderNearEP> object
     *
     */
    void buildsylinderNearDataDirectory();

    /**
     * @brief update the ZDD<SylinderNearEP> object without modifying the index
     *
     */
    void updatesylinderNearDataDirectory();


    /**
     * @brief Get the sylinderNearDataDirectory object
     *
     * @return std::shared_ptr<const ZDD<SylinderNearEP>>&
     */
    std::shared_ptr<ZDD<SylinderNearEP>> &getsylinderNearDataDirectory() { return sylinderNearDataDirectoryPtr; }

    // collect constraints
    void collectConstraints();     ///< collect all constraints
    void collectPairCollision();     ///< collect pair collision constraints
    void collectBoundaryCollision(); ///< collect boundary collision constraints
    void collectLinkBilateral();     ///< setup link constraints
    void updatePairCollision();     ///< collect pair collision constraints

    void saveForceVelocityConstraints(const Teuchos::RCP<const TV> &forceRcp, 
                                      const Teuchos::RCP<const TV> &velocityRcp); ///< write back to sylinder.velCol and velBi
    void stepEuler(); ///< Euler step update position and orientation, with both collision and non-collision velocity
    void advanceParticles(); ///< Euler step update position and orientation, with both collision and non-collision velocity

    // write results
    void writeResult(const int stepCount, const std::string &baseFolder, const std::string &postfix); ///< write result regardless of runConfig

    // expose raw vectors and operators

    // mobility
    Teuchos::RCP<TCMAT> getMobMatrix() { return mobilityMatrixRcp; };
    Teuchos::RCP<TOP> getMobOperator() { return mobilityOperatorRcp; };

    // get information
    /**
     * @brief Get the local and global max gid for sylinders
     *
     * @return std::pair<int, int> [localMaxGid,globalMaxGid]
     */
    std::pair<int, int> getMaxGid();
};

#endif