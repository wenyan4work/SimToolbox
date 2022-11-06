/**
 * @file ParticleSystem.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief System for particles
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef SYLINDERSYSTEM_HPP_
#define SYLINDERSYSTEM_HPP_

#include "Particle.hpp"
#include "ParticleConfig.hpp"
#include "ParticleNear.hpp"

#include "Boundary/Boundary.hpp"
// #include "Constraint/ConstraintSolver.hpp"
#include "FDPS/particle_simulator.hpp"
#include "Trilinos/TpetraUtil.hpp"
#include "Trilinos/ZDD.hpp"
#include "Util/TRngPool.hpp"

#include <unordered_map>
#include <string>

/**
 * @brief A collection of particles distributed to multiple MPI ranks.
 *
 */
class ParticleSystem {
    // FDPS stuff
    PS::DomainInfo dinfo; ///< domain size, boundary condition, and decomposition info
    void setDomainInfo();

    PS::ParticleSystem<Particle> particleContainer;        ///< particles
    std::unique_ptr<TreeParticleNear> treeParticleNearPtr; ///< short range interaction of particles
    int treeParticleNumber;                                ///< the current max_glb number of treeParticleNear
    void setTreeParticle();

    std::unordered_multimap<int, int> linkMap;        ///< links prev,next
    std::unordered_multimap<int, int> linkReverseMap; ///< links next, prev

    // constraint stuff
    std::shared_ptr<ConstraintCollector> conCollectorPtr; ///<  pointer to ConstraintCollector

    // MPI stuff
    std::shared_ptr<TRngPool> rngPoolPtr;      ///< TRngPool object for thread-safe random number generation
    const Teuchos::RCP<const TCOMM> commRcp;   ///< TCOMM, set as a Teuchos::MpiComm object in constructor
    Teuchos::RCP<TMAP> particleMapRcp;         ///< TMAP, contiguous and sequentially ordered 1 dof per particle
    Teuchos::RCP<TMAP> particleMobilityMapRcp; ///< TMAP, contiguous and sequentially ordered 6 dofs per particle
    Teuchos::RCP<TMAP> particleStressMapRcp;   ///< TMAP, contiguous and sequentially ordered 9 dofs per particle
    Teuchos::RCP<TCMAT> mobilityMatrixRcp;     ///< block-diagonal mobility matrix
    Teuchos::RCP<TCMAT> mobilityMatrixInvRcp;  ///< block-diagonal inverse mobility matrix
    Teuchos::RCP<TOP> mobilityOperatorRcp;     ///< full mobility operator (matrix-free), to be implemented
    Teuchos::RCP<TOP> mobilityInvOperatorRcp;  ///< full inverse mobility operator (matrix-free), to be implemented

    // Data directory
    std::shared_ptr<ZDD<ParticleNearEP>> particleNearDataDirectoryPtr; ///< distributed data directory for particle data

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
     * This function move the position of all particles into a cylindrical tube fit in initBox
     */
    void setInitialCircularCrossSection();

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
     * @brief update the rank data field of particle
     *
     */
    void updateParticleRank();

  public:
    ParticleConfig runConfig; ///< system configuration. Be careful if this is modified on the fly

    /**
     * @brief Construct a new ParticleSystem object
     *
     * initialize() should be called after this constructor
     */
    ParticleSystem() = default;

    /**
     * @brief Construct a new ParticleSystem object
     *
     * This constructor calls initialize() internally
     * @param config ParticleConfig object
     * @param argc command line argument
     * @param argv command line argument
     */
    ParticleSystem(const ParticleConfig &config, const Teuchos::RCP<const TCOMM> &commRcp_, 
                   std::shared_ptr<TRngPool> rngPoolPtr_, std::shared_ptr<ConstraintCollector> conCollectorPtr_);

    ~ParticleSystem() = default;

    // forbid copy
    ParticleSystem(const ParticleSystem &) = delete;
    ParticleSystem &operator=(const ParticleSystem &) = delete;

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
    void reinitialize(const std::string &pvtpFileName_);

    void initParticleGrowth();
    
    void calcParticleGrowth(const Teuchos::RCP<const TV> &ptcStressRcp);

    void calcParticleDivision();

    /**
     * @brief compute axis-aligned bounding box of particles
     *
     * @param localLow
     * @param localHigh
     * @param globalLow
     * @param globalHigh
     */
    void calcBoundingBox(double localLow[3], double localHigh[3], double globalLow[3], double globalHigh[3]);

    /**
     * @brief compute domain decomposition by sampling particle distribution
     *
     * domain decomposition must be triggered when particle distribution significantly changes
     */
    void decomposeDomain();

    /**
     * @brief exchange between mpi ranks according to domain decomposition
     *
     * particle exchange must be triggered every timestep:
     */
    void exchangeParticle();

    /**
     * one-step high level API
     */
    // get information
    /**
     * @brief Get particleContainer
     *
     * @return PS::ParticleSystem<Particle>&
     */
    const PS::ParticleSystem<Particle> &getContainer() { return particleContainer; }
    PS::ParticleSystem<Particle> &getContainerNonConst() { return particleContainer; }

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
     * exchangeParticle() at every step
     * clear velocity
     * rebuild map
     * compute mobility matrix&operator
     * between prepareStep() and runStep(), particles should not be moved, added, or removed
     */
    void prepareStep(const int stepCount);

    // These should run after runStep()
    /**
     * @brief add new Particles into the system from all ranks
     *
     * add new particles
     * 1. new gids will be randomly generated and assigned to each new particle
     * 2. added new particles will be appended to the local rank
     *
     * @param newParticle list of new particles.
     * @return the generated new gids of the added new particles
     */
    std::vector<int> addNewParticle(const std::vector<Particle> &newParticle);

    /**
     * @brief add new links into the system from all ranks
     *
     * the newLink will be gathered from all ranks, remove duplication, and synchronized to the linkMap on all ranks
     *
     * @param newLink
     */
    void addNewLink(const std::vector<Link> &newLink);

    /**
     * @brief update the particleMap and particleMobilityMap
     *
     * This function MUST be called after adding adding/removing/exchanging to update the corresponding map
     */
    void updateParticleMap(); ///< update particlemap and particleMobilityMap

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
     * write back to particle.velBrown/omegaBrown
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
     * @brief build the ZDD<ParticleNearEP> object
     *
     */
    void buildparticleNearDataDirectory();

    /**
     * @brief update the ZDD<ParticleNearEP> object without modifying the index
     *
     */
    void updateparticleNearDataDirectory();


    /**
     * @brief Get the particleNearDataDirectory object
     *
     * @return std::shared_ptr<const ZDD<ParticleNearEP>>&
     */
    std::shared_ptr<ZDD<ParticleNearEP>> &getparticleNearDataDirectory() { return particleNearDataDirectoryPtr; }

    // collect constraints
    void collectConstraints();     ///< collect all constraints
    void collectPairCollision();     ///< collect pair collision constraints
    void collectBoundaryCollision(); ///< collect boundary collision constraints
    void collectLinkBilateral();     ///< setup link constraints
    void updatePairCollision();     ///< collect pair collision constraints
    void collectUnresolvedConstraints(); ///< collect unresolved constraints
     
    void getForceVelocityNonConstraint(const Teuchos::RCP<TV> &forceNCRcp, 
                                      const Teuchos::RCP<TV> &velocityNCRcp) const; ///< fill force and velocity nonconstraint

    void saveForceVelocityConstraints(const Teuchos::RCP<const TV> &forceRcp, 
                                      const Teuchos::RCP<const TV> &velocityRcp); ///< write back to particle.velCol and velBi

                                      
    void stepEuler(const int stepType=0); ///< Euler step update position and orientation, with both collision and non-collision velocity
    void resetConfiguration(); ///< reset position and orientation, to the stores position/orientation
    void advanceParticles(); ///< Euler step update position and orientation, with both collision and non-collision velocity

    // write results
    void writeResult(const int stepCount, const std::string &baseFolder, const std::string &postfix); ///< write result regardless of runConfig

    // expose raw vectors and operators

    // get information
    Teuchos::RCP<const TCMAT> getMobMatrix() { return mobilityMatrixRcp; };
    Teuchos::RCP<const TOP> getMobOperator() { return mobilityOperatorRcp; };
    Teuchos::RCP<const TMAP> getParticleMap() { return particleMapRcp; };
    Teuchos::RCP<const TMAP> getParticleMobilityMap() { return particleMobilityMapRcp; };
    Teuchos::RCP<const TMAP> getParticleStressMap() { return particleStressMapRcp; };

    /**
     * @brief Get the local and global max gid for particles
     *
     * @return std::pair<int, int> [localMaxGid,globalMaxGid]
     */
    std::pair<int, int> getMaxGid();
};

#endif