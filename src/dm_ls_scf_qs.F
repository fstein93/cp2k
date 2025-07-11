!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2023 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Routines for a linear scaling quickstep SCF run based on the density
!>        matrix, with a focus on the interface between dm_ls_scf and qs
!> \par History
!>       2011.04 created [Joost VandeVondele]
!> \author Joost VandeVondele
! **************************************************************************************************
MODULE dm_ls_scf_qs
   USE atomic_kind_types,               ONLY: atomic_kind_type
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_cp2k_link,              ONLY: cp_dbcsr_alloc_block_from_nbl
   USE cp_dbcsr_operations,             ONLY: dbcsr_allocate_matrix_set
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_get_default_unit_nr,&
                                              cp_logger_type
   USE cp_realspace_grid_cube,          ONLY: cp_pw_to_cube
   USE dbcsr_api,                       ONLY: &
        dbcsr_complete_redistribute, dbcsr_copy, dbcsr_copy_into_existing, dbcsr_create, &
        dbcsr_desymmetrize, dbcsr_distribution_get, dbcsr_distribution_hold, &
        dbcsr_distribution_new, dbcsr_distribution_release, dbcsr_distribution_type, &
        dbcsr_finalize, dbcsr_get_info, dbcsr_multiply, dbcsr_nblkrows_total, dbcsr_p_type, &
        dbcsr_release, dbcsr_set, dbcsr_type, dbcsr_type_real_8
   USE dm_ls_scf_types,                 ONLY: ls_cluster_atomic,&
                                              ls_cluster_molecular,&
                                              ls_mstruct_type,&
                                              ls_scf_env_type
   USE input_constants,                 ONLY: ls_cluster_atomic,&
                                              ls_cluster_molecular
   USE kinds,                           ONLY: default_string_length,&
                                              dp
   USE message_passing,                 ONLY: mp_para_env_type
   USE particle_list_types,             ONLY: particle_list_type
   USE particle_types,                  ONLY: particle_type
   USE pw_env_types,                    ONLY: pw_env_get,&
                                              pw_env_type
   USE pw_methods,                      ONLY: pw_zero
   USE pw_pool_types,                   ONLY: pw_pool_create_pw,&
                                              pw_pool_give_back_pw,&
                                              pw_pool_p_type,&
                                              pw_pool_type
   USE pw_types,                        ONLY: COMPLEXDATA1D,&
                                              REALDATA3D,&
                                              REALSPACE,&
                                              RECIPROCALSPACE,&
                                              pw_type
   USE qs_atomic_block,                 ONLY: calculate_atomic_block_dm
   USE qs_collocate_density,            ONLY: calculate_rho_elec
   USE qs_density_mixing_types,         ONLY: direct_mixing_nr,&
                                              gspace_mixing_nr
   USE qs_energy_types,                 ONLY: qs_energy_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_gspace_mixing,                ONLY: gspace_mixing
   USE qs_initial_guess,                ONLY: calculate_mopac_dm
   USE qs_kind_types,                   ONLY: qs_kind_type
   USE qs_ks_methods,                   ONLY: qs_ks_update_qs_env
   USE qs_ks_types,                     ONLY: qs_ks_did_change,&
                                              qs_ks_env_type,&
                                              set_ks_env
   USE qs_mixing_utils,                 ONLY: charge_mixing_init,&
                                              mixing_allocate,&
                                              mixing_init
   USE qs_neighbor_list_types,          ONLY: neighbor_list_set_p_type
   USE qs_rho_atom_types,               ONLY: rho_atom_type
   USE qs_rho_methods,                  ONLY: qs_rho_update_rho
   USE qs_rho_types,                    ONLY: qs_rho_get,&
                                              qs_rho_type
   USE qs_subsys_types,                 ONLY: qs_subsys_get,&
                                              qs_subsys_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dm_ls_scf_qs'

   PUBLIC :: matrix_ls_create, matrix_qs_to_ls, matrix_ls_to_qs, ls_scf_init_qs, &
             ls_scf_dm_to_ks, ls_scf_qs_atomic_guess, write_matrix_to_cube, rho_mixing_ls_init, &
             matrix_decluster

CONTAINS

! **************************************************************************************************
!> \brief create a matrix for use (and as a template) in ls based on a qs template
!> \param matrix_ls ...
!> \param matrix_qs ...
!> \param ls_mstruct ...
!> \par History
!>       2011.03 created [Joost VandeVondele]
!>       2015.09 add support for PAO [Ole Schuett]
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE matrix_ls_create(matrix_ls, matrix_qs, ls_mstruct)
      TYPE(dbcsr_type)                                   :: matrix_ls, matrix_qs
      TYPE(ls_mstruct_type), INTENT(IN)                  :: ls_mstruct

      CHARACTER(len=*), PARAMETER                        :: routineN = 'matrix_ls_create'

      CHARACTER(len=default_string_length)               :: name
      INTEGER                                            :: handle, iatom, imol, jatom, &
                                                            ls_data_type, natom, nmol
      INTEGER, ALLOCATABLE, DIMENSION(:), TARGET         :: atom_to_cluster, atom_to_cluster_primus, &
                                                            clustered_blk_sizes, primus_of_mol
      INTEGER, DIMENSION(:), POINTER                     :: clustered_col_dist, clustered_row_dist, &
                                                            ls_blk_sizes, ls_col_dist, ls_row_dist
      TYPE(dbcsr_distribution_type)                      :: ls_dist, ls_dist_clustered

      CALL timeset(routineN, handle)

      ! Defaults -----------------------------------------------------------------------------------
      CALL dbcsr_get_info(matrix_qs, col_blk_size=ls_blk_sizes, distribution=ls_dist)
      CALL dbcsr_distribution_hold(ls_dist)
      CALL dbcsr_distribution_get(ls_dist, row_dist=ls_row_dist, col_dist=ls_col_dist)
      ls_data_type = dbcsr_type_real_8

      ! PAO ----------------------------------------------------------------------------------------
      IF (ls_mstruct%do_pao) THEN
         CALL dbcsr_get_info(ls_mstruct%matrix_A, col_blk_size=ls_blk_sizes)
      END IF

      ! Clustering ---------------------------------------------------------------------------------
      SELECT CASE (ls_mstruct%cluster_type)
      CASE (ls_cluster_atomic)
         ! do nothing
      CASE (ls_cluster_molecular)
         ! create format of the clustered matrix
         natom = dbcsr_nblkrows_total(matrix_qs)
         nmol = MAXVAL(ls_mstruct%atom_to_molecule)
         ALLOCATE (atom_to_cluster_primus(natom))
         ALLOCATE (atom_to_cluster(natom))
         ALLOCATE (primus_of_mol(nmol))
         DO iatom = 1, natom
            atom_to_cluster(iatom) = ls_mstruct%atom_to_molecule(iatom)
            ! the first atom of the molecule is the primus
            ! if the number of atoms per molecule is independent of system size, this is not a quadratic loop
            ! it assumes that all atoms of the molecule are consecutive.
            DO jatom = iatom, 1, -1
               IF (ls_mstruct%atom_to_molecule(jatom) == atom_to_cluster(iatom)) THEN
                  atom_to_cluster_primus(iatom) = jatom
               ELSE
                  EXIT
               END IF
            END DO
            primus_of_mol(atom_to_cluster(iatom)) = atom_to_cluster_primus(iatom)
         END DO

         ! row
         ALLOCATE (clustered_row_dist(nmol))
         DO imol = 1, nmol
            clustered_row_dist(imol) = ls_row_dist(primus_of_mol(imol))
         END DO

         ! col
         ALLOCATE (clustered_col_dist(nmol))
         DO imol = 1, nmol
            clustered_col_dist(imol) = ls_col_dist(primus_of_mol(imol))
         END DO

         ALLOCATE (clustered_blk_sizes(nmol))
         clustered_blk_sizes = 0
         DO iatom = 1, natom
            clustered_blk_sizes(atom_to_cluster(iatom)) = clustered_blk_sizes(atom_to_cluster(iatom)) + &
                                                          ls_blk_sizes(iatom)
         END DO
         ls_blk_sizes => clustered_blk_sizes ! redirect pointer

         ! create new distribution
         CALL dbcsr_distribution_new(ls_dist_clustered, &
                                     template=ls_dist, &
                                     row_dist=clustered_row_dist, &
                                     col_dist=clustered_col_dist, &
                                     reuse_arrays=.TRUE.)
         CALL dbcsr_distribution_release(ls_dist)
         ls_dist = ls_dist_clustered

      CASE DEFAULT
         CPABORT("Unknown LS cluster type")
      END SELECT

      ! Create actual matrix -----------------------------------------------------------------------
      CALL dbcsr_get_info(matrix_qs, name=name)
      CALL dbcsr_create(matrix_ls, &
                        name=name, &
                        dist=ls_dist, &
                        matrix_type="S", &
                        data_type=ls_data_type, &
                        row_blk_size=ls_blk_sizes, &
                        col_blk_size=ls_blk_sizes)
      CALL dbcsr_distribution_release(ls_dist)
      CALL dbcsr_finalize(matrix_ls)

      CALL timestop(handle)

   END SUBROUTINE matrix_ls_create

! **************************************************************************************************
!> \brief first link to QS, copy a QS matrix to LS matrix
!>        used to isolate QS style matrices from LS style
!>        will be useful for future features (e.g. precision, symmetry, blocking, ...)
!> \param matrix_ls ...
!> \param matrix_qs ...
!> \param ls_mstruct ...
!> \param covariant ...
!> \par History
!>       2010.10 created [Joost VandeVondele]
!>       2015.09 add support for PAO [Ole Schuett]
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE matrix_qs_to_ls(matrix_ls, matrix_qs, ls_mstruct, covariant)
      TYPE(dbcsr_type)                                   :: matrix_ls, matrix_qs
      TYPE(ls_mstruct_type), INTENT(IN), TARGET          :: ls_mstruct
      LOGICAL, INTENT(IN)                                :: covariant

      CHARACTER(len=*), PARAMETER                        :: routineN = 'matrix_qs_to_ls'

      INTEGER                                            :: handle
      INTEGER, DIMENSION(:), POINTER                     :: pao_blk_sizes
      TYPE(dbcsr_type)                                   :: matrix_pao, matrix_tmp
      TYPE(dbcsr_type), POINTER                          :: matrix_trafo

      CALL timeset(routineN, handle)

      IF (.NOT. ls_mstruct%do_pao) THEN
         CALL matrix_cluster(matrix_ls, matrix_qs, ls_mstruct)

      ELSE ! using pao
         CALL dbcsr_get_info(ls_mstruct%matrix_A, col_blk_size=pao_blk_sizes)
         CALL dbcsr_create(matrix_pao, &
                           matrix_type="N", &
                           template=matrix_qs, &
                           row_blk_size=pao_blk_sizes, &
                           col_blk_size=pao_blk_sizes)

         matrix_trafo => ls_mstruct%matrix_A ! contra-variant
         IF (covariant) matrix_trafo => ls_mstruct%matrix_B ! co-variant
         CALL dbcsr_create(matrix_tmp, template=matrix_trafo)

         CALL dbcsr_multiply("N", "N", 1.0_dp, matrix_qs, matrix_trafo, 0.0_dp, matrix_tmp)
         CALL dbcsr_multiply("T", "N", 1.0_dp, matrix_trafo, matrix_tmp, 0.0_dp, matrix_pao)
         CALL dbcsr_release(matrix_tmp)

         CALL matrix_cluster(matrix_ls, matrix_pao, ls_mstruct)
         CALL dbcsr_release(matrix_pao)
      END IF

      CALL timestop(handle)

   END SUBROUTINE matrix_qs_to_ls

! **************************************************************************************************
!> \brief Performs molecular blocking and reduction to single precision if enabled
!> \param matrix_out ...
!> \param matrix_in ...
!> \param ls_mstruct ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE matrix_cluster(matrix_out, matrix_in, ls_mstruct)
      TYPE(dbcsr_type)                                   :: matrix_out, matrix_in
      TYPE(ls_mstruct_type), INTENT(IN)                  :: ls_mstruct

      CHARACTER(len=*), PARAMETER                        :: routineN = 'matrix_cluster'

      INTEGER                                            :: handle
      TYPE(dbcsr_type)                                   :: matrix_in_nosym

      CALL timeset(routineN, handle)

      SELECT CASE (ls_mstruct%cluster_type)
      CASE (ls_cluster_atomic)
         CALL dbcsr_copy(matrix_out, matrix_in) ! takes care of an eventual data_type conversion

      CASE (ls_cluster_molecular)
         ! desymmetrize the qs matrix
         CALL dbcsr_create(matrix_in_nosym, template=matrix_in, matrix_type="N")
         CALL dbcsr_desymmetrize(matrix_in, matrix_in_nosym)

         ! perform the magic complete redistribute copy
         CALL dbcsr_complete_redistribute(matrix_in_nosym, matrix_out); 
         CALL dbcsr_release(matrix_in_nosym)

      CASE DEFAULT
         CPABORT("Unknown LS cluster type")
      END SELECT

      CALL timestop(handle)

   END SUBROUTINE matrix_cluster

! **************************************************************************************************
!> \brief second link to QS, copy a LS matrix to QS matrix
!>        used to isolate QS style matrices from LS style
!>        will be useful for future features (e.g. precision, symmetry, blocking, ...)
!> \param matrix_qs ...
!> \param matrix_ls ...
!> \param ls_mstruct ...
!> \param covariant ...
!> \param keep_sparsity If set dbcsr_copy_into_existing will be used, by default set to .TRUE.
!> \par History
!>       2010.10 created [Joost VandeVondele]
!>       2015.09 add support for PAO [Ole Schuett]
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE matrix_ls_to_qs(matrix_qs, matrix_ls, ls_mstruct, covariant, keep_sparsity)
      TYPE(dbcsr_type)                                   :: matrix_qs, matrix_ls
      TYPE(ls_mstruct_type), INTENT(IN), TARGET          :: ls_mstruct
      LOGICAL                                            :: covariant
      LOGICAL, OPTIONAL                                  :: keep_sparsity

      CHARACTER(len=*), PARAMETER                        :: routineN = 'matrix_ls_to_qs'

      INTEGER                                            :: handle
      INTEGER, DIMENSION(:), POINTER                     :: pao_blk_sizes
      LOGICAL                                            :: my_keep_sparsity
      TYPE(dbcsr_type)                                   :: matrix_declustered, matrix_tmp1, &
                                                            matrix_tmp2
      TYPE(dbcsr_type), POINTER                          :: matrix_trafo

      CALL timeset(routineN, handle)

      my_keep_sparsity = .TRUE.
      IF (PRESENT(keep_sparsity)) &
         my_keep_sparsity = keep_sparsity

      IF (.NOT. ls_mstruct%do_pao) THEN
         CALL dbcsr_create(matrix_declustered, template=matrix_qs)
         CALL matrix_decluster(matrix_declustered, matrix_ls, ls_mstruct)
         IF (my_keep_sparsity) THEN
            CALL dbcsr_copy_into_existing(matrix_qs, matrix_declustered) ! preserve sparsity of matrix_qs
         ELSE
            CALL dbcsr_copy(matrix_qs, matrix_declustered) ! overwrite sparsity of matrix_qs
         END IF
         CALL dbcsr_release(matrix_declustered)

      ELSE ! using pao
         CALL dbcsr_get_info(ls_mstruct%matrix_A, col_blk_size=pao_blk_sizes)
         CALL dbcsr_create(matrix_declustered, &
                           template=matrix_qs, &
                           row_blk_size=pao_blk_sizes, &
                           col_blk_size=pao_blk_sizes)

         CALL matrix_decluster(matrix_declustered, matrix_ls, ls_mstruct)

         matrix_trafo => ls_mstruct%matrix_B ! contra-variant
         IF (covariant) matrix_trafo => ls_mstruct%matrix_A ! co-variant
         CALL dbcsr_create(matrix_tmp1, template=matrix_trafo)
         CALL dbcsr_create(matrix_tmp2, template=matrix_qs)
         CALL dbcsr_multiply("N", "N", 1.0_dp, matrix_trafo, matrix_declustered, 0.0_dp, matrix_tmp1)
         CALL dbcsr_multiply("N", "T", 1.0_dp, matrix_tmp1, matrix_trafo, 0.0_dp, matrix_tmp2)
         IF (my_keep_sparsity) THEN
            CALL dbcsr_copy_into_existing(matrix_qs, matrix_tmp2) ! preserve sparsity of matrix_qs
         ELSE
            CALL dbcsr_copy(matrix_qs, matrix_tmp2) ! overwrite sparsity of matrix_qs
         END IF
         CALL dbcsr_release(matrix_declustered)
         CALL dbcsr_release(matrix_tmp1)
         CALL dbcsr_release(matrix_tmp2)
      END IF

      CALL timestop(handle)

   END SUBROUTINE matrix_ls_to_qs

! **************************************************************************************************
!> \brief Reverses molecular blocking and reduction to single precision if enabled
!> \param matrix_out ...
!> \param matrix_in ...
!> \param ls_mstruct ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE matrix_decluster(matrix_out, matrix_in, ls_mstruct)
      TYPE(dbcsr_type)                                   :: matrix_out, matrix_in
      TYPE(ls_mstruct_type), INTENT(IN)                  :: ls_mstruct

      CHARACTER(len=*), PARAMETER                        :: routineN = 'matrix_decluster'

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      SELECT CASE (ls_mstruct%cluster_type)
      CASE (ls_cluster_atomic)
         CALL dbcsr_copy(matrix_out, matrix_in) ! takes care of an eventual data_type conversion

      CASE (ls_cluster_molecular)
         ! perform the magic complete redistribute copy
         CALL dbcsr_complete_redistribute(matrix_in, matrix_out)

      CASE DEFAULT
         CPABORT("Unknown LS cluster type")
      END SELECT

      CALL timestop(handle)

   END SUBROUTINE matrix_decluster

! **************************************************************************************************
!> \brief further required initialization of QS.
!>        Might be factored-out since this seems common code with the other SCF.
!> \param qs_env ...
!> \par History
!>       2010.10 created [Joost VandeVondele]
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE ls_scf_init_qs(qs_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env

      CHARACTER(len=*), PARAMETER                        :: routineN = 'ls_scf_init_qs'

      INTEGER                                            :: handle, ispin, nspin, unit_nr
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_ks, matrix_s
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_orb
      TYPE(qs_ks_env_type), POINTER                      :: ks_env

      NULLIFY (sab_orb)
      CALL timeset(routineN, handle)

      ! get a useful output_unit
      logger => cp_get_default_logger()
      IF (logger%para_env%is_source()) THEN
         unit_nr = cp_logger_get_default_unit_nr(logger, local=.TRUE.)
      ELSE
         unit_nr = -1
      END IF

      ! get basic quantities from the qs_env
      CALL get_qs_env(qs_env, dft_control=dft_control, &
                      matrix_s=matrix_s, &
                      matrix_ks=matrix_ks, &
                      ks_env=ks_env, &
                      sab_orb=sab_orb)

      nspin = dft_control%nspins

      ! we might have to create matrix_ks
      IF (.NOT. ASSOCIATED(matrix_ks)) THEN
         CALL dbcsr_allocate_matrix_set(matrix_ks, nspin)
         DO ispin = 1, nspin
            ALLOCATE (matrix_ks(ispin)%matrix)
            CALL dbcsr_create(matrix_ks(ispin)%matrix, template=matrix_s(1)%matrix)
            CALL cp_dbcsr_alloc_block_from_nbl(matrix_ks(ispin)%matrix, sab_orb)
            CALL dbcsr_set(matrix_ks(ispin)%matrix, 0.0_dp)
         END DO
         CALL set_ks_env(ks_env, matrix_ks=matrix_ks)
      END IF

      CALL timestop(handle)

   END SUBROUTINE ls_scf_init_qs

! **************************************************************************************************
!> \brief get an atomic initial guess
!> \param qs_env ...
!> \param energy ...
!> \par History
!>       2012.11 created [Joost VandeVondele]
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE ls_scf_qs_atomic_guess(qs_env, energy)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      REAL(KIND=dp)                                      :: energy

      CHARACTER(len=*), PARAMETER :: routineN = 'ls_scf_qs_atomic_guess'

      INTEGER                                            :: handle, nspin, unit_nr
      INTEGER, DIMENSION(2)                              :: nelectron_spin
      LOGICAL                                            :: has_unit_metric
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_ks, matrix_s, rho_ao
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_energy_type), POINTER                      :: qs_energy
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(qs_rho_type), POINTER                         :: rho

      CALL timeset(routineN, handle)
      NULLIFY (rho, rho_ao)

      ! get a useful output_unit
      logger => cp_get_default_logger()
      IF (logger%para_env%is_source()) THEN
         unit_nr = cp_logger_get_default_unit_nr(logger, local=.TRUE.)
      ELSE
         unit_nr = -1
      END IF

      ! get basic quantities from the qs_env
      CALL get_qs_env(qs_env, dft_control=dft_control, &
                      matrix_s=matrix_s, &
                      matrix_ks=matrix_ks, &
                      ks_env=ks_env, &
                      energy=qs_energy, &
                      atomic_kind_set=atomic_kind_set, &
                      qs_kind_set=qs_kind_set, &
                      particle_set=particle_set, &
                      has_unit_metric=has_unit_metric, &
                      para_env=para_env, &
                      nelectron_spin=nelectron_spin, &
                      rho=rho)

      CALL qs_rho_get(rho, rho_ao=rho_ao)

      nspin = dft_control%nspins

      ! create an initial atomic guess
      IF (dft_control%qs_control%dftb .OR. dft_control%qs_control%semi_empirical .OR. &
          dft_control%qs_control%xtb) THEN
         CALL calculate_mopac_dm(rho_ao, matrix_s(1)%matrix, has_unit_metric, &
                                 dft_control, particle_set, atomic_kind_set, qs_kind_set, &
                                 nspin, nelectron_spin, para_env)
      ELSE
         CALL calculate_atomic_block_dm(rho_ao, matrix_s(1)%matrix, atomic_kind_set, qs_kind_set, &
                                        nspin, nelectron_spin, unit_nr, para_env)
      END IF

      CALL qs_rho_update_rho(rho, qs_env=qs_env)
      CALL qs_ks_did_change(qs_env%ks_env, rho_changed=.TRUE.)
      CALL qs_ks_update_qs_env(qs_env, calculate_forces=.FALSE., just_energy=.FALSE.)

      energy = qs_energy%total

      CALL timestop(handle)

   END SUBROUTINE ls_scf_qs_atomic_guess

! **************************************************************************************************
!> \brief use the density matrix in ls_scf_env to compute the new energy and KS matrix
!> \param qs_env ...
!> \param ls_scf_env ...
!> \param energy_new ...
!> \param iscf ...
!> \par History
!>       2011.04 created [Joost VandeVondele]
!>       2015.02 added gspace density mixing [Patrick Seewald]
!> \author Joost VandeVondele
! **************************************************************************************************
   SUBROUTINE ls_scf_dm_to_ks(qs_env, ls_scf_env, energy_new, iscf)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(ls_scf_env_type)                              :: ls_scf_env
      REAL(KIND=dp)                                      :: energy_new
      INTEGER, INTENT(IN)                                :: iscf

      CHARACTER(len=*), PARAMETER                        :: routineN = 'ls_scf_dm_to_ks'

      INTEGER                                            :: handle, ispin, nspin, unit_nr
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: rho_ao
      TYPE(mp_para_env_type), POINTER                    :: para_env
      TYPE(qs_energy_type), POINTER                      :: energy
      TYPE(qs_rho_type), POINTER                         :: rho

      NULLIFY (energy, rho, rho_ao)
      CALL timeset(routineN, handle)

      logger => cp_get_default_logger()
      IF (logger%para_env%is_source()) THEN
         unit_nr = cp_logger_get_default_unit_nr(logger, local=.TRUE.)
      ELSE
         unit_nr = -1
      END IF

      nspin = ls_scf_env%nspins
      CALL get_qs_env(qs_env, para_env=para_env, energy=energy, rho=rho)
      CALL qs_rho_get(rho, rho_ao=rho_ao)

      ! set the new density matrix
      DO ispin = 1, nspin
         CALL matrix_ls_to_qs(rho_ao(ispin)%matrix, ls_scf_env%matrix_p(ispin), &
                              ls_scf_env%ls_mstruct, covariant=.FALSE.)
      END DO

      ! compute the corresponding KS matrix and new energy, mix density if requested
      CALL qs_rho_update_rho(rho, qs_env=qs_env)
      IF (ls_scf_env%do_rho_mixing) THEN
         IF (ls_scf_env%density_mixing_method .EQ. direct_mixing_nr) &
            CPABORT("Direct P mixing not implemented in linear scaling SCF. ")
         IF (ls_scf_env%density_mixing_method >= gspace_mixing_nr) THEN
            IF (iscf .GT. MAX(ls_scf_env%mixing_store%nskip_mixing, 1)) THEN
               CALL gspace_mixing(qs_env, ls_scf_env%density_mixing_method, &
                                  ls_scf_env%mixing_store, rho, para_env, &
                                  iscf - 1)
               IF (unit_nr > 0) THEN
                  WRITE (unit_nr, '(A57)') &
                     "*********************************************************"
                  WRITE (unit_nr, '(A13,F5.3,A20,A6,A7,I3)') &
                     " Using ALPHA=", ls_scf_env%mixing_store%alpha, &
                     " to mix rho: method=", ls_scf_env%mixing_store%iter_method, ", iscf=", iscf
                  WRITE (unit_nr, '(A8,F5.3,A6,F5.3,A8)') &
                     " rho_nw=", ls_scf_env%mixing_store%alpha, "*rho + ", &
                     1.0_dp - ls_scf_env%mixing_store%alpha, "*rho_old"
                  WRITE (unit_nr, '(A57)') &
                     "*********************************************************"
               END IF
            END IF
         END IF
      END IF

      CALL qs_ks_did_change(qs_env%ks_env, rho_changed=.TRUE.)
      CALL qs_ks_update_qs_env(qs_env, calculate_forces=.FALSE., &
                               just_energy=.FALSE., print_active=.TRUE.)
      energy_new = energy%total

      CALL timestop(handle)

   END SUBROUTINE ls_scf_dm_to_ks

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param ls_scf_env ...
!> \param matrix_p_ls ...
!> \param unit_nr ...
!> \param title ...
!> \param stride ...
! **************************************************************************************************
   SUBROUTINE write_matrix_to_cube(qs_env, ls_scf_env, matrix_p_ls, unit_nr, title, stride)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(ls_scf_env_type)                              :: ls_scf_env
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_p_ls
      INTEGER, INTENT(IN)                                :: unit_nr
      CHARACTER(LEN=*), INTENT(IN)                       :: title
      INTEGER, DIMENSION(:), POINTER                     :: stride

      CHARACTER(len=*), PARAMETER :: routineN = 'write_matrix_to_cube'

      INTEGER                                            :: handle
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_ks
      TYPE(dbcsr_type), TARGET                           :: matrix_p_qs
      TYPE(particle_list_type), POINTER                  :: particles
      TYPE(pw_env_type), POINTER                         :: pw_env
      TYPE(pw_pool_p_type), DIMENSION(:), POINTER        :: pw_pools
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(pw_type)                                      :: wf_g, wf_r
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(qs_subsys_type), POINTER                      :: subsys

      CALL timeset(routineN, handle)

      NULLIFY (ks_env, pw_env, auxbas_pw_pool, pw_pools, particles, subsys, matrix_ks)

      CALL get_qs_env(qs_env, &
                      ks_env=ks_env, &
                      subsys=subsys, &
                      pw_env=pw_env, &
                      matrix_ks=matrix_ks)

      CALL qs_subsys_get(subsys, particles=particles)

      ! convert the density matrix (ls style) to QS style
      CALL dbcsr_copy(matrix_p_qs, matrix_ks(1)%matrix)
      CALL dbcsr_set(matrix_p_qs, 0.0_dp) !zero matrix creation
      CALL matrix_ls_to_qs(matrix_p_qs, matrix_p_ls, ls_scf_env%ls_mstruct, covariant=.FALSE.)

      ! Print total electronic density
      CALL pw_env_get(pw_env=pw_env, &
                      auxbas_pw_pool=auxbas_pw_pool, &
                      pw_pools=pw_pools)
      CALL pw_pool_create_pw(pool=auxbas_pw_pool, &
                             pw=wf_r, &
                             use_data=REALDATA3D, &
                             in_space=REALSPACE)
      CALL pw_zero(wf_r)
      CALL pw_pool_create_pw(pool=auxbas_pw_pool, &
                             pw=wf_g, &
                             use_data=COMPLEXDATA1D, &
                             in_space=RECIPROCALSPACE)
      CALL pw_zero(wf_g)
      CALL calculate_rho_elec(matrix_p=matrix_p_qs, &
                              rho=wf_r, &
                              rho_gspace=wf_g, &
                              ks_env=ks_env)

      ! write this to a cube
      CALL cp_pw_to_cube(wf_r, unit_nr=unit_nr, title=title, &
                         particles=particles, stride=stride)

      !free memory
      CALL pw_pool_give_back_pw(auxbas_pw_pool, wf_r)
      CALL pw_pool_give_back_pw(auxbas_pw_pool, wf_g)
      CALL dbcsr_release(matrix_p_qs)

      CALL timestop(handle)

   END SUBROUTINE write_matrix_to_cube

! **************************************************************************************************
!> \brief Initialize g-space density mixing
!> \param qs_env ...
!> \param ls_scf_env ...
! **************************************************************************************************
   SUBROUTINE rho_mixing_ls_init(qs_env, ls_scf_env)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(ls_scf_env_type)                              :: ls_scf_env

      CHARACTER(len=*), PARAMETER :: routineN = 'rho_mixing_ls_init'

      INTEGER                                            :: handle
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(qs_rho_type), POINTER                         :: rho
      TYPE(rho_atom_type), DIMENSION(:), POINTER         :: rho_atom

      CALL timeset(routineN, handle)

      CALL get_qs_env(qs_env, dft_control=dft_control, rho=rho)

      CALL mixing_allocate(qs_env, ls_scf_env%density_mixing_method, nspins=ls_scf_env%nspins, &
                           mixing_store=ls_scf_env%mixing_store)
      IF (ls_scf_env%density_mixing_method >= gspace_mixing_nr) THEN
         IF (dft_control%qs_control%gapw) THEN
            CALL get_qs_env(qs_env, rho_atom_set=rho_atom)
            CALL mixing_init(ls_scf_env%density_mixing_method, rho, ls_scf_env%mixing_store, &
                             ls_scf_env%para_env, rho_atom=rho_atom)
         ELSEIF (dft_control%qs_control%dftb .OR. dft_control%qs_control%xtb) THEN
            CALL charge_mixing_init(ls_scf_env%mixing_store)
         ELSEIF (dft_control%qs_control%semi_empirical) THEN
            CPABORT('SE Code not possible')
         ELSE
            CALL mixing_init(ls_scf_env%density_mixing_method, rho, ls_scf_env%mixing_store, &
                             ls_scf_env%para_env)
         END IF
      END IF
      CALL timestop(handle)
   END SUBROUTINE rho_mixing_ls_init

END MODULE dm_ls_scf_qs
