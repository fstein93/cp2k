!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Transfers densities from PW to RS grids and potentials from PW to RS
!> \par History
!>      - Copied from qs_coolocate_Density and qs_integrate_potenntial
!> \author JGH (04.2014)
! **************************************************************************************************
MODULE rs_pw_interface
   USE grid_api,                        ONLY: grid_copy_from_multigrid_single,&
                                              grid_copy_to_multigrid_single,&
                                              grid_multigrid_type
   USE pw_env_types,                    ONLY: pw_env_get,&
                                              pw_env_type
   USE pw_methods,                      ONLY: pw_transfer
   USE pw_pool_types,                   ONLY: pw_pool_p_type
   USE pw_types,                        ONLY: pw_c1d_gs_type,&
                                              pw_r3d_rs_type
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'rs_pw_interface'
! *** Public subroutines ***

   PUBLIC :: density_rs2pw, &
             potential_pw2rs

CONTAINS

! **************************************************************************************************
!> \brief given partial densities on the realspace multigrids,
!>      computes the full density on the plane wave grids, both in real and
!>      gspace
!> \param pw_env ...
!> \param rho ...
!> \param rho_gspace ...
!> \param multigrid_ext ...
!> \note
!>      should contain all communication in the collocation of the density
!>      in the case of replicated grids
! **************************************************************************************************
   SUBROUTINE density_rs2pw(pw_env, rho, rho_gspace, multigrid_ext)

      TYPE(pw_env_type), INTENT(IN)                      :: pw_env
      TYPE(pw_r3d_rs_type), INTENT(INOUT)                :: rho
      TYPE(pw_c1d_gs_type), INTENT(INOUT)                :: rho_gspace
      TYPE(grid_multigrid_type), INTENT(IN), OPTIONAL    :: multigrid_ext

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'density_rs2pw'

      INTEGER                                            :: handle
      TYPE(grid_multigrid_type)                          :: multigrid

      CALL timeset(routineN, handle)
      CALL pw_env_get(pw_env, multigrid=multigrid)

      IF (PRESENT(multigrid_ext)) THEN
         CALL grid_copy_from_multigrid_single(multigrid_ext, rho, rho_gspace)
      ELSE
         CALL grid_copy_from_multigrid_single(multigrid, rho, rho_gspace)
      END IF

      IF (rho_gspace%pw_grid%spherical) CALL pw_transfer(rho_gspace, rho)

      CALL timestop(handle)

   END SUBROUTINE density_rs2pw

! **************************************************************************************************
!> \brief transfers a potential from a pw_grid to a vector of
!>      realspace multigrids
!> \param v_rspace INPUT : the potential on a planewave grid in Rspace
!> \param pw_env ...
!> \par History
!>      09.2006 created [Joost VandeVondele]
!> \note
!>      extracted from integrate_v_rspace
!>      should contain all parallel communication of integrate_v_rspace in the
!>      case of replicated grids.
! **************************************************************************************************
   SUBROUTINE potential_pw2rs(v_rspace, pw_env)
      TYPE(pw_r3d_rs_type), INTENT(IN)                   :: v_rspace
      TYPE(pw_env_type), INTENT(IN)                      :: pw_env

      CHARACTER(len=*), PARAMETER                        :: routineN = 'potential_pw2rs'

      INTEGER                                            :: handle
      TYPE(grid_multigrid_type)                          :: multigrid
      TYPE(pw_pool_p_type), DIMENSION(:), POINTER        :: pw_pools

      CALL timeset(routineN, handle)

      ! *** set up of the potential on the multigrids
      CALL pw_env_get(pw_env, pw_pools=pw_pools, multigrid=multigrid)

      CALL grid_copy_to_multigrid_single(multigrid, v_rspace)

      CALL timestop(handle)

   END SUBROUTINE potential_pw2rs

END MODULE rs_pw_interface
