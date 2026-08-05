!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: BSD-3-Clause                                                          !
!--------------------------------------------------------------------------------------------------!

MODULE mp2_gpu_api
   USE ISO_C_BINDING, ONLY   : C_DOUBLE, C_INT, C_BOOL, C_PTR, C_LOC, C_NULL_PTR, C_F_POINTER
   USE kinds, ONLY           : dp
   USE message_passing,  ONLY: mp_comm_type, mp_para_env_type
   USE mp2_types,        ONLY: mp2_type
   USE group_dist_types, ONLY: group_dist_d1_type, get_group_dist, maxsize

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: calc_ri_mp2_energy

CONTAINS

   SUBROUTINE calc_ri_mp2_energy( &
      E_cou, &
      E_ex, &
      E_s, &
      E_t, &
      BIb_C, &
      comm_all, &
      comm_sub, &
      eigenval, &
      homo, &
      nmo, &
      dimen_RI, &
      calc_forces, &
      mp2_env, &
      gd_array, &
      gd_B_virtual, &
      virtual_start, &
      virtual_size, &
      aux_start, &
      aux_size, &
      preferred_dgemm_lib, &
      para_env_sub, &
      color_sub)
      REAL(KIND=dp), INTENT(OUT)                         :: E_cou, E_ex, E_s, E_t
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER, &
                           CONTIGUOUS, INTENT(INOUT)     :: BIb_C
      CLASS(mp_comm_type), INTENT(IN)                    :: comm_all, comm_sub
      REAL(KIND=dp), DIMENSION(:), INTENT(IN)            :: eigenval
      ! INTEGER, DIMENSION(:), INTENT(IN)                  :: homo
      INTEGER, INTENT(IN)                                :: homo, nmo, dimen_RI
      LOGICAL, INTENT(IN)                                :: calc_forces
      TYPE(mp2_type)                                     :: mp2_env
      TYPE(group_dist_d1_type), INTENT(IN)               :: gd_array
      TYPE(group_dist_d1_type), INTENT(IN)               :: gd_B_virtual
      INTEGER, INTENT(IN)                                :: virtual_start
      INTEGER, INTENT(IN)                                :: virtual_size
      INTEGER, INTENT(IN)                                :: aux_start
      INTEGER, INTENT(IN)                                :: aux_size
      INTEGER, INTENT(IN)                                :: preferred_dgemm_lib
      TYPE(mp_para_env_type), INTENT(IN), POINTER        :: para_env_sub
      INTEGER, INTENT(IN)                                :: color_sub

      INTEGER(C_INT)  :: comm_all_handle, comm_sub_handle
      INTEGER(C_INT)  :: nmo_c, dimen_RI_c
      INTEGER(C_INT)  :: gd_array_size
      INTEGER(C_INT)  :: maxsize_gd_array, maxsize_gd_B_virtual, maxval_gd_B_virtual
      ! INTEGER(C_INT), ALLOCATABLE, &
      ! TARGET          :: gd_array(:)
      INTEGER(C_INT), ALLOCATABLE, &
      TARGET          :: gd_B_virtual_sizes(:)
      LOGICAL(C_BOOL) :: calc_forces_c
      INTEGER(C_INT)  :: unit_nr_c, preferred_dgemm_lib_c
      REAL(C_DOUBLE)  :: scale_S, scale_T, mp2_memory
      INTEGER(C_INT)  :: user_block_size

      INTERFACE
         SUBROUTINE calc_ri_mp2_energy_c( &
            E_cou, &
            E_ex, &
            E_s, &
            E_t, &
            BIb_C, &
            mp2_memory, &
            user_block_size, &
            scale_S, &
            scale_T, &
            comm_all, &
            comm_sub, &
            color_sub, &
            gd_array_sizes, &
            gd_array_sizes_size, &
            gd_B_virtual_sizes, &
            gd_B_virtual_sizes_size, &
            eigenval, &
            homo, &
            nmo, &
            dimen_RI, &
            calc_forces_c, &
            aux_start, &
            aux_size, &
            maxsize_gd_array, &
            maxsize_gd_B_virtual, &
            maxval_gd_B_virtual, &
            preferred_dgemm_lib)
            IMPORT :: C_DOUBLE, C_INT, C_BOOL, C_PTR
            BIND(C, name="calc_ri_mp2_energy")
            REAL(C_DOUBLE)               :: E_cou, E_ex, E_s, E_t
            TYPE(C_PTR)                  :: BIb_C
            REAL(C_DOUBLE), VALUE        :: mp2_memory
            INTEGER(C_INT), VALUE        :: user_block_size
            REAL(C_DOUBLE), VALUE        :: scale_S, scale_T
            INTEGER(C_INT), VALUE        :: comm_all, comm_sub
            INTEGER(C_INT), VALUE        :: color_sub
            INTEGER(C_INT), DIMENSION(*) :: gd_array_sizes
            INTEGER(C_INT), VALUE        :: gd_array_sizes_size
            INTEGER(C_INT), DIMENSION(*) :: gd_B_virtual_sizes
            INTEGER(C_INT), VALUE        :: gd_B_virtual_sizes_size
            REAL(C_DOUBLE), DIMENSION(*) :: eigenval
            INTEGER(C_INT), VALUE        :: homo
            INTEGER(C_INT), VALUE        :: nmo
            INTEGER(C_INT), VALUE        :: dimen_RI
            LOGICAL(C_BOOL), VALUE       :: calc_forces_c
            INTEGER(C_INT), VALUE        :: aux_start
            INTEGER(C_INT), VALUE        :: aux_size
            INTEGER(C_INT), VALUE        :: maxsize_gd_array
            INTEGER(C_INT), VALUE        :: maxsize_gd_B_virtual
            INTEGER(C_INT), VALUE        :: maxval_gd_B_virtual
            INTEGER(C_INT), VALUE        :: preferred_dgemm_lib
         END SUBROUTINE calc_ri_mp2_energy_c
      END INTERFACE

      ! MPI communicators
      comm_all_handle = comm_all%get_handle()
      comm_sub_handle = comm_sub%get_handle()

      ALLOCATE(gd_B_virtual_sizes(SIZE(homo)))

      mp2_memory = mp2_env%mp2_memory
      scale_S = mp2_env%scale_S
      scale_T = mp2_env%scale_T
      user_block_size = mp2_env%ri_mp2%block_size
      calc_forces_c = calc_forces

      ! set sizes
      maxsize_gd_array = maxsize(gd_array)
      maxsize_gd_B_virtual = MAXVAL(gd_B_virtual_sizes)
      maxval_gd_B_virtual = maxsize_gd_B_virtual

      ! === KEEP ONLY _j FAMILY VARS ===============
      CALL calc_ri_mp2_energy_c( &
         E_cou, &
         E_ex, &
         E_s, &
         E_t, &
         C_LOC(BIb_C), &
         mp2_memory, &
         user_block_size, &
         scale_S, &
         scale_T, &
         comm_all_handle, &
         comm_sub_handle, &
         color_sub, &
         gd_array%sizes, &
         SIZE(gd_array%sizes), &
         gd_B_virtual%sizes, &
         SIZE(gd_B_virtual%sizes), &
         eigenval, &
         homo, &
         nmo, &
         dimen_RI, &
         calc_forces_c, &
         aux_start, &
         aux_size, &
         maxsize_gd_array, &
         maxsize_gd_B_virtual, &
         maxval_gd_B_virtual, &
         preferred_dgemm_lib)

      DEALLOCATE(gd_B_virtual_sizes)
      NULLIFY(BIb_C)

   END SUBROUTINE calc_ri_mp2_energy
END MODULE mp2_gpu_api
