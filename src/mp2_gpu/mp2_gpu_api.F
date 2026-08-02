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
      unit_nr, &
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
      REAL(KIND=dp), DIMENSION(:, :, :), INTENT(IN)      :: BIb_C
      CLASS(mp_comm_type), INTENT(IN)                    :: comm_all, comm_sub
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: eigenval
      INTEGER, DIMENSION(:), INTENT(IN)                  :: homo
      INTEGER, INTENT(IN)                                :: nmo, dimen_RI
      LOGICAL, INTENT(IN)                                :: calc_forces
      INTEGER, INTENT(IN)                                :: unit_nr
      TYPE(mp2_type)                                     :: mp2_env
      TYPE(group_dist_d1_type), INTENT(IN), TARGET       :: gd_array
      TYPE(group_dist_d1_type), DIMENSION(SIZE(homo)), &
         INTENT(INOUT)                                   :: gd_B_virtual
      INTEGER, INTENT(IN)                                :: virtual_start
      INTEGER, INTENT(IN)                                :: virtual_size
      INTEGER, INTENT(IN)                                :: aux_start
      INTEGER, INTENT(IN)                                :: aux_size
      INTEGER, INTENT(IN)                                :: preferred_dgemm_lib
      TYPE(mp_para_env_type), INTENT(IN), POINTER        :: para_env_sub
      INTEGER, INTENT(IN)                                :: color_sub

      INTEGER(C_INT)  :: comm_all_handle, comm_sub_handle
      INTEGER(C_INT)  :: homo_i, homo_j, nmo_c, dimen_RI_c
      INTEGER(C_INT)  :: my_B_size_i, my_B_size_j
      INTEGER(C_INT)  :: my_B_virtual_start_i, my_B_virtual_end_i
      INTEGER(C_INT)  :: my_B_virtual_start_j, my_B_virtual_end_j
      INTEGER(C_INT)  :: gd_array_size
      INTEGER(C_INT)  :: my_group_L_start, my_group_L_end, my_group_L_size
      INTEGER(C_INT)  :: maxsize_gd_array, maxsize_gd_B_virtual, maxval_gd_B_virtual
      INTEGER(C_INT), ALLOCATABLE, &
      TARGET          :: gd_array_sizes(:)
      INTEGER(C_INT), ALLOCATABLE, &
      TARGET          :: gd_B_virtual_sizes(:)
      LOGICAL(C_BOOL) :: calc_forces_c
      INTEGER(C_INT)  :: unit_nr_c, preferred_dgemm_lib_c
      INTEGER(C_INT)  :: nspins, jspin, ispin, num_pe
      INTEGER(C_INT), &
      ALLOCATABLE     :: rec_B_virtual_start_i(:)
      INTEGER(C_INT), &
      ALLOCATABLE     :: rec_B_virtual_start_j(:)
      INTEGER(C_INT), &
      ALLOCATABLE     :: rec_B_sizes_i(:), rec_B_sizes_j(:)
      INTEGER(C_INT), &
      ALLOCATABLE     :: rec_B_virtual_end_i(:)
      INTEGER(C_INT), &
      ALLOCATABLE     :: rec_B_virtual_end_j(:)
      REAL(C_DOUBLE)  :: scale_S, scale_T, mp2_memory
      INTEGER(C_INT)  :: user_block_size
      INTEGER(C_INT)  :: my_B_virtual_start(2)
      INTEGER(C_INT)  :: my_B_virtual_end(2)
      INTEGER(C_INT)  :: my_B_size(2)

      INTERFACE
         SUBROUTINE calc_ri_mp2_energy_c( &
            E_cou, &
            E_ex, &
            E_s, &
            E_t, &
            BIb_C_j, &
            BIb_C_i, &
            mp2_memory, &
            user_block_size, &
            scale_S, &
            scale_T, &
            comm_all, &
            comm_sub, &
            color_sub, &
            gd_array_ptr, &
            gd_array_sizes, &
            gd_array_sizes_size, &
            gd_B_virtual_sizes, &
            gd_B_virtual_sizes_size, &
            eigenval_j, &
            eigenval_i, &
            homo_j, &
            homo_i, &
            nmo, &
            dimen_RI, &
            calc_forces_c, &
            unit_nr, &
            my_B_size_j, &
            my_B_size_i, &
            my_B_virtual_start_j, &
            my_B_virtual_end_j, &
            my_B_virtual_start_i, &
            my_B_virtual_end_i, &
            aux_start, aux_size, &
            maxsize_gd_array, &
            maxsize_gd_B_virtual, &
            maxval_gd_B_virtual, &
            rec_B_virtual_start_j, &
            rec_B_virtual_start_i, &
            rec_B_virtual_end_j, &
            rec_B_virtual_end_i, &
            rec_B_sizes_j, &
            rec_B_sizes_i, &
            num_pe, &
            my_group_L_start, &
            my_group_L_end, &
            my_group_L_size, &
            preferred_dgemm_lib)
            IMPORT :: C_DOUBLE, C_INT, C_BOOL, C_PTR
            BIND(C, name="calc_ri_mp2_energy")
            REAL(C_DOUBLE)               :: E_cou, E_ex, E_s, E_t
            REAL(C_DOUBLE), DIMENSION(*) :: BIb_C_j
            REAL(C_DOUBLE), DIMENSION(*) :: BIb_C_i
            REAL(C_DOUBLE), VALUE        :: mp2_memory
            INTEGER(C_INT), VALUE        :: user_block_size
            REAL(C_DOUBLE), VALUE        :: scale_S, scale_T
            INTEGER(C_INT), VALUE        :: comm_all, comm_sub
            INTEGER(C_INT), VALUE        :: color_sub
            TYPE(C_PTR), VALUE           :: gd_array_ptr
            INTEGER(C_INT), DIMENSION(*) :: gd_array_sizes
            INTEGER(C_INT), VALUE        :: gd_array_sizes_size
            INTEGER(C_INT), DIMENSION(*) :: gd_B_virtual_sizes
            INTEGER(C_INT), VALUE        :: gd_B_virtual_sizes_size
            REAL(C_DOUBLE), DIMENSION(*) :: eigenval_j, eigenval_i
            INTEGER(C_INT), VALUE        :: homo_j
            INTEGER(C_INT), VALUE        :: homo_i
            INTEGER(C_INT), VALUE        :: nmo
            INTEGER(C_INT), VALUE        :: dimen_RI
            LOGICAL(C_BOOL), VALUE       :: calc_forces_c
            INTEGER(C_INT), VALUE        :: unit_nr
            INTEGER(C_INT), VALUE        :: my_B_size_j
            INTEGER(C_INT), VALUE        :: my_B_size_i
            INTEGER(C_INT), VALUE        :: my_B_virtual_start_j
            INTEGER(C_INT), VALUE        :: my_B_virtual_end_j
            INTEGER(C_INT), VALUE        :: my_B_virtual_start_i
            INTEGER(C_INT), VALUE        :: my_B_virtual_end_i
            INTEGER(C_INT), VALUE        :: aux_start
            INTEGER(C_INT), VALUE        :: aux_size
            INTEGER(C_INT), VALUE        :: maxsize_gd_array
            INTEGER(C_INT), VALUE        :: maxsize_gd_B_virtual
            INTEGER(C_INT), VALUE        :: maxval_gd_B_virtual
            INTEGER(C_INT), DIMENSION(*) :: rec_B_virtual_start_j
            INTEGER(C_INT), DIMENSION(*) :: rec_B_virtual_start_i
            INTEGER(C_INT), DIMENSION(*) :: rec_B_virtual_end_j
            INTEGER(C_INT), DIMENSION(*) :: rec_B_virtual_end_i
            INTEGER(C_INT), DIMENSION(*) :: rec_B_sizes_j
            INTEGER(C_INT), DIMENSION(*) :: rec_B_sizes_i
            INTEGER(C_INT), VALUE        :: num_pe
            INTEGER(C_INT), VALUE        :: my_group_L_start
            INTEGER(C_INT), VALUE        :: my_group_L_end
            INTEGER(C_INT), VALUE        :: my_group_L_size
            INTEGER(C_INT), VALUE        :: preferred_dgemm_lib
         END SUBROUTINE calc_ri_mp2_energy_c
      END INTERFACE

      ! MPI communicators
      comm_all_handle = comm_all%get_handle()
      comm_sub_handle = comm_sub%get_handle()

      nspins = SIZE(homo)

      ALLOCATE(gd_array_sizes(SIZE(gd_array%sizes)))
      gd_array_sizes = gd_array%sizes

      ALLOCATE(gd_B_virtual_sizes(nspins))

      DO ispin = 1, nspins
         CALL get_group_dist(gd_B_virtual(ispin), para_env_sub%mepos, &
                             my_B_virtual_start(ispin), my_B_virtual_end(ispin), my_B_size(ispin))
      END DO

      CALL get_group_dist(gd_array, color_sub, my_group_L_start, my_group_L_end, my_group_L_size)

      mp2_memory = mp2_env%mp2_memory
      scale_S = mp2_env%scale_S
      scale_T = mp2_env%scale_T
      user_block_size = mp2_env%ri_mp2%block_size
      calc_forces_c = calc_forces

      ! set sizes
      maxsize_gd_array = maxsize(gd_array)
      maxsize_gd_B_virtual = MAXVAL(gd_B_virtual_sizes)
      maxval_gd_B_virtual = maxsize_gd_B_virtual

      DO jspin = 1, nspins
         homo_j = homo(jspin)

         DO ispin = 1, jspin
            homo_i = homo(ispin)

            my_B_virtual_start_j = my_B_virtual_start(jspin) - 1
            my_B_virtual_start_i = my_B_virtual_start(ispin) - 1
            my_B_virtual_end_j = my_B_virtual_end(jspin) - 1
            my_B_virtual_end_i = my_B_virtual_end(ispin) - 1
            my_B_size_j = my_B_size(jspin)
            my_B_size_i = my_B_size(ispin)

            CALL calc_ri_mp2_energy_c( &
               E_cou, &
               E_ex, &
               E_s, &
               E_t, &
               BIb_C(:, :, jspin), &
               BIb_C(:, :, ispin), &
               mp2_memory, &
               user_block_size, &
               scale_S, &
               scale_T, &
               comm_all_handle, &
               comm_sub_handle, &
               color_sub, &
               C_LOC(gd_array), &
               gd_array_sizes, &
               SIZE(gd_array_sizes), &
               gd_B_virtual_sizes, &
               SIZE(gd_B_virtual_sizes), &
               eigenval(:, jspin), &
               eigenval(:, ispin), &
               homo_j, &
               homo_i, &
               nmo, &
               dimen_RI, &
               calc_forces_c, &
               unit_nr, &
               my_B_size_j, &
               my_B_size_i, &
               my_B_virtual_start_j, &
               my_B_virtual_end_j, &
               my_B_virtual_start_i, &
               my_B_virtual_end_i, &
               aux_start, &
               aux_size, &
               maxsize_gd_array, &
               maxsize_gd_B_virtual, &
               maxval_gd_B_virtual, &
               rec_B_virtual_start_j, &
               rec_B_virtual_start_i, &
               rec_B_virtual_end_j, &
               rec_B_virtual_end_i, &
               rec_B_sizes_j, &
               rec_B_sizes_i, &
               comm_all%num_pe, &
               my_group_L_start, &
               my_group_L_end, &
               my_group_L_size, &
               preferred_dgemm_lib)
         END DO
      END DO

      DEALLOCATE(gd_array_sizes)
      DEALLOCATE(gd_B_virtual_sizes)

   END SUBROUTINE calc_ri_mp2_energy
END MODULE mp2_gpu_api
