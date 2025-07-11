!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2023 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Routines to calculate MP2 energy using GPW method
!> \par History
!>      10.2011 created [Joost VandeVondele and Mauro Del Ben]
! **************************************************************************************************
MODULE mp2_gpw_method
   USE atomic_kind_types,               ONLY: atomic_kind_type
   USE cell_types,                      ONLY: cell_type
   USE cp_blacs_env,                    ONLY: cp_blacs_env_type
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_operations,             ONLY: copy_dbcsr_to_fm,&
                                              cp_dbcsr_m_by_n_from_template
   USE cp_fm_struct,                    ONLY: cp_fm_struct_create,&
                                              cp_fm_struct_release,&
                                              cp_fm_struct_type
   USE cp_fm_types,                     ONLY: cp_fm_create,&
                                              cp_fm_get_info,&
                                              cp_fm_release,&
                                              cp_fm_type
   USE dbcsr_api,                       ONLY: &
        dbcsr_create, dbcsr_get_info, dbcsr_iterator_blocks_left, dbcsr_iterator_next_block, &
        dbcsr_iterator_start, dbcsr_iterator_stop, dbcsr_iterator_type, dbcsr_multiply, &
        dbcsr_p_type, dbcsr_release, dbcsr_set, dbcsr_type, dbcsr_type_no_symmetry
   USE group_dist_types,                ONLY: create_group_dist,&
                                              get_group_dist,&
                                              group_dist_d1_type,&
                                              release_group_dist
   USE kinds,                           ONLY: dp,&
                                              int_8
   USE machine,                         ONLY: m_memory
   USE message_passing,                 ONLY: mp_comm_type,&
                                              mp_para_env_type
   USE mp2_eri_gpw,                     ONLY: calc_potential_gpw,&
                                              cleanup_gpw,&
                                              prepare_gpw
   USE particle_types,                  ONLY: particle_type
   USE pw_env_types,                    ONLY: pw_env_type
   USE pw_methods,                      ONLY: pw_multiply,&
                                              pw_transfer,&
                                              pw_zero
   USE pw_poisson_types,                ONLY: pw_poisson_type
   USE pw_pool_types,                   ONLY: pw_pool_create_pw,&
                                              pw_pool_type
   USE pw_types,                        ONLY: REALDATA3D,&
                                              REALSPACE,&
                                              pw_release,&
                                              pw_type
   USE qs_collocate_density,            ONLY: calculate_wavefunction
   USE qs_environment_types,            ONLY: qs_environment_type
   USE qs_integrate_potential,          ONLY: integrate_v_rspace
   USE qs_kind_types,                   ONLY: qs_kind_type
   USE qs_neighbor_list_types,          ONLY: neighbor_list_set_p_type
   USE task_list_types,                 ONLY: task_list_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'mp2_gpw_method'

   PUBLIC :: mp2_gpw_compute

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param Emp2 ...
!> \param Emp2_Cou ...
!> \param Emp2_EX ...
!> \param qs_env ...
!> \param para_env ...
!> \param para_env_sub ...
!> \param color_sub ...
!> \param cell ...
!> \param particle_set ...
!> \param atomic_kind_set ...
!> \param qs_kind_set ...
!> \param mo_coeff ...
!> \param Eigenval ...
!> \param nmo ...
!> \param homo ...
!> \param mat_munu ...
!> \param sab_orb_sub ...
!> \param mo_coeff_o ...
!> \param mo_coeff_v ...
!> \param eps_filter ...
!> \param unit_nr ...
!> \param mp2_memory ...
!> \param calc_ex ...
!> \param blacs_env_sub ...
!> \param Emp2_AB ...
! **************************************************************************************************
   SUBROUTINE mp2_gpw_compute(Emp2, Emp2_Cou, Emp2_EX, qs_env, para_env, para_env_sub, color_sub, &
                              cell, particle_set, atomic_kind_set, qs_kind_set, mo_coeff, Eigenval, nmo, homo, &
                              mat_munu, sab_orb_sub, mo_coeff_o, mo_coeff_v, eps_filter, unit_nr, &
                              mp2_memory, calc_ex, blacs_env_sub, Emp2_AB)

      REAL(KIND=dp), INTENT(OUT)                         :: Emp2, Emp2_Cou, Emp2_EX
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(mp_para_env_type), POINTER                    :: para_env, para_env_sub
      INTEGER, INTENT(IN)                                :: color_sub
      TYPE(cell_type), POINTER                           :: cell
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(cp_fm_type), INTENT(IN)                       :: mo_coeff
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: Eigenval
      INTEGER, INTENT(IN)                                :: nmo
      INTEGER, DIMENSION(:), INTENT(IN)                  :: homo
      TYPE(dbcsr_p_type), INTENT(INOUT)                  :: mat_munu
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_orb_sub
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(IN)       :: mo_coeff_o, mo_coeff_v
      REAL(KIND=dp), INTENT(IN)                          :: eps_filter
      INTEGER, INTENT(IN)                                :: unit_nr
      REAL(KIND=dp), INTENT(IN)                          :: mp2_memory
      LOGICAL, INTENT(IN)                                :: calc_ex
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env_sub
      REAL(KIND=dp), INTENT(OUT), OPTIONAL               :: Emp2_AB

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'mp2_gpw_compute'

      INTEGER :: a, a_group_counter, b, b_global, b_group_counter, blk, col, col_offset, col_size, &
         color_counter, EX_end, EX_end_send, EX_start, EX_start_send, group_counter, handle, &
         handle2, handle3, i, i_counter, i_group_counter, index_proc_shift, ispin, j, max_b_size, &
         max_batch_size_A, max_batch_size_I, max_row_col_local, my_A_batch_size, my_A_virtual_end, &
         my_A_virtual_start, my_B_size, my_B_virtual_end, my_B_virtual_start, my_I_batch_size, &
         my_I_occupied_end, my_I_occupied_start, my_q_position, ncol_local, nfullcols_total, &
         nfullrows_total, ngroup, nrow_local, nspins, p, p_best, proc_receive
      INTEGER :: proc_send, q, q_best, row, row_offset, row_size, size_EX, size_EX_send, &
         sub_sub_color, wfn_calc, wfn_calc_best
      INTEGER(KIND=int_8)                                :: mem
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: vector_B_sizes, &
                                                            vector_batch_A_size_group, &
                                                            vector_batch_I_size_group
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: color_array, local_col_row_info
      INTEGER, DIMENSION(:), POINTER                     :: col_indices, row_indices
      INTEGER, DIMENSION(SIZE(homo))                     :: virtual
      LOGICAL                                            :: do_alpha_beta
      REAL(KIND=dp)                                      :: cutoff_old, mem_min, mem_real, mem_try, &
                                                            relative_cutoff_old, wfn_size
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: e_cutoff_old
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: my_Cocc, my_Cvirt
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: BIb_C, BIb_Ex, BIb_send
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: data_block
      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct
      TYPE(cp_fm_type)                                   :: fm_BIb_jb
      TYPE(dbcsr_iterator_type)                          :: iter
      TYPE(dbcsr_type), DIMENSION(SIZE(homo))            :: matrix_ia_jb, matrix_ia_jnu
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(group_dist_d1_type)                           :: gd_exchange
      TYPE(mp_comm_type)                                 :: comm_exchange
      TYPE(pw_env_type), POINTER                         :: pw_env_sub
      TYPE(pw_poisson_type), POINTER                     :: poisson_env
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(pw_type)                                      :: pot_g, psi_a, rho_g, rho_r
      TYPE(pw_type), ALLOCATABLE, DIMENSION(:)           :: psi_i
      TYPE(task_list_type), POINTER                      :: task_list_sub

      CALL timeset(routineN, handle)

      do_alpha_beta = .FALSE.
      IF (PRESENT(Emp2_AB)) do_alpha_beta = .TRUE.

      nspins = SIZE(homo)
      virtual = nmo - homo

      DO ispin = 1, nspins
         ! initialize and create the matrix (ia|jnu)
         CALL dbcsr_create(matrix_ia_jnu(ispin), template=mo_coeff_o(ispin)%matrix)

         ! Allocate Sparse matrices: (ia|jb)
    CALL cp_dbcsr_m_by_n_from_template(matrix_ia_jb(ispin), template=mo_coeff_o(ispin)%matrix, m=homo(ispin), n=nmo - homo(ispin), &
                                            sym=dbcsr_type_no_symmetry)

         ! set all to zero in such a way that the memory is actually allocated
         CALL dbcsr_set(matrix_ia_jnu(ispin), 0.0_dp)
         CALL dbcsr_set(matrix_ia_jb(ispin), 0.0_dp)
      END DO
      CALL dbcsr_set(mat_munu%matrix, 0.0_dp)

      IF (calc_ex) THEN
         ! create the analogous of matrix_ia_jb in fm type
         NULLIFY (fm_struct)
         CALL dbcsr_get_info(matrix_ia_jb(1), nfullrows_total=nfullrows_total, nfullcols_total=nfullcols_total)
         CALL cp_fm_struct_create(fm_struct, context=blacs_env_sub, nrow_global=nfullrows_total, &
                                  ncol_global=nfullcols_total, para_env=para_env_sub)
         CALL cp_fm_create(fm_BIb_jb, fm_struct, name="fm_BIb_jb")

         CALL copy_dbcsr_to_fm(matrix_ia_jb(1), fm_BIb_jb)
         CALL cp_fm_struct_release(fm_struct)

         CALL cp_fm_get_info(matrix=fm_BIb_jb, &
                             nrow_local=nrow_local, &
                             ncol_local=ncol_local, &
                             row_indices=row_indices, &
                             col_indices=col_indices)

         max_row_col_local = MAX(nrow_local, ncol_local)
         CALL para_env_sub%max(max_row_col_local)

         ALLOCATE (local_col_row_info(0:max_row_col_local, 2))
         local_col_row_info = 0
         ! 0,1 nrows
         local_col_row_info(0, 1) = nrow_local
         local_col_row_info(1:nrow_local, 1) = row_indices(1:nrow_local)
         ! 0,2 ncols
         local_col_row_info(0, 2) = ncol_local
         local_col_row_info(1:ncol_local, 2) = col_indices(1:ncol_local)
      END IF

      ! Get everything for GPW calculations
      CALL prepare_gpw(qs_env, dft_control, e_cutoff_old, cutoff_old, relative_cutoff_old, para_env_sub, pw_env_sub, &
                       auxbas_pw_pool, poisson_env, task_list_sub, rho_r, rho_g, pot_g, psi_a, sab_orb_sub)

      wfn_size = REAL(SIZE(rho_r%cr3d), KIND=dp)
      CALL para_env%max(wfn_size)

      ngroup = para_env%num_pe/para_env_sub%num_pe

      ! calculate the minimal memory required per MPI task (p=occupied division,q=virtual division)
      p_best = ngroup
      q_best = 1
      mem_min = HUGE(0)
      DO p = 1, ngroup
         q = ngroup/p
         IF (p*q .NE. ngroup) CYCLE

         CALL estimate_memory_usage(wfn_size, p, q, para_env_sub%num_pe, nmo, virtual(1), homo(1), calc_ex, mem_try)

         IF (mem_try <= mem_min) THEN
            mem_min = mem_try
            p_best = p
            q_best = q
         END IF
      END DO
      IF (unit_nr > 0) WRITE (unit_nr, '(T3,A,T69,F9.2,A3)') 'Minimum required memory per MPI process for MP2:', &
         mem_min, ' MB'

      CALL m_memory(mem)
      mem_real = (mem + 1024*1024 - 1)/(1024*1024)
      CALL para_env%min(mem_real)

      mem_real = mp2_memory - mem_real
      mem_real = MAX(mem_real, mem_min)
      IF (unit_nr > 0) WRITE (unit_nr, '(T3,A,T69,F9.2,A3)') 'Available memory per MPI process for MP2:', &
         mem_real, ' MB'

      wfn_calc_best = HUGE(wfn_calc_best)
      DO p = 1, ngroup
         q = ngroup/p
         IF (p*q .NE. ngroup) CYCLE

         CALL estimate_memory_usage(wfn_size, p, q, para_env_sub%num_pe, nmo, virtual(1), homo(1), calc_ex, mem_try)

         IF (mem_try > mem_real) CYCLE
         wfn_calc = ((homo(1) + p - 1)/p) + ((virtual(1) + q - 1)/q)
         IF (wfn_calc < wfn_calc_best) THEN
            wfn_calc_best = wfn_calc
            p_best = p
            q_best = q
         END IF
      END DO

      max_batch_size_I = (homo(1) + p_best - 1)/p_best
      max_batch_size_A = (virtual(1) + q_best - 1)/q_best

      IF (unit_nr > 0) THEN
         WRITE (UNIT=unit_nr, FMT="(T3,A,T77,i4)") &
            "MP2_GPW| max. batch size for the occupied states:", max_batch_size_I
         WRITE (UNIT=unit_nr, FMT="(T3,A,T77,i4)") &
            "MP2_GPW| max. batch size for the virtual states:", max_batch_size_A
      END IF

      CALL get_vector_batch(vector_batch_I_size_group, p_best, max_batch_size_I, homo(1))
      CALL get_vector_batch(vector_batch_A_size_group, q_best, max_batch_size_A, virtual(1))

      !XXXXXXXXXXXXX inverse group distribution
      group_counter = 0
      a_group_counter = 0
      my_A_virtual_start = 1
      DO j = 0, q_best - 1
         my_I_occupied_start = 1
         i_group_counter = 0
         DO i = 0, p_best - 1
            group_counter = group_counter + 1
            IF (color_sub == group_counter - 1) EXIT
            my_I_occupied_start = my_I_occupied_start + vector_batch_I_size_group(i)
            i_group_counter = i_group_counter + 1
         END DO
         my_q_position = j
         IF (color_sub == group_counter - 1) EXIT
         my_A_virtual_start = my_A_virtual_start + vector_batch_A_size_group(j)
         a_group_counter = a_group_counter + 1
      END DO
      !XXXXXXXXXXXXX inverse group distribution

      my_I_occupied_end = my_I_occupied_start + vector_batch_I_size_group(i_group_counter) - 1
      my_I_batch_size = vector_batch_I_size_group(i_group_counter)
      my_A_virtual_end = my_A_virtual_start + vector_batch_A_size_group(a_group_counter) - 1
      my_A_batch_size = vector_batch_A_size_group(a_group_counter)

      DEALLOCATE (vector_batch_I_size_group)
      DEALLOCATE (vector_batch_A_size_group)

      ! replicate on a local array on proc 0 the occupied and virtual wavevectior
      ! needed for the calculation of the WF's by calculate_wavefunction
      ! (external vector)
      CALL grep_occ_virt_wavefunc(para_env_sub, nmo, &
                                  my_I_occupied_start, my_I_occupied_end, my_I_batch_size, &
                                  my_A_virtual_start, my_A_virtual_end, my_A_batch_size, &
                                  mo_coeff_o(1)%matrix, mo_coeff_v(1)%matrix, my_Cocc, my_Cvirt)

      ! divide the b states in the sub_group in such a way to create
      ! b_start and b_end for each proc inside the sub_group
      max_b_size = (virtual(1) + para_env_sub%num_pe - 1)/para_env_sub%num_pe
      CALL get_vector_batch(vector_B_sizes, para_env_sub%num_pe, max_b_size, virtual(1))

      ! now give to each proc its b_start and b_end
      b_group_counter = 0
      my_B_virtual_start = 1
      DO j = 0, para_env_sub%num_pe - 1
         b_group_counter = b_group_counter + 1
         IF (b_group_counter - 1 == para_env_sub%mepos) EXIT
         my_B_virtual_start = my_B_virtual_start + vector_B_sizes(j)
      END DO
      my_B_virtual_end = my_B_virtual_start + vector_B_sizes(para_env_sub%mepos) - 1
      my_B_size = vector_B_sizes(para_env_sub%mepos)

      DEALLOCATE (vector_B_sizes)

      ! create an array containing a different "color" for each pair of
      ! A_start and B_start, communication will take place only among
      ! those proc that have the same A_start and B_start
      ALLOCATE (color_array(0:para_env_sub%num_pe - 1, 0:q_best - 1))
      color_array = 0
      color_counter = 0
      DO j = 0, q_best - 1
         DO i = 0, para_env_sub%num_pe - 1
            color_counter = color_counter + 1
            color_array(i, j) = color_counter
         END DO
      END DO
      sub_sub_color = color_array(para_env_sub%mepos, my_q_position)

      DEALLOCATE (color_array)

      ! now create a group that contains all the proc that have the same 2 virtual starting points
      ! in this way it is possible to sum the common integrals needed for the full MP2 energy
      CALL comm_exchange%from_split(para_env, sub_sub_color)

      ! create an array containing the information for communication
      CALL create_group_dist(gd_exchange, my_I_occupied_start, my_I_occupied_end, my_I_batch_size, comm_exchange)

      ALLOCATE (psi_i(my_I_occupied_start:my_I_occupied_end))
      DO i = my_I_occupied_start, my_I_occupied_end
         CALL pw_pool_create_pw(auxbas_pw_pool, psi_i(i), &
                                use_data=REALDATA3D, &
                                in_space=REALSPACE)
         CALL calculate_wavefunction(mo_coeff, i, psi_i(i), rho_g, atomic_kind_set, &
                                     qs_kind_set, cell, dft_control, particle_set, &
                                     pw_env_sub, external_vector=my_Cocc(:, i - my_I_occupied_start + 1))
      END DO

      Emp2 = 0.0_dp
      Emp2_Cou = 0.0_dp
      Emp2_EX = 0.0_dp
      IF (do_alpha_beta) Emp2_AB = 0.0_dp
      IF (calc_ex) THEN
         ALLOCATE (BIb_C(my_B_size, homo(1), my_I_batch_size))
      END IF

      CALL timeset(routineN//"_loop", handle2)
      DO a = homo(1) + my_A_virtual_start, homo(1) + my_A_virtual_end

         IF (calc_ex) BIb_C = 0.0_dp

         ! psi_a
         CALL calculate_wavefunction(mo_coeff, a, psi_a, rho_g, atomic_kind_set, &
                                     qs_kind_set, cell, dft_control, particle_set, &
                                     pw_env_sub, external_vector=my_Cvirt(:, a - (homo(1) + my_A_virtual_start) + 1))
         i_counter = 0
         DO i = my_I_occupied_start, my_I_occupied_end
            i_counter = i_counter + 1

            ! potential
            CALL pw_zero(rho_r)
            CALL pw_multiply(rho_r, psi_i(i), psi_a)
            CALL pw_transfer(rho_r, rho_g)
            CALL calc_potential_gpw(rho_r, rho_g, poisson_env, pot_g, qs_env%mp2_env%potential_parameter)

            ! and finally (ia|munu)
            CALL timeset(routineN//"_int", handle3)
            CALL dbcsr_set(mat_munu%matrix, 0.0_dp)
            CALL integrate_v_rspace(rho_r, hmat=mat_munu, qs_env=qs_env, &
                                    calculate_forces=.FALSE., compute_tau=.FALSE., gapw=.FALSE., &
                                    pw_env_external=pw_env_sub, task_list_external=task_list_sub)
            CALL timestop(handle3)

            ! multiply and goooooooo ...
            CALL timeset(routineN//"_mult_o", handle3)
            DO ispin = 1, nspins
               CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu%matrix, mo_coeff_o(ispin)%matrix, &
                                   0.0_dp, matrix_ia_jnu(ispin), filter_eps=eps_filter)
            END DO
            CALL timestop(handle3)
            CALL timeset(routineN//"_mult_v", handle3)
            DO ispin = 1, nspins
               CALL dbcsr_multiply("T", "N", 1.0_dp, matrix_ia_jnu(ispin), mo_coeff_v(ispin)%matrix, &
                                   0.0_dp, matrix_ia_jb(ispin), filter_eps=eps_filter)
            END DO
            CALL timestop(handle3)

            CALL timeset(routineN//"_E_Cou", handle3)
            CALL dbcsr_iterator_start(iter, matrix_ia_jb(1))
            DO WHILE (dbcsr_iterator_blocks_left(iter))
               CALL dbcsr_iterator_next_block(iter, row, col, data_block, blk, &
                                              row_size=row_size, col_size=col_size, &
                                              row_offset=row_offset, col_offset=col_offset)
               DO b = 1, col_size
               DO j = 1, row_size
                  ! Compute the coulomb MP2 energy
                  Emp2_Cou = Emp2_Cou - 2.0_dp*data_block(j, b)**2/ &
                     (Eigenval(a, 1) + Eigenval(homo(1) + col_offset + b - 1, 1) - Eigenval(i, 1) - Eigenval(row_offset + j - 1, 1))
               END DO
               END DO
            END DO
            CALL dbcsr_iterator_stop(iter)
            IF (do_alpha_beta) THEN
               ! Compute the coulomb only= SO = MP2 alpha-beta  MP2 energy component
               CALL dbcsr_iterator_start(iter, matrix_ia_jb(2))
               DO WHILE (dbcsr_iterator_blocks_left(iter))
                  CALL dbcsr_iterator_next_block(iter, row, col, data_block, blk, &
                                                 row_size=row_size, col_size=col_size, &
                                                 row_offset=row_offset, col_offset=col_offset)
                  DO b = 1, col_size
                  DO j = 1, row_size
                     ! Compute the coulomb MP2 energy alpha beta case
                     Emp2_AB = Emp2_AB - data_block(j, b)**2/ &
                     (Eigenval(a, 1) + Eigenval(homo(2) + col_offset + b - 1, 2) - Eigenval(i, 1) - Eigenval(row_offset + j - 1, 2))
                  END DO
                  END DO
               END DO
               CALL dbcsr_iterator_stop(iter)
            END IF
            CALL timestop(handle3)

            ! now collect my local data from all the other members of the group
            ! b_start, b_end
            IF (calc_ex) THEN
               CALL timeset(routineN//"_E_Ex_1", handle3)
               CALL copy_dbcsr_to_fm(matrix_ia_jb(1), fm_BIb_jb)
               CALL grep_my_integrals(para_env_sub, fm_BIb_jb, BIb_C(1:my_B_size, 1:homo(1), i_counter), max_row_col_local, &
                                      local_col_row_info, my_B_virtual_end, my_B_virtual_start)
               CALL timestop(handle3)
            END IF

         END DO

         IF (calc_ex) THEN

            ASSOCIATE (mepos_in_EX_group => comm_exchange%mepos, size_of_exchange_group => comm_exchange%num_pe)
               CALL timeset(routineN//"_E_Ex_2", handle3)
               ! calculate the contribution to MP2 energy for my local data
               DO i = 1, my_I_batch_size
                  DO j = my_I_occupied_start, my_I_occupied_end
                     DO b = 1, my_B_size
                        b_global = b - 1 + my_B_virtual_start
                        Emp2_EX = Emp2_EX + BIb_C(b, j, i)*BIb_C(b, i + my_I_occupied_start - 1, j - my_I_occupied_start + 1) &
                     /(Eigenval(a, 1) + Eigenval(homo(1) + b_global, 1) - Eigenval(i + my_I_occupied_start - 1, 1) - Eigenval(j, 1))
                     END DO
                  END DO
               END DO

               ! start communicating and collecting exchange contributions from
               ! other processes in my exchange group
               DO index_proc_shift = 1, size_of_exchange_group - 1
                  proc_send = MODULO(mepos_in_EX_group + index_proc_shift, size_of_exchange_group)
                  proc_receive = MODULO(mepos_in_EX_group - index_proc_shift, size_of_exchange_group)

                  CALL get_group_dist(gd_exchange, proc_receive, EX_start, EX_end, size_EX)

                  ALLOCATE (BIb_EX(my_B_size, my_I_batch_size, size_EX))
                  BIb_EX = 0.0_dp

                  CALL get_group_dist(gd_exchange, proc_send, EX_start_send, EX_end_send, size_EX_send)

                  ALLOCATE (BIb_send(my_B_size, size_EX_send, my_I_batch_size))
                  BIb_send(1:my_B_size, 1:size_EX_send, 1:my_I_batch_size) = &
                     BIb_C(1:my_B_size, EX_start_send:EX_end_send, 1:my_I_batch_size)

                  ! send and receive the exchange array
                  CALL comm_exchange%sendrecv(BIb_send, proc_send, BIb_EX, proc_receive)

                  DO i = 1, my_I_batch_size
                     DO j = 1, size_EX
                        DO b = 1, my_B_size
                           b_global = b - 1 + my_B_virtual_start
                           Emp2_EX = Emp2_EX + BIb_C(b, j + EX_start - 1, i)*BIb_EX(b, i, j) &
                                     /(Eigenval(a, 1) + Eigenval(homo(1) + b_global, 1) - Eigenval(i + my_I_occupied_start - 1, 1) &
                                       - Eigenval(j + EX_start - 1, 1))
                        END DO
                     END DO
                  END DO

                  DEALLOCATE (BIb_EX)
                  DEALLOCATE (BIb_send)

               END DO
               CALL timestop(handle3)
            END ASSOCIATE
         END IF

      END DO
      CALL timestop(handle2)

      CALL para_env%sum(Emp2_Cou)
      CALL para_env%sum(Emp2_EX)
      Emp2 = Emp2_Cou + Emp2_EX
      IF (do_alpha_beta) CALL para_env%sum(Emp2_AB)

      DEALLOCATE (my_Cocc)
      DEALLOCATE (my_Cvirt)

      IF (calc_ex) THEN
         CALL cp_fm_release(fm_BIb_jb)
         DEALLOCATE (local_col_row_info)
         DEALLOCATE (BIb_C)
      END IF
      CALL release_group_dist(gd_exchange)

      CALL comm_exchange%free()

      DO ispin = 1, nspins
         CALL dbcsr_release(matrix_ia_jnu(ispin))
         CALL dbcsr_release(matrix_ia_jb(ispin))
      END DO

      DO i = my_I_occupied_start, my_I_occupied_end
         CALL pw_release(psi_i(i))
      END DO
      DEALLOCATE (psi_i)

      CALL cleanup_gpw(qs_env, e_cutoff_old, cutoff_old, relative_cutoff_old, para_env_sub, pw_env_sub, &
                       task_list_sub, auxbas_pw_pool, rho_r, rho_g, pot_g, psi_a)

      CALL timestop(handle)

   END SUBROUTINE mp2_gpw_compute

! **************************************************************************************************
!> \brief ...
!> \param wfn_size ...
!> \param p ...
!> \param q ...
!> \param num_w ...
!> \param nmo ...
!> \param virtual ...
!> \param homo ...
!> \param calc_ex ...
!> \param mem_try ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE estimate_memory_usage(wfn_size, p, q, num_w, nmo, virtual, homo, calc_ex, mem_try)
      REAL(KIND=dp), INTENT(IN)                          :: wfn_size
      INTEGER, INTENT(IN)                                :: p, q, num_w, nmo, virtual, homo
      LOGICAL, INTENT(IN)                                :: calc_ex
      REAL(KIND=dp), INTENT(OUT)                         :: mem_try

      mem_try = 0.0_dp
      ! integrals
      mem_try = mem_try + virtual*REAL(homo, KIND=dp)**2/(p*num_w)
      ! array for the coefficient matrix and wave vectors
      mem_try = mem_try + REAL(homo, KIND=dp)*nmo/p + &
                REAL(virtual, KIND=dp)*nmo/q + &
                2.0_dp*MAX(REAL(homo, KIND=dp)*nmo/p, REAL(virtual, KIND=dp)*nmo/q)
      ! temporary array for MO integrals and MO integrals to be exchanged
      IF (calc_ex) THEN
         mem_try = mem_try + 2.0_dp*MAX(virtual*REAL(homo, KIND=dp)*MIN(1, num_w - 1)/num_w, &
                                        virtual*REAL(homo, KIND=dp)**2/(p*p*num_w))
      ELSE
         mem_try = mem_try + 2.0_dp*virtual*REAL(homo, KIND=dp)
      END IF
      ! wfn
      mem_try = mem_try + ((homo + p - 1)/p)*wfn_size
      ! Mb
      mem_try = mem_try*8.0D+00/1024.0D+00**2

   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param vector_batch_I_size_group ...
!> \param p_best ...
!> \param max_batch_size_I ...
!> \param homo ...
! **************************************************************************************************
   PURE SUBROUTINE get_vector_batch(vector_batch_I_size_group, p_best, max_batch_size_I, homo)
      INTEGER, ALLOCATABLE, DIMENSION(:), INTENT(OUT)    :: vector_batch_I_size_group
      INTEGER, INTENT(IN)                                :: p_best, max_batch_size_I, homo

      INTEGER                                            :: i, one

      ALLOCATE (vector_batch_I_size_group(0:p_best - 1))

      vector_batch_I_size_group = max_batch_size_I
      IF (SUM(vector_batch_I_size_group) /= homo) THEN
         one = 1
         IF (SUM(vector_batch_I_size_group) > homo) one = -1
         i = -1
         DO
            i = i + 1
            vector_batch_I_size_group(i) = vector_batch_I_size_group(i) + one
            IF (SUM(vector_batch_I_size_group) == homo) EXIT
            IF (i == p_best - 1) i = -1
         END DO
      END IF

   END SUBROUTINE get_vector_batch

! **************************************************************************************************
!> \brief ...
!> \param para_env_sub ...
!> \param fm_BIb_jb ...
!> \param BIb_jb ...
!> \param max_row_col_local ...
!> \param local_col_row_info ...
!> \param my_B_virtual_end ...
!> \param my_B_virtual_start ...
! **************************************************************************************************
   SUBROUTINE grep_my_integrals(para_env_sub, fm_BIb_jb, BIb_jb, max_row_col_local, &
                                local_col_row_info, &
                                my_B_virtual_end, my_B_virtual_start)
      TYPE(mp_para_env_type), INTENT(IN)                 :: para_env_sub
      TYPE(cp_fm_type), INTENT(IN)                       :: fm_BIb_jb
      REAL(KIND=dp), DIMENSION(:, :), INTENT(OUT)        :: BIb_jb
      INTEGER, INTENT(IN)                                :: max_row_col_local
      INTEGER, ALLOCATABLE, DIMENSION(:, :), INTENT(IN)  :: local_col_row_info
      INTEGER, INTENT(IN)                                :: my_B_virtual_end, my_B_virtual_start

      INTEGER                                            :: i_global, iiB, j_global, jjB, ncol_rec, &
                                                            nrow_rec, proc_receive, proc_send, &
                                                            proc_shift
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: rec_col_row_info
      INTEGER, DIMENSION(:), POINTER                     :: col_indices_rec, row_indices_rec
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: local_BI, rec_BI

      ALLOCATE (rec_col_row_info(0:max_row_col_local, 2))

      rec_col_row_info(:, :) = local_col_row_info

      nrow_rec = rec_col_row_info(0, 1)
      ncol_rec = rec_col_row_info(0, 2)

      ALLOCATE (row_indices_rec(nrow_rec))
      row_indices_rec = rec_col_row_info(1:nrow_rec, 1)

      ALLOCATE (col_indices_rec(ncol_rec))
      col_indices_rec = rec_col_row_info(1:ncol_rec, 2)

      ! accumulate data on BIb_jb buffer starting from myself
      DO jjB = 1, ncol_rec
         j_global = col_indices_rec(jjB)
         IF (j_global >= my_B_virtual_start .AND. j_global <= my_B_virtual_end) THEN
            DO iiB = 1, nrow_rec
               i_global = row_indices_rec(iiB)
               BIb_jb(j_global - my_B_virtual_start + 1, i_global) = fm_BIb_jb%local_data(iiB, jjB)
            END DO
         END IF
      END DO

      DEALLOCATE (row_indices_rec)
      DEALLOCATE (col_indices_rec)

      IF (para_env_sub%num_pe > 1) THEN
         ALLOCATE (local_BI(nrow_rec, ncol_rec))
         local_BI(1:nrow_rec, 1:ncol_rec) = fm_BIb_jb%local_data(1:nrow_rec, 1:ncol_rec)

         DO proc_shift = 1, para_env_sub%num_pe - 1
            proc_send = MODULO(para_env_sub%mepos + proc_shift, para_env_sub%num_pe)
            proc_receive = MODULO(para_env_sub%mepos - proc_shift, para_env_sub%num_pe)

            ! first exchange information on the local data
            rec_col_row_info = 0
            CALL para_env_sub%sendrecv(local_col_row_info, proc_send, rec_col_row_info, proc_receive)
            nrow_rec = rec_col_row_info(0, 1)
            ncol_rec = rec_col_row_info(0, 2)

            ALLOCATE (row_indices_rec(nrow_rec))
            row_indices_rec = rec_col_row_info(1:nrow_rec, 1)

            ALLOCATE (col_indices_rec(ncol_rec))
            col_indices_rec = rec_col_row_info(1:ncol_rec, 2)

            ALLOCATE (rec_BI(nrow_rec, ncol_rec))
            rec_BI = 0.0_dp

            ! then send and receive the real data
            CALL para_env_sub%sendrecv(local_BI, proc_send, rec_BI, proc_receive)

            ! accumulate the received data on BIb_jb buffer
            DO jjB = 1, ncol_rec
               j_global = col_indices_rec(jjB)
               IF (j_global >= my_B_virtual_start .AND. j_global <= my_B_virtual_end) THEN
                  DO iiB = 1, nrow_rec
                     i_global = row_indices_rec(iiB)
                     BIb_jb(j_global - my_B_virtual_start + 1, i_global) = rec_BI(iiB, jjB)
                  END DO
               END IF
            END DO

            DEALLOCATE (col_indices_rec)
            DEALLOCATE (row_indices_rec)
            DEALLOCATE (rec_BI)
         END DO

         DEALLOCATE (local_BI)
      END IF

      DEALLOCATE (rec_col_row_info)

   END SUBROUTINE grep_my_integrals

! **************************************************************************************************
!> \brief ...
!> \param para_env_sub ...
!> \param dimen ...
!> \param my_I_occupied_start ...
!> \param my_I_occupied_end ...
!> \param my_I_batch_size ...
!> \param my_A_virtual_start ...
!> \param my_A_virtual_end ...
!> \param my_A_batch_size ...
!> \param mo_coeff_o ...
!> \param mo_coeff_v ...
!> \param my_Cocc ...
!> \param my_Cvirt ...
! **************************************************************************************************
   SUBROUTINE grep_occ_virt_wavefunc(para_env_sub, dimen, &
                                     my_I_occupied_start, my_I_occupied_end, my_I_batch_size, &
                                     my_A_virtual_start, my_A_virtual_end, my_A_batch_size, &
                                     mo_coeff_o, mo_coeff_v, my_Cocc, my_Cvirt)

      TYPE(mp_para_env_type), INTENT(IN)                 :: para_env_sub
      INTEGER, INTENT(IN) :: dimen, my_I_occupied_start, my_I_occupied_end, my_I_batch_size, &
         my_A_virtual_start, my_A_virtual_end, my_A_batch_size
      TYPE(dbcsr_type), POINTER                          :: mo_coeff_o, mo_coeff_v
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :), &
         INTENT(OUT)                                     :: my_Cocc, my_Cvirt

      CHARACTER(LEN=*), PARAMETER :: routineN = 'grep_occ_virt_wavefunc'

      INTEGER                                            :: blk, col, col_offset, col_size, handle, &
                                                            i, i_global, j, j_global, row, &
                                                            row_offset, row_size
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: data_block
      TYPE(dbcsr_iterator_type)                          :: iter

      CALL timeset(routineN, handle)

      ALLOCATE (my_Cocc(dimen, my_I_batch_size))
      my_Cocc = 0.0_dp

      ALLOCATE (my_Cvirt(dimen, my_A_batch_size))
      my_Cvirt = 0.0_dp

      ! accumulate data from mo_coeff_o into Cocc
      CALL dbcsr_iterator_start(iter, mo_coeff_o)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, data_block, blk, &
                                        row_size=row_size, col_size=col_size, &
                                        row_offset=row_offset, col_offset=col_offset)
         DO j = 1, col_size
            j_global = col_offset + j - 1
            IF (j_global >= my_I_occupied_start .AND. j_global <= my_I_occupied_end) THEN
               DO i = 1, row_size
                  i_global = row_offset + i - 1
                  my_Cocc(i_global, j_global - my_I_occupied_start + 1) = data_block(i, j)
               END DO
            END IF
         END DO
      END DO
      CALL dbcsr_iterator_stop(iter)

      CALL para_env_sub%sum(my_Cocc)

      ! accumulate data from mo_coeff_o into Cocc
      CALL dbcsr_iterator_start(iter, mo_coeff_v)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, data_block, blk, &
                                        row_size=row_size, col_size=col_size, &
                                        row_offset=row_offset, col_offset=col_offset)
         DO j = 1, col_size
            j_global = col_offset + j - 1
            IF (j_global >= my_A_virtual_start .AND. j_global <= my_A_virtual_end) THEN
               DO i = 1, row_size
                  i_global = row_offset + i - 1
                  my_Cvirt(i_global, j_global - my_A_virtual_start + 1) = data_block(i, j)
               END DO
            END IF
         END DO
      END DO
      CALL dbcsr_iterator_stop(iter)

      CALL para_env_sub%sum(my_Cvirt)

      CALL timestop(handle)

   END SUBROUTINE grep_occ_virt_wavefunc

END MODULE mp2_gpw_method
