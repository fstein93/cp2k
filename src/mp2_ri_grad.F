!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Routines to calculate gradients of RI-GPW-MP2 energy using pw
!> \par History
!>      10.2013 created [Mauro Del Ben]
! **************************************************************************************************
MODULE mp2_ri_grad
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind_set
   USE cell_types,                      ONLY: cell_type
   USE cp_blacs_env,                    ONLY: cp_blacs_env_type
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_api,                    ONLY: &
        dbcsr_add, dbcsr_copy, dbcsr_create, dbcsr_get_info, dbcsr_iterator_blocks_left, &
        dbcsr_iterator_next_block, dbcsr_iterator_start, dbcsr_iterator_stop, dbcsr_iterator_type, &
        dbcsr_multiply, dbcsr_p_type, dbcsr_release, dbcsr_set, dbcsr_transposed, dbcsr_type, &
        dbcsr_type_no_symmetry, dbcsr_type_symmetric
   USE cp_dbcsr_operations,             ONLY: copy_dbcsr_to_fm,&
                                              dbcsr_deallocate_matrix_set
   USE cp_eri_mme_interface,            ONLY: cp_eri_mme_param
   USE cp_fm_basic_linalg,              ONLY: cp_fm_geadd
   USE cp_fm_struct,                    ONLY: cp_fm_struct_create,&
                                              cp_fm_struct_release,&
                                              cp_fm_struct_type
   USE cp_fm_types,                     ONLY: cp_fm_create,&
                                              cp_fm_get_info,&
                                              cp_fm_release,&
                                              cp_fm_set_all,&
                                              cp_fm_type
   USE input_constants,                 ONLY: do_eri_gpw,&
                                              do_eri_mme,&
                                              ri_mp2_laplace,&
                                              ri_mp2_method_gpw,&
                                              ri_rpa_method_gpw
   USE kinds,                           ONLY: dp
   USE libint_2c_3c,                    ONLY: compare_potential_types
   USE message_passing,                 ONLY: mp_para_env_release,&
                                              mp_para_env_type,&
                                              mp_request_null,&
                                              mp_request_type,&
                                              mp_waitall
   USE mp2_eri,                         ONLY: mp2_eri_2c_integrate,&
                                              mp2_eri_3c_integrate,&
                                              mp2_eri_deallocate_forces,&
                                              mp2_eri_force
   USE mp2_eri_gpw,                     ONLY: cleanup_gpw,&
                                              integrate_potential_forces_2c,&
                                              integrate_potential_forces_3c_1c,&
                                              integrate_potential_forces_3c_2c,&
                                              prepare_gpw
   USE mp2_types,                       ONLY: integ_mat_buffer_type,&
                                              integ_mat_buffer_type_2D,&
                                              mp2_type
   USE parallel_gemm_api,               ONLY: parallel_gemm
   USE particle_types,                  ONLY: particle_type
   USE pw_env_types,                    ONLY: pw_env_type
   USE pw_poisson_types,                ONLY: pw_poisson_type
   USE pw_pool_types,                   ONLY: pw_pool_type
   USE pw_types,                        ONLY: pw_c1d_gs_type,&
                                              pw_r3d_rs_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_force_types,                  ONLY: allocate_qs_force,&
                                              qs_force_type,&
                                              zero_qs_force
   USE qs_kind_types,                   ONLY: qs_kind_type
   USE qs_ks_types,                     ONLY: qs_ks_env_type
   USE qs_neighbor_list_types,          ONLY: neighbor_list_set_p_type
   USE task_list_types,                 ONLY: task_list_type
   USE util,                            ONLY: get_limit
   USE virial_types,                    ONLY: virial_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'mp2_ri_grad'

   PUBLIC :: calc_ri_mp2_nonsep

CONTAINS

! **************************************************************************************************
!> \brief Calculate the non-separable part of the gradients and update the
!>        Lagrangian
!> \param qs_env ...
!> \param mp2_env ...
!> \param para_env ...
!> \param para_env_sub ...
!> \param cell ...
!> \param particle_set ...
!> \param atomic_kind_set ...
!> \param qs_kind_set ...
!> \param mo_coeff ...
!> \param dimen_RI ...
!> \param Eigenval ...
!> \param occupation ...
!> \param my_group_L_start ...
!> \param my_group_L_end ...
!> \param my_group_L_size ...
!> \param sab_orb_sub ...
!> \param mat_munu ...
!> \param blacs_env_sub ...
!> \author Mauro Del Ben
! **************************************************************************************************
   SUBROUTINE calc_ri_mp2_nonsep(qs_env, mp2_env, para_env, para_env_sub, cell, particle_set, &
                                 atomic_kind_set, qs_kind_set, mo_coeff, dimen_RI, Eigenval, occupation, &
                                 my_group_L_start, my_group_L_end, my_group_L_size, sab_orb_sub, mat_munu, &
                                 blacs_env_sub)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(mp2_type)                                     :: mp2_env
      TYPE(mp_para_env_type), POINTER                    :: para_env, para_env_sub
      TYPE(cell_type), POINTER                           :: cell
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(cp_fm_type), DIMENSION(:), INTENT(IN)         :: mo_coeff
      INTEGER, INTENT(IN)                                :: dimen_RI
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: Eigenval, occupation
      INTEGER, INTENT(IN)                                :: my_group_L_start, my_group_L_end, &
                                                            my_group_L_size
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_orb_sub
      TYPE(dbcsr_p_type), INTENT(INOUT)                  :: mat_munu
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env_sub

      CHARACTER(LEN=*), PARAMETER :: routineN = 'calc_ri_mp2_nonsep'

      INTEGER                                            :: eri_method, handle, handle2, i, ikind, &
                                                            ispin, itmp(2), L_counter, LLL, &
                                                            my_P_end, my_P_size, my_P_start, nao, &
                                                            nmo, nspins
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: atom_of_kind, homo, kind_of, &
                                                            natom_of_kind, virtual
      LOGICAL                                            :: do_smearing, use_virial
      REAL(KIND=dp)                                      :: cutoff_old, eps_filter, factor_2c, &
                                                            relative_cutoff_old
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: e_cutoff_old
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: G_PQ_local, G_PQ_local_2
      REAL(KIND=dp), DIMENSION(3, 3)                     :: h_stress, pv_virial
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: I_tmp2
      TYPE(cp_eri_mme_param), POINTER                    :: eri_param
      TYPE(dbcsr_p_type)                                 :: matrix_P_munu
      TYPE(dbcsr_p_type), ALLOCATABLE, DIMENSION(:)      :: mo_coeff_o, mo_coeff_v
      TYPE(dbcsr_p_type), ALLOCATABLE, DIMENSION(:, :)   :: G_P_ia
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: mat_munu_local, matrix_P_munu_local
      TYPE(dbcsr_type)                                   :: matrix_P_munu_nosym
      TYPE(dbcsr_type), ALLOCATABLE, DIMENSION(:) :: Lag2_mu_i_1, Lag2_nu_a_2, Lag_mu_i_1, &
         Lag_nu_a_2, LagFT2_mu_i_1, LagFT2_nu_a_2, LagFT_mu_i_1, LagFT_nu_a_2, matrix_P_inu
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(mp2_eri_force), ALLOCATABLE, DIMENSION(:)     :: force_2c, force_2c_RI, force_3c_aux, &
                                                            force_3c_orb_mu, force_3c_orb_nu
      TYPE(pw_c1d_gs_type)                               :: dvg(3), pot_g, rho_g, rho_g_copy
      TYPE(pw_env_type), POINTER                         :: pw_env_sub
      TYPE(pw_poisson_type), POINTER                     :: poisson_env
      TYPE(pw_pool_type), POINTER                        :: auxbas_pw_pool
      TYPE(pw_r3d_rs_type)                               :: psi_L, rho_r
      TYPE(qs_force_type), DIMENSION(:), POINTER         :: force, mp2_force
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(task_list_type), POINTER                      :: task_list_sub
      TYPE(virial_type), POINTER                         :: virial

      CALL timeset(routineN, handle)

      eri_method = mp2_env%eri_method
      eri_param => mp2_env%eri_mme_param

      nmo = SIZE(Eigenval, 1)
      nspins = SIZE(Eigenval, 2)

      ALLOCATE (virtual(nspins), homo(nspins))
      eps_filter = mp2_env%mp2_gpw%eps_filter
      ALLOCATE (mo_coeff_o(nspins), mo_coeff_v(nspins), G_P_ia(nspins, my_group_L_size))
      DO ispin = 1, nspins
         mo_coeff_o(ispin)%matrix => mp2_env%ri_grad%mo_coeff_o(ispin)%matrix
         mo_coeff_v(ispin)%matrix => mp2_env%ri_grad%mo_coeff_v(ispin)%matrix
         CALL dbcsr_get_info(mo_coeff_o(ispin)%matrix, &
                             nfullrows_total=nao, &
                             nfullcols_total=homo(ispin))
         CALL dbcsr_get_info(mo_coeff_v(ispin)%matrix, &
                             nfullrows_total=nao, &
                             nfullcols_total=virtual(ispin))
         DO LLL = 1, my_group_L_size
            G_P_ia(ispin, LLL)%matrix => mp2_env%ri_grad%G_P_ia(LLL, ispin)%matrix
         END DO
      END DO
      DEALLOCATE (mp2_env%ri_grad%G_P_ia)

      do_smearing = ANY(homo + virtual > nmo)

      itmp = get_limit(dimen_RI, para_env_sub%num_pe, para_env_sub%mepos)
      my_P_start = itmp(1)
      my_P_end = itmp(2)
      my_P_size = itmp(2) - itmp(1) + 1

      ALLOCATE (G_PQ_local(dimen_RI, my_group_L_size))
      G_PQ_local = 0.0_dp
      G_PQ_local(my_P_start:my_P_end, :) = mp2_env%ri_grad%Gamma_PQ
      DEALLOCATE (mp2_env%ri_grad%Gamma_PQ)
      G_PQ_local(my_P_start:my_P_end, :) = G_PQ_local(my_P_start:my_P_end, :)/REAL(nspins, dp)
      CALL para_env_sub%sum(G_PQ_local)
      IF (.NOT. compare_potential_types(mp2_env%ri_metric, mp2_env%potential_parameter)) THEN
         ALLOCATE (G_PQ_local_2(dimen_RI, my_group_L_size))
         G_PQ_local_2 = 0.0_dp
         G_PQ_local_2(my_P_start:my_P_end, :) = mp2_env%ri_grad%Gamma_PQ_2
         DEALLOCATE (mp2_env%ri_grad%Gamma_PQ_2)
         G_PQ_local_2(my_P_start:my_P_end, :) = G_PQ_local_2(my_P_start:my_P_end, :)/REAL(nspins, dp)
         CALL para_env_sub%sum(G_PQ_local_2)
      END IF

      ! create matrix holding the back transformation (G_P_inu)
      ALLOCATE (matrix_P_inu(nspins))
      DO ispin = 1, nspins
         CALL dbcsr_create(matrix_P_inu(ispin), template=mo_coeff_o(ispin)%matrix)
      END DO

      ! non symmetric matrix
      CALL dbcsr_create(matrix_P_munu_nosym, template=mat_munu%matrix, &
                        matrix_type=dbcsr_type_no_symmetry)

      ! create Lagrangian matrices in mixed AO/MO formalism
      ALLOCATE (Lag_mu_i_1(nspins))
      DO ispin = 1, nspins
         CALL dbcsr_create(Lag_mu_i_1(ispin), template=mo_coeff_o(ispin)%matrix)
         CALL dbcsr_set(Lag_mu_i_1(ispin), 0.0_dp)
      END DO

      ALLOCATE (Lag_nu_a_2(nspins))
      DO ispin = 1, nspins
         CALL dbcsr_create(Lag_nu_a_2(ispin), template=mo_coeff_v(ispin)%matrix)
         CALL dbcsr_set(Lag_nu_a_2(ispin), 0.0_dp)
      END DO

      IF (do_smearing) THEN
         ALLOCATE (LagFT_mu_i_1(nspins))
         DO ispin = 1, nspins
            CALL dbcsr_create(LagFT_mu_i_1(ispin), template=mo_coeff_o(ispin)%matrix)
            CALL dbcsr_set(LagFT_mu_i_1(ispin), 0.0_dp)
         END DO

         ALLOCATE (LagFT_nu_a_2(nspins))
         DO ispin = 1, nspins
            CALL dbcsr_create(LagFT_nu_a_2(ispin), template=mo_coeff_v(ispin)%matrix)
            CALL dbcsr_set(LagFT_nu_a_2(ispin), 0.0_dp)
         END DO
         ALLOCATE (Lag2_mu_i_1(nspins))
         DO ispin = 1, nspins
            CALL dbcsr_create(Lag2_mu_i_1(ispin), template=mo_coeff_o(ispin)%matrix)
            CALL dbcsr_set(Lag2_mu_i_1(ispin), 0.0_dp)
         END DO

         ALLOCATE (Lag2_nu_a_2(nspins))
         DO ispin = 1, nspins
            CALL dbcsr_create(Lag2_nu_a_2(ispin), template=mo_coeff_v(ispin)%matrix)
            CALL dbcsr_set(Lag2_nu_a_2(ispin), 0.0_dp)
         END DO

         ALLOCATE (LagFT2_mu_i_1(nspins))
         DO ispin = 1, nspins
            CALL dbcsr_create(LagFT2_mu_i_1(ispin), template=mo_coeff_o(ispin)%matrix)
            CALL dbcsr_set(LagFT2_mu_i_1(ispin), 0.0_dp)
         END DO

         ALLOCATE (LagFT2_nu_a_2(nspins))
         DO ispin = 1, nspins
            CALL dbcsr_create(LagFT2_nu_a_2(ispin), template=mo_coeff_v(ispin)%matrix)
            CALL dbcsr_set(LagFT2_nu_a_2(ispin), 0.0_dp)
         END DO
      END IF

      ! get forces
      NULLIFY (force, virial)
      CALL get_qs_env(qs_env=qs_env, force=force, virial=virial)

      ! check if we want to calculate the virial
      use_virial = virial%pv_availability .AND. (.NOT. virial%pv_numer)

      CALL get_atomic_kind_set(atomic_kind_set=atomic_kind_set, natom_of_kind=natom_of_kind)
      NULLIFY (mp2_force)
      CALL allocate_qs_force(mp2_force, natom_of_kind)
      DEALLOCATE (natom_of_kind)
      CALL zero_qs_force(mp2_force)
      mp2_env%ri_grad%mp2_force => mp2_force

      factor_2c = -4.0_dp
      IF (mp2_env%method == ri_rpa_method_gpw) THEN
         factor_2c = -1.0_dp
         IF (nspins == 2) factor_2c = -2.0_dp
      END IF

      ! prepare integral derivatives with mme method
      IF (eri_method == do_eri_mme) THEN
         ALLOCATE (matrix_P_munu_local(my_group_L_size))
         ALLOCATE (mat_munu_local(my_group_L_size))
         L_counter = 0
         DO LLL = my_group_L_start, my_group_L_end
            L_counter = L_counter + 1

            ALLOCATE (matrix_P_munu_local(L_counter)%matrix)
            CALL dbcsr_create(matrix_P_munu_local(L_counter)%matrix, template=mat_munu%matrix, &
                              matrix_type=dbcsr_type_symmetric)
            ALLOCATE (mat_munu_local(L_counter)%matrix)
            CALL dbcsr_create(mat_munu_local(L_counter)%matrix, template=mat_munu%matrix, &
                              matrix_type=dbcsr_type_symmetric)
            CALL dbcsr_copy(mat_munu_local(L_counter)%matrix, mat_munu%matrix)
            CALL dbcsr_set(mat_munu_local(L_counter)%matrix, 0.0_dp)

            CALL G_P_transform_MO_to_AO(matrix_P_munu_local(L_counter)%matrix, matrix_P_munu_nosym, mat_munu%matrix, &
                                        G_P_ia(:, L_counter), matrix_P_inu, &
                                        mo_coeff_v, mo_coeff_o, eps_filter)
         END DO

         ALLOCATE (I_tmp2(dimen_RI, my_group_L_size))
         I_tmp2(:, :) = 0.0_dp
         CALL mp2_eri_2c_integrate(eri_param, mp2_env%potential_parameter, para_env_sub, qs_env, &
                                   basis_type_a="RI_AUX", basis_type_b="RI_AUX", &
                                   hab=I_tmp2, first_b=my_group_L_start, last_b=my_group_L_end, &
                                   eri_method=eri_method, pab=G_PQ_local, force_a=force_2c)
         IF (.NOT. compare_potential_types(mp2_env%potential_parameter, mp2_env%ri_metric)) THEN
            I_tmp2(:, :) = 0.0_dp
            CALL mp2_eri_2c_integrate(eri_param, mp2_env%ri_metric, para_env_sub, qs_env, &
                                      basis_type_a="RI_AUX", basis_type_b="RI_AUX", &
                                      hab=I_tmp2, first_b=my_group_L_start, last_b=my_group_L_end, &
                                      eri_method=eri_method, pab=G_PQ_local_2, force_a=force_2c_RI)
         END IF
         DEALLOCATE (I_tmp2)

         CALL mp2_eri_3c_integrate(eri_param, mp2_env%ri_metric, para_env_sub, qs_env, &
                                   first_c=my_group_L_start, last_c=my_group_L_end, mat_ab=mat_munu_local, &
                                   basis_type_a="ORB", basis_type_b="ORB", basis_type_c="RI_AUX", &
                                   sab_nl=sab_orb_sub, eri_method=eri_method, &
                                   pabc=matrix_P_munu_local, &
                                   force_a=force_3c_orb_mu, force_b=force_3c_orb_nu, force_c=force_3c_aux)

         L_counter = 0
         DO LLL = my_group_L_start, my_group_L_end
            L_counter = L_counter + 1
            DO ispin = 1, nspins
               CALL dbcsr_multiply("N", "T", 1.0_dp, mo_coeff_v(ispin)%matrix, G_P_ia(ispin, L_counter)%matrix, &
                                   0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter)
            END DO

            IF (do_smearing) THEN
               CALL update_G_P_ia_FT(matrix_P_inu, G_P_ia(:, L_counter), mo_coeff_o, mo_coeff_v, occupation, mat_munu%matrix, &
                                     Lag_mu_i_1, LagFT_mu_i_1, Lag_nu_a_2, LagFT_nu_a_2, &
                                     Lag2_mu_i_1, LagFT2_mu_i_1, Lag2_nu_a_2, LagFT2_nu_a_2, eps_filter)
            ELSE
               CALL update_lagrangian(mat_munu_local(L_counter)%matrix, matrix_P_inu, Lag_mu_i_1, &
                                      G_P_ia(:, L_counter), mo_coeff_o, Lag_nu_a_2, &
                                      eps_filter)
            END IF

            DO ispin = 1, nspins
               CALL dbcsr_release(G_P_ia(ispin, L_counter)%matrix)
               DEALLOCATE (G_P_ia(ispin, L_counter)%matrix)
            END DO
         END DO

         DO ikind = 1, SIZE(force)
            mp2_force(ikind)%mp2_non_sep(:, :) = factor_2c*force_2c(ikind)%forces(:, :) + &
                                                 force_3c_orb_mu(ikind)%forces(:, :) + &
                                                 force_3c_orb_nu(ikind)%forces(:, :) + &
                                                 force_3c_aux(ikind)%forces(:, :)

            IF (.NOT. compare_potential_types(mp2_env%potential_parameter, mp2_env%ri_metric)) THEN
               mp2_force(ikind)%mp2_non_sep(:, :) = mp2_force(ikind)%mp2_non_sep(:, :) + factor_2c*force_2c_RI(ikind)%forces
            END IF
         END DO

         CALL mp2_eri_deallocate_forces(force_2c)
         IF (.NOT. compare_potential_types(mp2_env%potential_parameter, mp2_env%ri_metric)) THEN
            CALL mp2_eri_deallocate_forces(force_2c_RI)
         END IF
         CALL mp2_eri_deallocate_forces(force_3c_aux)
         CALL mp2_eri_deallocate_forces(force_3c_orb_mu)
         CALL mp2_eri_deallocate_forces(force_3c_orb_nu)
         CALL dbcsr_deallocate_matrix_set(matrix_P_munu_local)
         CALL dbcsr_deallocate_matrix_set(mat_munu_local)

      ELSEIF (eri_method == do_eri_gpw) THEN
         CALL get_qs_env(qs_env, ks_env=ks_env)

         CALL get_atomic_kind_set(atomic_kind_set, kind_of=kind_of, atom_of_kind=atom_of_kind)

         ! Supporting stuff for GPW
         CALL prepare_gpw(qs_env, dft_control, e_cutoff_old, cutoff_old, relative_cutoff_old, para_env_sub, pw_env_sub, &
                          auxbas_pw_pool, poisson_env, task_list_sub, rho_r, rho_g, pot_g, psi_L, sab_orb_sub)

         ! in case virial is required we need auxiliary pw
         ! for calculate the MP2-volume contribution to the virial
         ! (hartree potential derivatives)
         IF (use_virial) THEN
            CALL auxbas_pw_pool%create_pw(rho_g_copy)
            DO i = 1, 3
               CALL auxbas_pw_pool%create_pw(dvg(i))
            END DO
         END IF

         ! start main loop over auxiliary basis functions
         CALL timeset(routineN//"_loop", handle2)

         IF (use_virial) h_stress = 0.0_dp

         ALLOCATE (matrix_P_munu%matrix)
         CALL dbcsr_create(matrix_P_munu%matrix, template=mat_munu%matrix, &
                           matrix_type=dbcsr_type_symmetric)

         L_counter = 0
         DO LLL = my_group_L_start, my_group_L_end
            L_counter = L_counter + 1

            CALL G_P_transform_MO_to_AO(matrix_P_munu%matrix, matrix_P_munu_nosym, mat_munu%matrix, &
                                        G_P_ia(:, L_counter), matrix_P_inu, &
                                        mo_coeff_v, mo_coeff_o, eps_filter)

            CALL integrate_potential_forces_2c(rho_r, LLL, rho_g, atomic_kind_set, &
                                               qs_kind_set, particle_set, cell, pw_env_sub, poisson_env, &
                                               pot_g, mp2_env%potential_parameter, use_virial, &
                                               rho_g_copy, dvg, kind_of, atom_of_kind, G_PQ_local(:, L_counter), &
                                               mp2_force, h_stress, para_env_sub, dft_control, psi_L, factor_2c)

            IF (.NOT. compare_potential_types(mp2_env%ri_metric, mp2_env%potential_parameter)) THEN

               CALL integrate_potential_forces_2c(rho_r, LLL, rho_g, atomic_kind_set, &
                                                  qs_kind_set, particle_set, cell, pw_env_sub, poisson_env, &
                                                  pot_g, mp2_env%ri_metric, use_virial, &
                                                  rho_g_copy, dvg, kind_of, atom_of_kind, G_PQ_local_2(:, L_counter), &
                                                  mp2_force, h_stress, para_env_sub, dft_control, psi_L, factor_2c)
            END IF

            IF (use_virial) pv_virial = virial%pv_virial
            CALL integrate_potential_forces_3c_1c(mat_munu, rho_r, matrix_P_munu, qs_env, pw_env_sub, &
                                                  task_list_sub)
            IF (use_virial) THEN
               h_stress = h_stress + (virial%pv_virial - pv_virial)
               virial%pv_virial = pv_virial
            END IF

            IF (do_smearing) THEN
               CALL update_G_P_ia_FT(matrix_P_inu, G_P_ia(:, L_counter), mo_coeff_o, mo_coeff_v, occupation, mat_munu%matrix, &
                                     Lag_mu_i_1, LagFT_mu_i_1, Lag_nu_a_2, LagFT_nu_a_2, &
                                     Lag2_mu_i_1, LagFT2_mu_i_1, Lag2_nu_a_2, LagFT2_nu_a_2, eps_filter)
            ELSE
               CALL update_lagrangian(mat_munu%matrix, matrix_P_inu, Lag_mu_i_1, &
                                      G_P_ia(:, L_counter), mo_coeff_o, Lag_nu_a_2, &
                                      eps_filter)
            END IF

            DO ispin = 1, nspins
               CALL dbcsr_release(G_P_ia(ispin, L_counter)%matrix)
               DEALLOCATE (G_P_ia(ispin, L_counter)%matrix)
            END DO

            CALL integrate_potential_forces_3c_2c(matrix_P_munu, rho_r, rho_g, task_list_sub, pw_env_sub, &
                                                  mp2_env%ri_metric, &
                                                  ks_env, poisson_env, pot_g, use_virial, rho_g_copy, dvg, &
                                                  h_stress, para_env_sub, kind_of, atom_of_kind, &
                                                  qs_kind_set, particle_set, cell, LLL, mp2_force, dft_control)
         END DO

         CALL timestop(handle2)

         DEALLOCATE (kind_of)
         DEALLOCATE (atom_of_kind)

         IF (use_virial) THEN
            CALL auxbas_pw_pool%give_back_pw(rho_g_copy)
            DO i = 1, 3
               CALL auxbas_pw_pool%give_back_pw(dvg(i))
            END DO
         END IF

         CALL cleanup_gpw(qs_env, e_cutoff_old, cutoff_old, relative_cutoff_old, para_env_sub, pw_env_sub, &
                          task_list_sub, auxbas_pw_pool, rho_r, rho_g, pot_g, psi_L)

         CALL dbcsr_release(matrix_P_munu%matrix)
         DEALLOCATE (matrix_P_munu%matrix)

      END IF

      IF (use_virial) mp2_env%ri_grad%mp2_virial = h_stress

      DEALLOCATE (G_PQ_local)
      IF (.NOT. compare_potential_types(mp2_env%ri_metric, mp2_env%potential_parameter)) DEALLOCATE (G_PQ_local_2)

      CALL dbcsr_release(matrix_P_munu_nosym)

      DO ispin = 1, nspins
         CALL dbcsr_release(matrix_P_inu(ispin))
      END DO
      DEALLOCATE (matrix_P_inu, G_P_ia)

      ! move the forces in the correct place
      IF (eri_method == do_eri_gpw) THEN
         DO ikind = 1, SIZE(mp2_force)
            mp2_force(ikind)%mp2_non_sep(:, :) = force(ikind)%rho_elec(:, :)
            force(ikind)%rho_elec(:, :) = 0.0_dp
         END DO
      END IF

      ! Now we move from the local matrices to the global ones
      ! defined over all MPI tasks
      ! Start with moving from the DBCSR to FM for the lagrangians

      IF (.NOT. do_smearing) THEN
         CALL create_W_P(qs_env, mp2_env, mo_coeff, homo, virtual, para_env, &
                         para_env_sub, blacs_env_sub, Eigenval, Lag_mu_i_1, Lag_nu_a_2)
      ELSE
         CALL create_W_P_FT(qs_env, mp2_env, mo_coeff, homo, virtual, para_env, &
                            para_env_sub, blacs_env_sub, Eigenval, Lag_mu_i_1, Lag_nu_a_2, &
                            Lag2_mu_i_1, Lag2_nu_a_2, LagFT_mu_i_1, LagFT_nu_a_2, &
                            LagFT2_mu_i_1, LagFT2_nu_a_2)
      END IF

      CALL timestop(handle)

   END SUBROUTINE calc_ri_mp2_nonsep

! **************************************************************************************************
!> \brief Transforms G_P_ia to G_P_munu
!> \param G_P_munu The container for G_P_munu, will be allocated and created if not allocated on entry
!> \param G_P_munu_nosym ...
!> \param mat_munu ...
!> \param G_P_ia ...
!> \param G_P_inu ...
!> \param mo_coeff_v ...
!> \param mo_coeff_o ...
!> \param eps_filter ...
! **************************************************************************************************
   SUBROUTINE G_P_transform_MO_to_AO(G_P_munu, G_P_munu_nosym, mat_munu, G_P_ia, G_P_inu, &
                                     mo_coeff_v, mo_coeff_o, eps_filter)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: G_P_munu, G_P_munu_nosym, mat_munu
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(IN)       :: G_P_ia
      TYPE(dbcsr_type), DIMENSION(:), INTENT(INOUT)      :: G_P_inu
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(IN)       :: mo_coeff_v, mo_coeff_o
      REAL(KIND=dp), INTENT(IN)                          :: eps_filter

      CHARACTER(LEN=*), PARAMETER :: routineN = 'G_P_transform_MO_to_AO'

      INTEGER                                            :: handle

      CALL G_P_transform_alpha_beta(G_P_ia, G_P_inu, G_P_munu_nosym, mo_coeff_v, mo_coeff_o, eps_filter)

      ! symmetrize
      CALL timeset(routineN//"_symmetrize", handle)
      CALL dbcsr_set(G_P_munu, 0.0_dp)
      CALL dbcsr_transposed(G_P_munu, G_P_munu_nosym)
      CALL dbcsr_add(G_P_munu, G_P_munu_nosym, &
                     alpha_scalar=2.0_dp, beta_scalar=2.0_dp)
      ! this is a trick to avoid that integrate_v_rspace starts to cry
      CALL dbcsr_copy(mat_munu, G_P_munu, keep_sparsity=.TRUE.)
      CALL dbcsr_copy(G_P_munu, mat_munu)

      CALL timestop(handle)

   END SUBROUTINE G_P_transform_MO_to_AO

! **************************************************************************************************
!> \brief ...
!> \param G_P_ia ...
!> \param G_P_inu ...
!> \param G_P_munu ...
!> \param mo_coeff_v ...
!> \param mo_coeff_o ...
!> \param eps_filter ...
! **************************************************************************************************
   SUBROUTINE G_P_transform_alpha_beta(G_P_ia, G_P_inu, G_P_munu, mo_coeff_v, mo_coeff_o, eps_filter)
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(IN)       :: G_P_ia
      TYPE(dbcsr_type), DIMENSION(:), INTENT(INOUT)      :: G_P_inu
      TYPE(dbcsr_type), INTENT(INOUT)                    :: G_P_munu
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(IN)       :: mo_coeff_v, mo_coeff_o
      REAL(KIND=dp), INTENT(IN)                          :: eps_filter

      CHARACTER(LEN=*), PARAMETER :: routineN = 'G_P_transform_alpha_beta'

      INTEGER                                            :: handle, ispin
      REAL(KIND=dp)                                      :: factor

      CALL timeset(routineN, handle)

      factor = 1.0_dp/REAL(SIZE(G_P_ia), dp)

      CALL dbcsr_set(G_P_munu, 0.0_dp)

      DO ispin = 1, SIZE(G_P_ia)
         ! first back-transformation a->nu
         CALL dbcsr_multiply("N", "T", 1.0_dp, mo_coeff_v(ispin)%matrix, G_P_ia(ispin)%matrix, &
                             0.0_dp, G_P_inu(ispin), filter_eps=eps_filter)

         ! second back-transformation i->mu
         CALL dbcsr_multiply("N", "T", factor, G_P_inu(ispin), mo_coeff_o(ispin)%matrix, &
                             1.0_dp, G_P_munu, filter_eps=eps_filter)
      END DO

      CALL timestop(handle)

   END SUBROUTINE G_P_transform_alpha_beta

! **************************************************************************************************
!> \brief ...
!> \param mat_munu ...
!> \param matrix_P_inu ...
!> \param Lag_mu_i_1 ...
!> \param G_P_ia ...
!> \param mo_coeff_o ...
!> \param Lag_nu_a_2 ...
!> \param eps_filter ...
! **************************************************************************************************
   SUBROUTINE update_lagrangian(mat_munu, matrix_P_inu, Lag_mu_i_1, &
                                G_P_ia, mo_coeff_o, Lag_nu_a_2, &
                                eps_filter)
      TYPE(dbcsr_type), INTENT(IN)                       :: mat_munu
      TYPE(dbcsr_type), DIMENSION(:), INTENT(INOUT)      :: matrix_P_inu, Lag_mu_i_1
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(INOUT)    :: G_P_ia
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(IN)       :: mo_coeff_o
      TYPE(dbcsr_type), DIMENSION(:), INTENT(INOUT)      :: Lag_nu_a_2
      REAL(KIND=dp), INTENT(IN)                          :: eps_filter

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'update_lagrangian'

      INTEGER                                            :: handle, ispin

      ! update lagrangian
      CALL timeset(routineN, handle)

      DO ispin = 1, SIZE(G_P_ia)
         ! transform first index of mat_munu and store the result into matrix_P_inu

         ! first contract mat_munu with the half back transformed Gamma_i_nu
         ! in order to update Lag_mu_i_1
         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, matrix_P_inu(ispin), &
                             1.0_dp, Lag_mu_i_1(ispin), filter_eps=eps_filter)

         ! transform first index of mat_munu and store the result into matrix_P_inu
         CALL dbcsr_set(matrix_P_inu(ispin), 0.0_dp)
         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, mo_coeff_o(ispin)%matrix, &
                             0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter)

         ! contract the transformend matrix_P_inu with the untransformend Gamma_i_a
         ! in order to update Lag_nu_a_2
         CALL dbcsr_multiply("N", "N", -1.0_dp, matrix_P_inu(ispin), G_P_ia(ispin)%matrix, &
                             1.0_dp, Lag_nu_a_2(ispin), filter_eps=eps_filter)
      END DO

      CALL timestop(handle)

   END SUBROUTINE update_lagrangian

! **************************************************************************************************
!> \brief ...
!> \param matrix_P_inu ...
!> \param G_P_ia ...
!> \param mo_coeff_o ...
!> \param mo_coeff_v ...
!> \param occupation ...
!> \param mat_munu ...
!> \param Lag_mu_i_1 ...
!> \param LagFT_mu_i_1 ...
!> \param Lag_nu_a_2 ...
!> \param LagFT_nu_a_2 ...
!> \param Lag2_mu_i_1 ...
!> \param LagFT2_mu_i_1 ...
!> \param Lag2_nu_a_2 ...
!> \param LagFT2_nu_a_2 ...
!> \param eps_filter ...
! **************************************************************************************************
   SUBROUTINE update_G_P_ia_FT(matrix_P_inu, G_P_ia, mo_coeff_o, mo_coeff_v, occupation, mat_munu, Lag_mu_i_1, LagFT_mu_i_1, &
                               Lag_nu_a_2, LagFT_nu_a_2, Lag2_mu_i_1, LagFT2_mu_i_1, Lag2_nu_a_2, LagFT2_nu_a_2, &
                               eps_filter)
      TYPE(dbcsr_type), DIMENSION(:), INTENT(INOUT)      :: matrix_P_inu
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(INOUT)    :: G_P_ia
      TYPE(dbcsr_p_type), DIMENSION(:), INTENT(IN)       :: mo_coeff_o, mo_coeff_v
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: occupation
      TYPE(dbcsr_type), INTENT(IN)                       :: mat_munu
      TYPE(dbcsr_type), DIMENSION(:), INTENT(INOUT)      :: Lag_mu_i_1, LagFT_mu_i_1, Lag_nu_a_2, &
                                                            LagFT_nu_a_2, Lag2_mu_i_1, &
                                                            LagFT2_mu_i_1, Lag2_nu_a_2, &
                                                            LagFT2_nu_a_2
      REAL(KIND=dp), INTENT(IN)                          :: eps_filter

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'update_G_P_ia_FT'
      REAL(KIND=dp), PARAMETER                           :: eps_diff = 1.0e-5_dp

      INTEGER                                            :: col, col_offset, col_size, handle, homo, &
                                                            i_col, i_row, ispin, nmo, row, &
                                                            row_offset, row_size, virtual
      REAL(KIND=dp)                                      :: occ_col, occ_diff, occ_row
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: data_block
      TYPE(dbcsr_iterator_type)                          :: iter

      ! update lagrangian
      CALL timeset(routineN, handle)

      CALL dbcsr_get_info(G_P_ia(1)%matrix, nfullrows_total=homo, nfullcols_total=virtual)
      nmo = SIZE(occupation, 1)

      ! We need to distinguish fully occupied/virtual and partially occupied orbitals due to the missing scaling factor

      DO ispin = 1, SIZE(G_P_ia)
         ! Step 1: P_mo
         ! Part A1: fully virtual orbitals
         ! transform first index of mat_munu and store the result into matrix_P_inu
         CALL dbcsr_set(matrix_P_inu(ispin), 0.0_dp)
         IF (homo + virtual - nmo + 1 <= virtual) THEN
            CALL dbcsr_multiply("N", "T", 1.0_dp, mo_coeff_v(ispin)%matrix, G_P_ia(ispin)%matrix, &
                                0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter, &
                                first_k=homo + virtual - nmo + 1)
         END IF

         ! first contract mat_munu with the half back transformed Gamma_i_nu
         ! in order to update Lag_mu_i_1
         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, matrix_P_inu(ispin), &
                             1.0_dp, Lag_mu_i_1(ispin), filter_eps=eps_filter)

         ! Part A2: fully occupied orbitals
         ! transform first index of mat_munu and store the result into matrix_P_inu
         CALL dbcsr_set(matrix_P_inu(ispin), 0.0_dp)
         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, mo_coeff_o(ispin)%matrix, &
                             0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter, &
                             last_row=nmo - virtual)

         CALL dbcsr_multiply("N", "N", -1.0_dp, matrix_P_inu(ispin), G_P_ia(ispin)%matrix, &
                             1.0_dp, Lag_nu_a_2(ispin), filter_eps=eps_filter)

         ! Part B1: partially virtual orbitals
         CALL dbcsr_set(matrix_P_inu(ispin), 0.0_dp)
         CALL dbcsr_multiply("N", "T", 1.0_dp, mo_coeff_v(ispin)%matrix, G_P_ia(ispin)%matrix, &
                             0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter, &
                             last_k=homo + virtual - nmo)

         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, matrix_P_inu(ispin), &
                             1.0_dp, Lag2_mu_i_1(ispin), filter_eps=eps_filter)

         ! Part B2: partially occupied orbitals
         CALL dbcsr_set(matrix_P_inu(ispin), 0.0_dp)
         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, mo_coeff_o(ispin)%matrix, &
                             0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter, &
                             first_row=nmo - virtual + 1)

         ! contract the transformend matrix_P_inu with the untransformend Gamma_i_a
         ! in order to update Lag_nu_a_2
         CALL dbcsr_multiply("N", "N", -1.0_dp, matrix_P_inu(ispin), G_P_ia(ispin)%matrix, &
                             1.0_dp, Lag2_nu_a_2(ispin), filter_eps=eps_filter)

         ! Remove the difference of occupation numbers
         CALL dbcsr_iterator_start(iter, G_P_ia(ispin)%matrix)
         DO WHILE (dbcsr_iterator_blocks_left(iter))

            CALL dbcsr_iterator_next_block(iter, row, col, data_block, &
                                           row_size=row_size, col_size=col_size, &
                                           row_offset=row_offset, col_offset=col_offset)

            ! "occupied orbital"
            DO i_row = 1, MIN(row_size, homo - row_offset + 1)
               occ_row = occupation(i_row + row_offset - 1, ispin)
               ! "virtual orbital"
               DO i_col = 1, MIN(col_size, virtual - col_offset + 1)
                  occ_col = occupation(i_col + col_offset - 1 + nmo - virtual, ispin)

                  occ_diff = occ_row - occ_col

                  IF (occ_diff > eps_diff) THEN
                     data_block(i_row, i_col) = data_block(i_row, i_col)/occ_diff
                  ELSE IF (ABS(occ_diff) < eps_diff) THEN
                     WRITE (*, *) "occ_row", occ_row, occ_row*(1.0_dp - occ_row)
                     data_block(i_row, i_col) = data_block(i_row, i_col)*(1.0_dp - 2.0_dp*occ_row)/(occ_row*(1.0_dp - occ_row))
                  ELSE
                     data_block(i_row, i_col) = 0.0_dp
                  END IF

               END DO

            END DO

         END DO
         CALL dbcsr_iterator_stop(iter)

         ! Step 2: Lagrangian (L_jb)
         ! Part A1: fully virtual orbitals
         ! transform first index of mat_munu and store the result into matrix_P_inu
         CALL dbcsr_set(matrix_P_inu(ispin), 0.0_dp)
         IF (homo + virtual - nmo + 1 <= virtual) THEN
            CALL dbcsr_multiply("N", "T", 1.0_dp, mo_coeff_v(ispin)%matrix, G_P_ia(ispin)%matrix, &
                                0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter, &
                                first_k=homo + virtual - nmo + 1)
         END IF

         ! first contract mat_munu with the half back transformed Gamma_i_nu
         ! in order to update Lag_mu_i_1
         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, matrix_P_inu(ispin), &
                             1.0_dp, LagFT_mu_i_1(ispin), filter_eps=eps_filter)

         ! Part A2: fully occupied orbitals
         ! transform first index of mat_munu and store the result into matrix_P_inu
         CALL dbcsr_set(matrix_P_inu(ispin), 0.0_dp)
         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, mo_coeff_o(ispin)%matrix, &
                             0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter, &
                             last_row=nmo - virtual)

         CALL dbcsr_multiply("N", "N", -1.0_dp, matrix_P_inu(ispin), G_P_ia(ispin)%matrix, &
                             1.0_dp, LagFT_nu_a_2(ispin), filter_eps=eps_filter)

         ! Part B1: partially virtual orbitals
         CALL dbcsr_set(matrix_P_inu(ispin), 0.0_dp)
         CALL dbcsr_multiply("N", "T", 1.0_dp, mo_coeff_v(ispin)%matrix, G_P_ia(ispin)%matrix, &
                             0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter, &
                             last_k=homo + virtual - nmo)

         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, matrix_P_inu(ispin), &
                             1.0_dp, LagFT2_mu_i_1(ispin), filter_eps=eps_filter)

         ! Part B2: partially occupied orbitals
         CALL dbcsr_set(matrix_P_inu(ispin), 0.0_dp)
         CALL dbcsr_multiply("N", "N", 1.0_dp, mat_munu, mo_coeff_o(ispin)%matrix, &
                             0.0_dp, matrix_P_inu(ispin), filter_eps=eps_filter, &
                             first_row=nmo - virtual + 1)

         ! contract the transformend matrix_P_inu with the untransformed Gamma_i_a
         ! in order to update Lag_nu_a_2
         CALL dbcsr_multiply("N", "N", -1.0_dp, matrix_P_inu(ispin), G_P_ia(ispin)%matrix, &
                             1.0_dp, LagFT2_nu_a_2(ispin), filter_eps=eps_filter)
      END DO

      CALL timestop(handle)

   END SUBROUTINE update_G_P_ia_FT

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param mp2_env ...
!> \param mo_coeff ...
!> \param homo ...
!> \param virtual ...
!> \param para_env ...
!> \param para_env_sub ...
!> \param blacs_env_sub ...
!> \param Eigenval ...
!> \param Lag1_mu_i ...
!> \param Lag2_nu_a ...
! **************************************************************************************************
   SUBROUTINE create_W_P(qs_env, mp2_env, mo_coeff, homo, virtual, para_env, para_env_sub, blacs_env_sub, &
                         Eigenval, Lag1_mu_i, Lag2_nu_a)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(mp2_type)                                     :: mp2_env
      TYPE(cp_fm_type), DIMENSION(:), INTENT(IN)         :: mo_coeff
      INTEGER, DIMENSION(:), INTENT(IN)                  :: homo, virtual
      TYPE(mp_para_env_type), POINTER                    :: para_env, para_env_sub
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env_sub
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: Eigenval
      TYPE(dbcsr_type), ALLOCATABLE, DIMENSION(:)        :: Lag1_mu_i, Lag2_nu_a

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'create_W_P'

      INTEGER :: handle, handle2, i_global, iiB, ispin, itmp(2), j_global, jjB, my_B_virtual_end, &
         my_B_virtual_start, mypcol, myprow, nao, ncol_local, nmo, nrow_local, nspins, &
         proc_receive, proc_send, proc_shift, send_col_size, send_pcol, send_prow, send_row_size, &
         virtual_offset
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: mepos_2_grid, sizes
      INTEGER, DIMENSION(:), POINTER                     :: col_indices, row_indices
      REAL(KIND=dp)                                      :: factor
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: ab_rec, ab_send
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct_tmp
      TYPE(cp_fm_type)                                   :: fm_P_ij
      TYPE(cp_fm_type), ALLOCATABLE, DIMENSION(:)        :: L_mu_q

      CALL timeset(routineN, handle)

      nspins = SIZE(homo)

      ! create the globally distributed mixed lagrangian
      NULLIFY (blacs_env)
      CALL get_qs_env(qs_env, blacs_env=blacs_env)
      CALL merge_Lagrangian_matrices(blacs_env, mo_coeff, para_env, para_env_sub, blacs_env_sub, &
                                     Lag1_mu_i, Lag2_nu_a, L_mu_q)

      ! Set the factor to multiply P_ij (depends on the open or closed shell)
      factor = 1.0_dp
      IF (nspins == 2) factor = 0.50_dp
      CALL cp_fm_get_info(mo_coeff(1), nrow_global=nao, ncol_global=nmo)

      ALLOCATE (mp2_env%ri_grad%P_mo(SIZE(mp2_env%ri_grad%mo_coeff_o)))
      ALLOCATE (mp2_env%ri_grad%W_mo(SIZE(mp2_env%ri_grad%mo_coeff_o)))
      ALLOCATE (mp2_env%ri_grad%L_jb(SIZE(mp2_env%ri_grad%mo_coeff_o)))

      DO ispin = 1, nspins
         virtual_offset = nmo - virtual(ispin)

         ! update the P_ij block of P_MP2 with the
         ! non-singular ij pairs
         CALL timeset(routineN//"_Pij", handle2)
         NULLIFY (fm_struct_tmp)
         CALL cp_fm_struct_create(fm_struct_tmp, para_env=para_env, context=blacs_env, &
                                  nrow_global=homo(ispin), ncol_global=homo(ispin))
         CALL cp_fm_create(fm_P_ij, fm_struct_tmp, name="fm_P_ij")
         CALL cp_fm_struct_release(fm_struct_tmp)
         CALL cp_fm_set_all(fm_P_ij, 0.0_dp)

         ! we have it, update P_ij local
         CALL cp_fm_get_info(matrix=fm_P_ij, &
                             nrow_local=nrow_local, &
                             ncol_local=ncol_local, &
                             row_indices=row_indices, &
                             col_indices=col_indices)

         IF (.NOT. mp2_env%method == ri_rpa_method_gpw) THEN
            CALL parallel_gemm('T', 'N', homo(ispin), homo(ispin), nao, 2.0_dp, &
                               mo_coeff(ispin), L_mu_q(ispin), 1.0_dp, fm_P_ij, &
                               a_first_col=1, &
                               a_first_row=1, &
                               b_first_col=1, &
                               b_first_row=1, &
                               c_first_col=1, &
                               c_first_row=1)
            CALL parallel_gemm('T', 'N', homo(ispin), homo(ispin), nao, -2.0_dp, &
                               L_mu_q(ispin), mo_coeff(ispin), 1.0_dp, fm_P_ij, &
                               a_first_col=1, &
                               a_first_row=1, &
                               b_first_col=1, &
                               b_first_row=1, &
                               c_first_col=1, &
                               c_first_row=1)

            IF (ALLOCATED(mp2_env%ri_grad%P_ij(ispin)%array)) THEN
            DO jjB = 1, ncol_local
               j_global = col_indices(jjB)
               DO iiB = 1, nrow_local
                  i_global = row_indices(iiB)
                  ! diagonal elements and nearly degenerate ij pairs already updated
                  IF (ABS(Eigenval(j_global, ispin) - Eigenval(i_global, ispin)) < mp2_env%ri_grad%eps_canonical) THEN
                     fm_P_ij%local_data(iiB, jjB) = mp2_env%ri_grad%P_ij(ispin)%array(i_global, j_global)
                  ELSE
                     fm_P_ij%local_data(iiB, jjB) = &
                        factor*fm_P_ij%local_data(iiB, jjB)/(Eigenval(j_global, ispin) - Eigenval(i_global, ispin))
                  END IF
               END DO
            END DO
            END IF
         ELSE IF (ALLOCATED(mp2_env%ri_grad%P_ij(ispin)%array)) THEN
            DO jjB = 1, ncol_local
               j_global = col_indices(jjB)
               DO iiB = 1, nrow_local
                  i_global = row_indices(iiB)
                  fm_P_ij%local_data(iiB, jjB) = mp2_env%ri_grad%P_ij(ispin)%array(i_global, j_global)
               END DO
            END DO
         END IF
         ! deallocate the local P_ij
         IF (ALLOCATED(mp2_env%ri_grad%P_ij(ispin)%array)) DEALLOCATE (mp2_env%ri_grad%P_ij(ispin)%array)
         CALL timestop(handle2)

         CALL timeset(routineN//"_PMO", handle2)
         ! Now create and fill the P matrix (MO)
         ! FOR NOW WE ASSUME P_ab AND P_ij ARE REPLICATED OVER EACH MPI RANK

         NULLIFY (fm_struct_tmp)
         CALL cp_fm_struct_create(fm_struct_tmp, para_env=para_env, context=blacs_env, &
                                  nrow_global=nmo, ncol_global=nmo)
         CALL cp_fm_create(mp2_env%ri_grad%P_mo(ispin), fm_struct_tmp, name="P_MP2_MO")
         CALL cp_fm_set_all(mp2_env%ri_grad%P_mo(ispin), 0.0_dp)

         ! start with the (easy) occ-occ block and locally held P_ab elements
         itmp = get_limit(virtual(ispin), para_env_sub%num_pe, para_env_sub%mepos)
         my_B_virtual_start = itmp(1)
         my_B_virtual_end = itmp(2)

         ! Fill occ-occ block
         CALL cp_fm_geadd(1.0_dp, "N", fm_P_ij, 1.0_dp, mp2_env%ri_grad%P_mo(ispin))
         CALL cp_fm_release(fm_P_ij)

         CALL cp_fm_get_info(mp2_env%ri_grad%P_mo(ispin), &
                             nrow_local=nrow_local, &
                             ncol_local=ncol_local, &
                             row_indices=row_indices, &
                             col_indices=col_indices)

         IF (mp2_env%method == ri_mp2_laplace) THEN
            CALL parallel_gemm('T', 'N', virtual(ispin), virtual(ispin), nao, 1.0_dp, &
                               mo_coeff(ispin), L_mu_q(ispin), 0.0_dp, mp2_env%ri_grad%P_mo(ispin), &
                               a_first_col=virtual_offset + 1, &
                               a_first_row=1, &
                               b_first_col=virtual_offset + 1, &
                               b_first_row=1, &
                               c_first_col=virtual_offset + 1, &
                               c_first_row=virtual_offset + 1)
            CALL parallel_gemm('T', 'N', virtual(ispin), virtual(ispin), nao, -2.0_dp, &
                               L_mu_q(ispin), mo_coeff(ispin), 2.0_dp, mp2_env%ri_grad%P_mo(ispin), &
                               a_first_col=virtual_offset + 1, &
                               a_first_row=1, &
                               b_first_col=virtual_offset + 1, &
                               b_first_row=1, &
                               c_first_col=virtual_offset + 1, &
                               c_first_row=virtual_offset + 1)
            ! With Laplace-SOS-MP2, we need to quench matrix elements of the non-degenerate pairs
            ! and rescale the occupied-occupied pairs
            DO jjB = 1, ncol_local
               j_global = col_indices(jjB)
               IF (j_global <= virtual_offset) CYCLE
               DO iiB = 1, nrow_local
                  i_global = row_indices(iiB)
                  IF (ABS(Eigenval(i_global, ispin) - Eigenval(j_global, ispin)) < mp2_env%ri_grad%eps_canonical) THEN
                     mp2_env%ri_grad%P_mo(ispin)%local_data(iiB, jjB) = 0.0_dp
                  ELSE
                     mp2_env%ri_grad%P_mo(ispin)%local_data(iiB, jjB) = &
                        factor*mp2_env%ri_grad%P_mo(ispin)%local_data(iiB, jjB)/ &
                        (Eigenval(i_global, ispin) - Eigenval(j_global, ispin))
                  END IF
               END DO
            END DO
         END IF

         IF (mp2_env%method == ri_mp2_method_gpw .OR. mp2_env%method == ri_rpa_method_gpw) THEN
            ! With MP2 and RPA, we have already calculated the density matrix elements
            DO jjB = 1, ncol_local
               j_global = col_indices(jjB)
               IF (j_global <= virtual_offset) CYCLE
               DO iiB = 1, nrow_local
                  i_global = row_indices(iiB)
                  IF (my_B_virtual_start <= i_global - virtual_offset .AND. &
                      i_global - virtual_offset <= my_B_virtual_end) THEN
                     mp2_env%ri_grad%P_mo(ispin)%local_data(iiB, jjB) = &
                        mp2_env%ri_grad%P_mo(ispin)%local_data(iiB, jjB) + &
                        mp2_env%ri_grad%P_ab(ispin)%array(i_global - virtual_offset - &
                                                          my_B_virtual_start + 1, j_global - virtual_offset)
                  END IF
               END DO
            END DO
         ELSE IF (mp2_env%method == ri_mp2_laplace) THEN
            ! Now, we only need to add the contributions from degenerate orbital pairs
            DO jjB = 1, ncol_local
               j_global = col_indices(jjB)
               IF (j_global <= virtual_offset) CYCLE
               DO iiB = 1, nrow_local
                  i_global = row_indices(iiB)
                  IF (ABS(Eigenval(i_global, ispin) - Eigenval(j_global, ispin)) < mp2_env%ri_grad%eps_canonical) THEN
                  IF (my_B_virtual_start <= i_global - virtual_offset .AND. &
                      i_global - virtual_offset <= my_B_virtual_end) THEN
                     mp2_env%ri_grad%P_mo(ispin)%local_data(iiB, jjB) = &
                        mp2_env%ri_grad%P_mo(ispin)%local_data(iiB, jjB) + &
                        mp2_env%ri_grad%P_ab(ispin)%array(i_global - virtual_offset - &
                                                          my_B_virtual_start + 1, j_global - virtual_offset)
                  END IF
                  END IF
               END DO
            END DO
         ELSE
            CPABORT("Calculation of virt-virt block of density matrix is dealt with elsewhere!")
         END IF

         ! send around the sub_group the local data and check if we
         ! have to update our block with external elements
         ALLOCATE (mepos_2_grid(2, 0:para_env_sub%num_pe - 1))
         myprow = mp2_env%ri_grad%P_mo(ispin)%matrix_struct%context%mepos(1)
         mypcol = mp2_env%ri_grad%P_mo(ispin)%matrix_struct%context%mepos(2)
         CALL para_env_sub%allgather([myprow, mypcol], mepos_2_grid)

         ALLOCATE (sizes(2, 0:para_env_sub%num_pe - 1))
         CALL para_env_sub%allgather([nrow_local, ncol_local], sizes)

         ALLOCATE (ab_rec(nrow_local, ncol_local))
         DO proc_shift = 1, para_env_sub%num_pe - 1
            proc_send = MODULO(para_env_sub%mepos + proc_shift, para_env_sub%num_pe)
            proc_receive = MODULO(para_env_sub%mepos - proc_shift, para_env_sub%num_pe)

            send_prow = mepos_2_grid(1, proc_send)
            send_pcol = mepos_2_grid(2, proc_send)

            send_row_size = sizes(1, proc_send)
            send_col_size = sizes(2, proc_send)

            ALLOCATE (ab_send(send_row_size, send_col_size))
            ab_send = 0.0_dp

            ! first loop over row since in this way we can cycle
            DO iiB = 1, send_row_size
               i_global = mp2_env%ri_grad%P_mo(ispin)%matrix_struct%l2g_row(iiB, send_prow)
               IF (i_global <= virtual_offset) CYCLE
               i_global = i_global - virtual_offset
               IF (.NOT. (my_B_virtual_start <= i_global .AND. i_global <= my_B_virtual_end)) CYCLE
               DO jjB = 1, send_col_size
                  j_global = mp2_env%ri_grad%P_mo(ispin)%matrix_struct%l2g_col(jjB, send_pcol)
                  IF (j_global > virtual_offset) THEN
                     j_global = j_global - virtual_offset
                     ab_send(iiB, jjB) = mp2_env%ri_grad%P_ab(ispin)%array(i_global - my_B_virtual_start + 1, j_global)
                  END IF
               END DO
            END DO

            ab_rec = 0.0_dp
            CALL para_env_sub%sendrecv(ab_send, proc_send, &
                                       ab_rec, proc_receive)
            mp2_env%ri_grad%P_mo(ispin)%local_data(1:nrow_local, 1:ncol_local) = &
               mp2_env%ri_grad%P_mo(ispin)%local_data(1:nrow_local, 1:ncol_local) + &
               ab_rec(1:nrow_local, 1:ncol_local)

            DEALLOCATE (ab_send)
         END DO
         DEALLOCATE (ab_rec)
         DEALLOCATE (mepos_2_grid)
         DEALLOCATE (sizes)

         ! deallocate the local P_ab
         DEALLOCATE (mp2_env%ri_grad%P_ab(ispin)%array)
         CALL timestop(handle2)

         ! create also W_MP2_MO
         CALL timeset(routineN//"_WMO", handle2)

         CALL cp_fm_create(mp2_env%ri_grad%W_mo(ispin), fm_struct_tmp, name="W_MP2_MO")
         CALL cp_fm_struct_release(fm_struct_tmp)

         ! all block
         CALL parallel_gemm('T', 'N', nmo, nmo, nao, 2.0_dp*factor, &
                            L_mu_q(ispin), mo_coeff(ispin), 0.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)

         ! occ-occ block
         CALL parallel_gemm('T', 'N', homo(ispin), homo(ispin), nao, -2.0_dp*factor, &
                            L_mu_q(ispin), mo_coeff(ispin), 0.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)

         ! occ-virt block
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor, &
                            mo_coeff(ispin), L_mu_q(ispin), 0.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=virtual_offset + 1, &
                            c_first_row=1)
         CALL timestop(handle2)

         ! Calculate occ-virt block of the lagrangian in MO
         CALL timeset(routineN//"_Ljb", handle2)

         CALL cp_fm_struct_create(fm_struct_tmp, para_env=para_env, context=blacs_env, &
                                  nrow_global=homo(ispin), ncol_global=virtual(ispin))
         CALL cp_fm_create(mp2_env%ri_grad%L_jb(ispin), fm_struct_tmp, name="fm_L_jb")
         CALL cp_fm_struct_release(fm_struct_tmp)

         ! first Virtual
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor, &
                            L_mu_q(ispin), mo_coeff(ispin), 0.0_dp, mp2_env%ri_grad%L_jb(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)
         ! then occupied
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor, &
                            mo_coeff(ispin), L_mu_q(ispin), 1.0_dp, mp2_env%ri_grad%L_jb(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)

         ! finally release L_mu_q
         CALL timestop(handle2)

         CALL cp_fm_release(L_mu_q(ispin))
      END DO
      DEALLOCATE (L_mu_q)

      CALL timestop(handle)

   END SUBROUTINE create_W_P

! **************************************************************************************************
!> \brief ...
!> \param qs_env ...
!> \param mp2_env ...
!> \param mo_coeff ...
!> \param homo ...
!> \param virtual ...
!> \param para_env ...
!> \param para_env_sub ...
!> \param blacs_env_sub ...
!> \param Eigenval ...
!> \param LagG1_mu_i ...
!> \param LagG1_nu_a ...
!> \param LagG2_mu_i ...
!> \param LagG2_nu_a ...
!> \param LagL1_mu_i ...
!> \param LagL1_nu_a ...
!> \param LagL2_mu_i ...
!> \param LagL2_nu_a ...
! **************************************************************************************************
   SUBROUTINE create_W_P_FT(qs_env, mp2_env, mo_coeff, homo, virtual, para_env, para_env_sub, blacs_env_sub, &
                            Eigenval, LagG1_mu_i, LagG1_nu_a, LagG2_mu_i, LagG2_nu_a, LagL1_mu_i, LagL1_nu_a, &
                            LagL2_mu_i, LagL2_nu_a)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(mp2_type)                                     :: mp2_env
      TYPE(cp_fm_type), DIMENSION(:), INTENT(IN)         :: mo_coeff
      INTEGER, DIMENSION(:), INTENT(IN)                  :: homo, virtual
      TYPE(mp_para_env_type), POINTER                    :: para_env, para_env_sub
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env_sub
      REAL(KIND=dp), DIMENSION(:, :), INTENT(IN)         :: Eigenval
      TYPE(dbcsr_type), ALLOCATABLE, DIMENSION(:)        :: LagG1_mu_i, LagG1_nu_a, LagG2_mu_i, &
                                                            LagG2_nu_a, LagL1_mu_i, LagL1_nu_a, &
                                                            LagL2_mu_i, LagL2_nu_a

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'create_W_P_FT'

      INTEGER :: handle, handle2, i_global, iiB, ispin, itmp(2), j_global, jjB, my_B_virtual_end, &
         my_B_virtual_start, mypcol, myprow, nao, ncol_local, nmo, nrow_local, nspins, &
         proc_receive, proc_send, proc_shift, send_col_size, send_pcol, send_prow, send_row_size, &
         virtual_offset
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: mepos_2_grid, sizes
      INTEGER, DIMENSION(:), POINTER                     :: col_indices, row_indices
      REAL(KIND=dp)                                      :: factor
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: ab_rec, ab_send
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct_tmp
      TYPE(cp_fm_type)                                   :: fm_P_ij
      TYPE(cp_fm_type), ALLOCATABLE, DIMENSION(:)        :: LG1_mu_q, LG2_mu_q, LL1_mu_q, LL2_mu_q

      CALL timeset(routineN, handle)

      MARK_USED(Eigenval)

      ! At finite temperatures, we need to consider the weights to prevent double-counting from the different orbital pairs
      ! Thus, there are two different Lagrangians, one for which the sum ran over all fully occupied/unoccupied orbitals
      ! and one running over partially occupied orbitals
      ! Secondly, we need the matrix with occupation and without. This is not necessary at T=0 as these weights are one anyways

      nspins = SIZE(homo)

      ! create the globally distributed mixed lagrangian
      NULLIFY (blacs_env)
      CALL get_qs_env(qs_env, blacs_env=blacs_env)
      CALL merge_Lagrangian_matrices(blacs_env, mo_coeff, para_env, para_env_sub, blacs_env_sub, &
                                     LagG1_mu_i, LagG1_nu_a, LG1_mu_q)
      CALL merge_Lagrangian_matrices(blacs_env, mo_coeff, para_env, para_env_sub, blacs_env_sub, &
                                     LagG2_mu_i, LagG2_nu_a, LG2_mu_q)
      CALL merge_Lagrangian_matrices(blacs_env, mo_coeff, para_env, para_env_sub, blacs_env_sub, &
                                     LagL1_mu_i, LagL1_nu_a, LL1_mu_q)
      CALL merge_Lagrangian_matrices(blacs_env, mo_coeff, para_env, para_env_sub, blacs_env_sub, &
                                     LagL2_mu_i, LagL2_nu_a, LL2_mu_q)

      ! Set the factor to multiply P_ij (depends on the open or closed shell)
      factor = 1.0_dp
      IF (nspins == 2) factor = 0.50_dp
      CALL cp_fm_get_info(mo_coeff(1), nrow_global=nao, ncol_global=nmo)

      ALLOCATE (mp2_env%ri_grad%P_mo(SIZE(mp2_env%ri_grad%mo_coeff_o)))
      ALLOCATE (mp2_env%ri_grad%P_mo_FTab(SIZE(mp2_env%ri_grad%mo_coeff_o)))
      ALLOCATE (mp2_env%ri_grad%W_mo(SIZE(mp2_env%ri_grad%mo_coeff_o)))

      DO ispin = 1, nspins

         virtual_offset = nmo - virtual(ispin)

         ! update the P_ij block of P_MP2 with the
         ! non-singular ij pairs
         CALL timeset(routineN//"_Pij", handle2)
         NULLIFY (fm_struct_tmp)
         CALL cp_fm_struct_create(fm_struct_tmp, para_env=para_env, context=blacs_env, &
                                  nrow_global=homo(ispin), ncol_global=homo(ispin))
         CALL cp_fm_create(fm_P_ij, fm_struct_tmp, name="fm_P_ij")
         CALL cp_fm_struct_release(fm_struct_tmp)
         CALL cp_fm_set_all(fm_P_ij, 0.0_dp)

         ! we have it, update P_ij local
         CALL cp_fm_get_info(matrix=fm_P_ij, &
                             nrow_local=nrow_local, &
                             ncol_local=ncol_local, &
                             row_indices=row_indices, &
                             col_indices=col_indices)

         IF (.NOT. mp2_env%method == ri_rpa_method_gpw) THEN
            CPABORT("Finite Temperature MP2 not implemented yet")
         ELSE IF (ALLOCATED(mp2_env%ri_grad%P_ij(ispin)%array)) THEN
            DO jjB = 1, ncol_local
               j_global = col_indices(jjB)
               DO iiB = 1, nrow_local
                  i_global = row_indices(iiB)
                  fm_P_ij%local_data(iiB, jjB) = mp2_env%ri_grad%P_ij(ispin)%array(i_global, j_global)
               END DO
            END DO
         END IF
         ! deallocate the local P_ij
         IF (ALLOCATED(mp2_env%ri_grad%P_ij(ispin)%array)) DEALLOCATE (mp2_env%ri_grad%P_ij(ispin)%array)
         CALL timestop(handle2)

         CALL timeset(routineN//"_PMO", handle2)
         ! Now create and fill the P matrix (MO)
         ! FOR NOW WE ASSUME P_ab AND P_ij ARE REPLICATED OVER EACH MPI RANK

         NULLIFY (fm_struct_tmp)
         CALL cp_fm_struct_create(fm_struct_tmp, para_env=para_env, context=blacs_env, &
                                  nrow_global=nmo, ncol_global=nmo)
         CALL cp_fm_create(mp2_env%ri_grad%P_mo(ispin), fm_struct_tmp, name="P_MP2_MO")
         CALL cp_fm_set_all(mp2_env%ri_grad%P_mo(ispin), 0.0_dp)

         ! start with the (easy) occ-occ block and locally held P_ab elements
         itmp = get_limit(virtual(ispin), para_env_sub%num_pe, para_env_sub%mepos)
         my_B_virtual_start = itmp(1)
         my_B_virtual_end = itmp(2)

         ! Fill occ-occ block
         CALL cp_fm_geadd(1.0_dp, "N", fm_P_ij, 1.0_dp, mp2_env%ri_grad%P_mo(ispin))
         CALL cp_fm_release(fm_P_ij)

         CALL cp_fm_create(mp2_env%ri_grad%P_mo_FTab(ispin), mp2_env%ri_grad%P_mo(ispin)%matrix_struct, name="P_MP2_MO")
         CALL cp_fm_set_all(mp2_env%ri_grad%P_mo_FTab(ispin), 0.0_dp)

         CALL cp_fm_get_info(mp2_env%ri_grad%P_mo(ispin), &
                             nrow_local=nrow_local, &
                             ncol_local=ncol_local, &
                             row_indices=row_indices, &
                             col_indices=col_indices)

         IF (mp2_env%method == ri_mp2_laplace .OR. mp2_env%method == ri_mp2_method_gpw) THEN
            CPABORT("NIY")
         ELSE IF (mp2_env%method == ri_rpa_method_gpw) THEN
            ! With MP2 and RPA, we have already calculated the density matrix elements
            DO jjB = 1, ncol_local
               j_global = col_indices(jjB)
               IF (j_global <= virtual_offset) CYCLE
               DO iiB = 1, nrow_local
                  i_global = row_indices(iiB)
                  IF (my_B_virtual_start <= i_global - virtual_offset .AND. &
                      i_global - virtual_offset <= my_B_virtual_end) THEN
                     mp2_env%ri_grad%P_mo_FTab(ispin)%local_data(iiB, jjB) = &
                        mp2_env%ri_grad%P_ab(ispin)%array(i_global - virtual_offset - &
                                                          my_B_virtual_start + 1, j_global - virtual_offset)
                  END IF
               END DO
            END DO
         END IF

         ! send around the sub_group the local data and check if we
         ! have to update our block with external elements
         ALLOCATE (mepos_2_grid(2, 0:para_env_sub%num_pe - 1))
         myprow = mp2_env%ri_grad%P_mo(ispin)%matrix_struct%context%mepos(1)
         mypcol = mp2_env%ri_grad%P_mo(ispin)%matrix_struct%context%mepos(2)
         CALL para_env_sub%allgather([myprow, mypcol], mepos_2_grid)

         ALLOCATE (sizes(2, 0:para_env_sub%num_pe - 1))
         CALL para_env_sub%allgather([nrow_local, ncol_local], sizes)

         ALLOCATE (ab_rec(nrow_local, ncol_local))
         DO proc_shift = 1, para_env_sub%num_pe - 1
            proc_send = MODULO(para_env_sub%mepos + proc_shift, para_env_sub%num_pe)
            proc_receive = MODULO(para_env_sub%mepos - proc_shift, para_env_sub%num_pe)

            send_prow = mepos_2_grid(1, proc_send)
            send_pcol = mepos_2_grid(2, proc_send)

            send_row_size = sizes(1, proc_send)
            send_col_size = sizes(2, proc_send)

            ALLOCATE (ab_send(send_row_size, send_col_size))
            ab_send = 0.0_dp

            ! first loop over row since in this way we can cycle
            DO iiB = 1, send_row_size
               i_global = mp2_env%ri_grad%P_mo(ispin)%matrix_struct%l2g_row(iiB, send_prow)
               IF (i_global <= virtual_offset) CYCLE
               i_global = i_global - virtual_offset
               IF (.NOT. (my_B_virtual_start <= i_global .AND. i_global <= my_B_virtual_end)) CYCLE
               DO jjB = 1, send_col_size
                  j_global = mp2_env%ri_grad%P_mo(ispin)%matrix_struct%l2g_col(jjB, send_pcol)
                  IF (j_global > virtual_offset) THEN
                     j_global = j_global - virtual_offset
                     ab_send(iiB, jjB) = mp2_env%ri_grad%P_ab(ispin)%array(i_global - my_B_virtual_start + 1, j_global)
                  END IF
               END DO
            END DO

            ab_rec = 0.0_dp
            CALL para_env_sub%sendrecv(ab_send, proc_send, ab_rec, proc_receive)
            mp2_env%ri_grad%P_mo_FTab(ispin)%local_data(1:nrow_local, 1:ncol_local) = &
               ab_rec(1:nrow_local, 1:ncol_local)

            DEALLOCATE (ab_send)
         END DO
         DEALLOCATE (ab_rec)
         DEALLOCATE (mepos_2_grid)
         DEALLOCATE (sizes)

         ! deallocate the local P_ab
         DEALLOCATE (mp2_env%ri_grad%P_ab(ispin)%array)
         CALL timestop(handle2)

         ! create also W_MP2_MO
         CALL timeset(routineN//"_WMO", handle2)

         CALL cp_fm_create(mp2_env%ri_grad%W_mo(ispin), fm_struct_tmp, name="W_MP2_MO")
         CALL cp_fm_struct_release(fm_struct_tmp)

         ! all block
         CALL parallel_gemm('T', 'N', nmo, nmo, nao, 2.0_dp*factor, &
                            LG1_mu_q(ispin), mo_coeff(ispin), 0.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', nmo, nmo, nao, 2.0_dp*factor, &
                            LG2_mu_q(ispin), mo_coeff(ispin), 1.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', homo(ispin) + virtual(ispin) - nmo, homo(ispin) + virtual(ispin) - nmo, &
                            nao, 2.0_dp*factor*(SQRT(0.5_dp) - 1.0_dp), &
                            LG2_mu_q(ispin), mo_coeff(ispin), 1.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=virtual_offset + 1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=virtual_offset + 1, &
                            c_first_row=virtual_offset + 1)

         ! occ-occ block
         CALL parallel_gemm('T', 'N', homo(ispin), homo(ispin), nao, -2.0_dp*factor, &
                            LG1_mu_q(ispin), mo_coeff(ispin), 1.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', nmo - virtual(ispin), nmo - virtual(ispin), nao, -2.0_dp*factor, &
                            LG2_mu_q(ispin), mo_coeff(ispin), 1.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', homo(ispin) + virtual(ispin) - nmo, homo(ispin) + virtual(ispin) - nmo, &
                            nao, -2.0_dp*factor*(1.0_dp - SQRT(0.5_dp)), &
                            LG2_mu_q(ispin), mo_coeff(ispin), 1.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=virtual_offset + 1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=virtual_offset + 1, &
                            c_first_row=virtual_offset + 1)

         ! occ-virt block
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor, &
                            mo_coeff(ispin), LG1_mu_q(ispin), 1.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=virtual_offset + 1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor, &
                            mo_coeff(ispin), LG2_mu_q(ispin), 1.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=virtual_offset + 1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', homo(ispin) + virtual(ispin) - nmo, homo(ispin) + virtual(ispin) - nmo, &
                            nao, 2.0_dp*factor, &
                            mo_coeff(ispin), LG2_mu_q(ispin), 1.0_dp, mp2_env%ri_grad%W_mo(ispin), &
                            a_first_col=virtual_offset + 1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=virtual_offset + 1, &
                            c_first_row=1)
         CALL timestop(handle2)

         ! Calculate occ-virt block of the lagrangian in MO
         CALL timeset(routineN//"_Ljb", handle2)
         IF (.NOT. ALLOCATED(mp2_env%ri_grad%L_jb)) THEN
            ALLOCATE (mp2_env%ri_grad%L_jb(SIZE(mp2_env%ri_grad%mo_coeff_o)))
         END IF

         CALL cp_fm_struct_create(fm_struct_tmp, para_env=para_env, context=blacs_env, &
                                  nrow_global=homo(ispin), ncol_global=virtual(ispin))
         CALL cp_fm_create(mp2_env%ri_grad%L_jb(ispin), fm_struct_tmp, name="fm_L_jb")
         CALL cp_fm_struct_release(fm_struct_tmp)

         ! first Virtual
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor, &
                            LL1_mu_q(ispin), mo_coeff(ispin), 0.0_dp, mp2_env%ri_grad%L_jb(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor, &
                            LL2_mu_q(ispin), mo_coeff(ispin), 0.0_dp, mp2_env%ri_grad%L_jb(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', homo(ispin) + virtual(ispin) - nmo, homo(ispin) + virtual(ispin) - nmo, &
                            nao, 2.0_dp*factor*(SQRT(0.5_dp) - 1.0_dp), &
                            LL2_mu_q(ispin), mo_coeff(ispin), 0.0_dp, mp2_env%ri_grad%L_jb(ispin), &
                            a_first_col=virtual_offset + 1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=virtual_offset + 1, &
                            c_first_row=1)
         ! then occupied
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor, &
                            mo_coeff(ispin), LL1_mu_q(ispin), 1.0_dp, mp2_env%ri_grad%L_jb(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor, &
                            mo_coeff(ispin), LL2_mu_q(ispin), 1.0_dp, mp2_env%ri_grad%L_jb(ispin), &
                            a_first_col=1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=1, &
                            c_first_row=1)
         CALL parallel_gemm('T', 'N', homo(ispin), virtual(ispin), nao, 2.0_dp*factor*(SQRT(0.5_dp) - 1.0_dp), &
                            mo_coeff(ispin), LL2_mu_q(ispin), 1.0_dp, mp2_env%ri_grad%L_jb(ispin), &
                            a_first_col=virtual_offset + 1, &
                            a_first_row=1, &
                            b_first_col=virtual_offset + 1, &
                            b_first_row=1, &
                            c_first_col=virtual_offset + 1, &
                            c_first_row=1)

         ! finally release LL1_mu_q
         CALL timestop(handle2)
         CALL cp_fm_release(LL1_mu_q(ispin))
      END DO
      DEALLOCATE (LL1_mu_q)

      CALL timestop(handle)

   END SUBROUTINE create_W_P_FT

! **************************************************************************************************
!> \brief ...
!> \param blacs_env ...
!> \param mo_coeff ...
!> \param para_env ...
!> \param para_env_sub ...
!> \param blacs_env_sub ...
!> \param Lag_mu_i ...
!> \param Lag_nu_a ...
!> \param L_mu_q ...
! **************************************************************************************************
   SUBROUTINE merge_Lagrangian_matrices(blacs_env, mo_coeff, para_env, para_env_sub, blacs_env_sub, &
                                        Lag_mu_i, Lag_nu_a, L_mu_q)
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env
      TYPE(cp_fm_type), DIMENSION(:), INTENT(IN)         :: mo_coeff
      TYPE(mp_para_env_type), POINTER                    :: para_env, para_env_sub
      TYPE(cp_blacs_env_type), POINTER                   :: blacs_env_sub
      TYPE(dbcsr_type), ALLOCATABLE, DIMENSION(:)        :: Lag_mu_i, Lag_nu_a
      TYPE(cp_fm_type), ALLOCATABLE, DIMENSION(:)        :: L_mu_q

      CHARACTER(LEN=*), PARAMETER :: routineN = 'merge_Lagrangian_matrices'

      INTEGER :: color_exchange, dummy_proc, handle, handle2, handle3, homo, i_global, i_local, &
         iiB, iii, iproc, ispin, j_global, j_local, jjB, max_col_size, max_row_size, mypcol, &
         myprow, nao, ncol_local, ncol_local_1i, ncol_local_2a, nmo, npcol, nprow, nrow_local, &
         nrow_local_1i, nrow_local_2a, nspins, number_of_rec, number_of_send, proc_receive, &
         proc_receive_static, proc_send, proc_send_ex, proc_send_static, proc_send_sub, &
         proc_shift, rec_col_size, rec_counter, rec_row_size, send_counter, send_pcol, send_prow, &
         size_rec_buffer, size_send_buffer, virtual, virtual_offset
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: iii_vet, map_rec_size, map_send_size, &
                                                            pos_info, pos_info_ex, proc_2_send_pos
      INTEGER, ALLOCATABLE, DIMENSION(:, :) :: grid_2_mepos, my_col_indeces_info_1i, &
         my_col_indeces_info_2a, my_row_indeces_info_1i, my_row_indeces_info_2a, sizes, sizes_1i, &
         sizes_2a
      INTEGER, ALLOCATABLE, DIMENSION(:, :, :)           :: col_indeces_info_1i, &
                                                            col_indeces_info_2a, &
                                                            row_indeces_info_1i, &
                                                            row_indeces_info_2a
      INTEGER, DIMENSION(:), POINTER                     :: col_indices_1i, col_indices_2a, &
                                                            row_indices_1i, row_indices_2a
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: mat_rec, mat_send
      TYPE(cp_fm_struct_type), POINTER                   :: fm_struct_tmp
      TYPE(cp_fm_type), ALLOCATABLE, DIMENSION(:)        :: L1_mu_i, L2_nu_a
      TYPE(integ_mat_buffer_type), ALLOCATABLE, &
         DIMENSION(:)                                    :: buffer_rec, buffer_send
      TYPE(integ_mat_buffer_type_2D), ALLOCATABLE, &
         DIMENSION(:)                                    :: buffer_cyclic
      TYPE(mp_para_env_type), POINTER                    :: para_env_exchange
      TYPE(mp_request_type), ALLOCATABLE, DIMENSION(:)   :: req_send

      CALL timeset(routineN, handle)

      nspins = SIZE(mo_coeff)

      ALLOCATE (L1_mu_i(nspins), L2_nu_a(nspins))
      DO ispin = 1, nspins
         CALL dbcsr_get_info(Lag_mu_i(ispin), nfullcols_total=homo, nfullrows_total=nao)
         ! Now we move from the local matrices to the global ones
         ! defined over all MPI tasks
         ! Start with moving from the DBCSR to FM for the lagrangians
         NULLIFY (fm_struct_tmp)
         CALL cp_fm_struct_create(fm_struct_tmp, para_env=para_env_sub, context=blacs_env_sub, &
                                  nrow_global=nao, ncol_global=homo)
         CALL cp_fm_create(L1_mu_i(ispin), fm_struct_tmp, name="Lag_mu_i")
         CALL cp_fm_struct_release(fm_struct_tmp)
         CALL cp_fm_set_all(L1_mu_i(ispin), 0.0_dp)
         CALL copy_dbcsr_to_fm(matrix=Lag_mu_i(ispin), fm=L1_mu_i(ispin))

         ! release Lag_mu_i
         CALL dbcsr_release(Lag_mu_i(ispin))

         CALL dbcsr_get_info(Lag_nu_a(ispin), nfullcols_total=virtual, nfullrows_total=nao)

         NULLIFY (fm_struct_tmp)
         CALL cp_fm_struct_create(fm_struct_tmp, para_env=para_env_sub, context=blacs_env_sub, &
                                  nrow_global=nao, ncol_global=virtual)
         CALL cp_fm_create(L2_nu_a(ispin), fm_struct_tmp, name="Lag_nu_a")
         CALL cp_fm_struct_release(fm_struct_tmp)
         CALL cp_fm_set_all(L2_nu_a(ispin), 0.0_dp)
         CALL copy_dbcsr_to_fm(matrix=Lag_nu_a(ispin), fm=L2_nu_a(ispin))

         ! release Lag_nu_a
         CALL dbcsr_release(Lag_nu_a(ispin))
      END DO
      DEALLOCATE (Lag_mu_i, Lag_nu_a)

      ALLOCATE (L_mu_q(nspins))

      DO ispin = 1, nspins
         CALL cp_fm_get_info(L2_nu_a(ispin), ncol_global=virtual)
         CALL cp_fm_get_info(mo_coeff(ispin), nrow_global=nao, ncol_global=nmo)

         virtual_offset = nmo - virtual

         NULLIFY (fm_struct_tmp)
         CALL cp_fm_struct_create(fm_struct_tmp, para_env=para_env, context=blacs_env, &
                                  nrow_global=nao, ncol_global=nmo)
         CALL cp_fm_create(L_mu_q(ispin), fm_struct_tmp, name="Lag_mu_q")
         CALL cp_fm_struct_release(fm_struct_tmp)
         CALL cp_fm_set_all(L_mu_q(ispin), 0.0_dp)

         ! create all information array
         ALLOCATE (pos_info(0:para_env%num_pe - 1))
         CALL para_env%allgather(para_env_sub%mepos, pos_info)

         ! get matrix information for the global
         CALL cp_fm_get_info(matrix=L_mu_q(ispin), &
                             nrow_local=nrow_local, &
                             ncol_local=ncol_local)
         myprow = L_mu_q(ispin)%matrix_struct%context%mepos(1)
         mypcol = L_mu_q(ispin)%matrix_struct%context%mepos(2)
         nprow = L_mu_q(ispin)%matrix_struct%context%num_pe(1)
         npcol = L_mu_q(ispin)%matrix_struct%context%num_pe(2)

         ALLOCATE (grid_2_mepos(0:nprow - 1, 0:npcol - 1))
         grid_2_mepos = 0
         grid_2_mepos(myprow, mypcol) = para_env%mepos
         CALL para_env%sum(grid_2_mepos)

         ! get matrix information for L1_mu_i
         CALL cp_fm_get_info(matrix=L1_mu_i(ispin), &
                             nrow_local=nrow_local_1i, &
                             ncol_local=ncol_local_1i, &
                             row_indices=row_indices_1i, &
                             col_indices=col_indices_1i)

         ALLOCATE (sizes_1i(2, 0:para_env_sub%num_pe - 1))
         CALL para_env_sub%allgather([nrow_local_1i, ncol_local_1i], sizes_1i)

         ! get matrix information for L2_nu_a
         CALL cp_fm_get_info(matrix=L2_nu_a(ispin), &
                             nrow_local=nrow_local_2a, &
                             ncol_local=ncol_local_2a, &
                             row_indices=row_indices_2a, &
                             col_indices=col_indices_2a)

         ALLOCATE (sizes_2a(2, 0:para_env_sub%num_pe - 1))
         CALL para_env_sub%allgather([nrow_local_2a, ncol_local_2a], sizes_2a)

         ! Here we perform a ring communication scheme taking into account
         ! for the sub-group distribution of the source matrices.
         ! as a first step we need to redistribute the data within
         ! the subgroup.
         ! In order to do so we have to allocate the structure
         ! that will hold the local data involved in communication, this
         ! structure will be the same for processes in different subgroups
         ! sharing the same position in the subgroup.
         ! -1) create the exchange para_env
         color_exchange = para_env_sub%mepos
         ALLOCATE (para_env_exchange)
         CALL para_env_exchange%from_split(para_env, color_exchange)
         ALLOCATE (pos_info_ex(0:para_env%num_pe - 1))
         CALL para_env%allgather(para_env_exchange%mepos, pos_info_ex)
         ALLOCATE (sizes(2, 0:para_env_exchange%num_pe - 1))
         CALL para_env_exchange%allgather([nrow_local, ncol_local], sizes)

         ! 0) store some info about indeces of the fm matrices (subgroup)
         CALL timeset(routineN//"_inx", handle2)
         ! matrix L1_mu_i
         max_row_size = MAXVAL(sizes_1i(1, :))
         max_col_size = MAXVAL(sizes_1i(2, :))
         ALLOCATE (row_indeces_info_1i(2, max_row_size, 0:para_env_sub%num_pe - 1))
         ALLOCATE (col_indeces_info_1i(2, max_col_size, 0:para_env_sub%num_pe - 1))
         ALLOCATE (my_row_indeces_info_1i(2, max_row_size))
         ALLOCATE (my_col_indeces_info_1i(2, max_col_size))
         row_indeces_info_1i = 0
         col_indeces_info_1i = 0
         dummy_proc = 0
         ! row
         DO iiB = 1, nrow_local_1i
            i_global = row_indices_1i(iiB)
            send_prow = L_mu_q(ispin)%matrix_struct%g2p_row(i_global)
            i_local = L_mu_q(ispin)%matrix_struct%g2l_row(i_global)
            my_row_indeces_info_1i(1, iiB) = send_prow
            my_row_indeces_info_1i(2, iiB) = i_local
         END DO
         ! col
         DO jjB = 1, ncol_local_1i
            j_global = col_indices_1i(jjB)
            send_pcol = L_mu_q(ispin)%matrix_struct%g2p_col(j_global)
            j_local = L_mu_q(ispin)%matrix_struct%g2l_col(j_global)
            my_col_indeces_info_1i(1, jjB) = send_pcol
            my_col_indeces_info_1i(2, jjB) = j_local
         END DO
         CALL para_env_sub%allgather(my_row_indeces_info_1i, row_indeces_info_1i)
         CALL para_env_sub%allgather(my_col_indeces_info_1i, col_indeces_info_1i)
         DEALLOCATE (my_row_indeces_info_1i, my_col_indeces_info_1i)

         ! matrix L2_nu_a
         max_row_size = MAXVAL(sizes_2a(1, :))
         max_col_size = MAXVAL(sizes_2a(2, :))
         ALLOCATE (row_indeces_info_2a(2, max_row_size, 0:para_env_sub%num_pe - 1))
         ALLOCATE (col_indeces_info_2a(2, max_col_size, 0:para_env_sub%num_pe - 1))
         ALLOCATE (my_row_indeces_info_2a(2, max_row_size))
         ALLOCATE (my_col_indeces_info_2a(2, max_col_size))
         row_indeces_info_2a = 0
         col_indeces_info_2a = 0
         ! row
         DO iiB = 1, nrow_local_2a
            i_global = row_indices_2a(iiB)
            send_prow = L_mu_q(ispin)%matrix_struct%g2p_row(i_global)
            i_local = L_mu_q(ispin)%matrix_struct%g2l_row(i_global)
            my_row_indeces_info_2a(1, iiB) = send_prow
            my_row_indeces_info_2a(2, iiB) = i_local
         END DO
         ! col
         DO jjB = 1, ncol_local_2a
            j_global = col_indices_2a(jjB) + virtual_offset
            send_pcol = L_mu_q(ispin)%matrix_struct%g2p_col(j_global)
            j_local = L_mu_q(ispin)%matrix_struct%g2l_col(j_global)
            my_col_indeces_info_2a(1, jjB) = send_pcol
            my_col_indeces_info_2a(2, jjB) = j_local
         END DO
         CALL para_env_sub%allgather(my_row_indeces_info_2a, row_indeces_info_2a)
         CALL para_env_sub%allgather(my_col_indeces_info_2a, col_indeces_info_2a)
         DEALLOCATE (my_row_indeces_info_2a, my_col_indeces_info_2a)
         CALL timestop(handle2)

         ! 1) define the map for sending data in the subgroup starting with L1_mu_i
         CALL timeset(routineN//"_subinfo", handle2)
         ALLOCATE (map_send_size(0:para_env_sub%num_pe - 1))
         map_send_size = 0
         DO jjB = 1, ncol_local_1i
            send_pcol = col_indeces_info_1i(1, jjB, para_env_sub%mepos)
            DO iiB = 1, nrow_local_1i
               send_prow = row_indeces_info_1i(1, iiB, para_env_sub%mepos)
               proc_send = grid_2_mepos(send_prow, send_pcol)
               proc_send_sub = pos_info(proc_send)
               map_send_size(proc_send_sub) = map_send_size(proc_send_sub) + 1
            END DO
         END DO
         ! and the same for L2_nu_a
         DO jjB = 1, ncol_local_2a
            send_pcol = col_indeces_info_2a(1, jjB, para_env_sub%mepos)
            DO iiB = 1, nrow_local_2a
               send_prow = row_indeces_info_2a(1, iiB, para_env_sub%mepos)
               proc_send = grid_2_mepos(send_prow, send_pcol)
               proc_send_sub = pos_info(proc_send)
               map_send_size(proc_send_sub) = map_send_size(proc_send_sub) + 1
            END DO
         END DO
         ! and exchange data in order to create map_rec_size
         ALLOCATE (map_rec_size(0:para_env_sub%num_pe - 1))
         map_rec_size = 0
         CALL para_env_sub%alltoall(map_send_size, map_rec_size, 1)
         CALL timestop(handle2)

         ! 2) reorder data in sending buffer
         CALL timeset(routineN//"_sub_Bsend", handle2)
         ! count the number of messages (include myself)
         number_of_send = 0
         DO proc_shift = 0, para_env_sub%num_pe - 1
            proc_send = MODULO(para_env_sub%mepos + proc_shift, para_env_sub%num_pe)
            IF (map_send_size(proc_send) > 0) THEN
               number_of_send = number_of_send + 1
            END IF
         END DO
         ! allocate the structure that will hold the messages to be sent
         ALLOCATE (buffer_send(number_of_send))
         send_counter = 0
         ALLOCATE (proc_2_send_pos(0:para_env_sub%num_pe - 1))
         proc_2_send_pos = 0
         DO proc_shift = 0, para_env_sub%num_pe - 1
            proc_send = MODULO(para_env_sub%mepos + proc_shift, para_env_sub%num_pe)
            size_send_buffer = map_send_size(proc_send)
            IF (map_send_size(proc_send) > 0) THEN
               send_counter = send_counter + 1
               ! allocate the sending buffer (msg)
               ALLOCATE (buffer_send(send_counter)%msg(size_send_buffer))
               buffer_send(send_counter)%msg = 0.0_dp
               buffer_send(send_counter)%proc = proc_send
               proc_2_send_pos(proc_send) = send_counter
            END IF
         END DO
         ! loop over the locally held data and fill the buffer_send
         ! for doing that we need an array that keep track if the
         ! sequential increase of the index for each message
         ALLOCATE (iii_vet(number_of_send))
         iii_vet = 0
         DO jjB = 1, ncol_local_1i
            send_pcol = col_indeces_info_1i(1, jjB, para_env_sub%mepos)
            DO iiB = 1, nrow_local_1i
               send_prow = row_indeces_info_1i(1, iiB, para_env_sub%mepos)
               proc_send = grid_2_mepos(send_prow, send_pcol)
               proc_send_sub = pos_info(proc_send)
               send_counter = proc_2_send_pos(proc_send_sub)
               iii_vet(send_counter) = iii_vet(send_counter) + 1
               iii = iii_vet(send_counter)
               buffer_send(send_counter)%msg(iii) = L1_mu_i(ispin)%local_data(iiB, jjB)
            END DO
         END DO
         ! release the local data of L1_mu_i
         DEALLOCATE (L1_mu_i(ispin)%local_data)
         ! and the same for L2_nu_a
         DO jjB = 1, ncol_local_2a
            send_pcol = col_indeces_info_2a(1, jjB, para_env_sub%mepos)
            DO iiB = 1, nrow_local_2a
               send_prow = row_indeces_info_2a(1, iiB, para_env_sub%mepos)
               proc_send = grid_2_mepos(send_prow, send_pcol)
               proc_send_sub = pos_info(proc_send)
               send_counter = proc_2_send_pos(proc_send_sub)
               iii_vet(send_counter) = iii_vet(send_counter) + 1
               iii = iii_vet(send_counter)
               buffer_send(send_counter)%msg(iii) = L2_nu_a(ispin)%local_data(iiB, jjB)
            END DO
         END DO
         DEALLOCATE (L2_nu_a(ispin)%local_data)
         DEALLOCATE (proc_2_send_pos)
         DEALLOCATE (iii_vet)
         CALL timestop(handle2)

         ! 3) create the buffer for receive, post the message with irecv
         !    and send the messages non-blocking
         CALL timeset(routineN//"_sub_isendrecv", handle2)
         ! count the number of messages to be received
         number_of_rec = 0
         DO proc_shift = 0, para_env_sub%num_pe - 1
            proc_receive = MODULO(para_env_sub%mepos - proc_shift, para_env_sub%num_pe)
            IF (map_rec_size(proc_receive) > 0) THEN
               number_of_rec = number_of_rec + 1
            END IF
         END DO
         ALLOCATE (buffer_rec(number_of_rec))
         rec_counter = 0
         DO proc_shift = 0, para_env_sub%num_pe - 1
            proc_receive = MODULO(para_env_sub%mepos - proc_shift, para_env_sub%num_pe)
            size_rec_buffer = map_rec_size(proc_receive)
            IF (map_rec_size(proc_receive) > 0) THEN
               rec_counter = rec_counter + 1
               ! prepare the buffer for receive
               ALLOCATE (buffer_rec(rec_counter)%msg(size_rec_buffer))
               buffer_rec(rec_counter)%msg = 0.0_dp
               buffer_rec(rec_counter)%proc = proc_receive
               ! post the message to be received (not need to send to myself)
               IF (proc_receive /= para_env_sub%mepos) THEN
                  CALL para_env_sub%irecv(buffer_rec(rec_counter)%msg, proc_receive, &
                                          buffer_rec(rec_counter)%msg_req)
               END IF
            END IF
         END DO
         ! send messages
         ALLOCATE (req_send(number_of_send))
         req_send = mp_request_null
         send_counter = 0
         DO proc_shift = 0, para_env_sub%num_pe - 1
            proc_send = MODULO(para_env_sub%mepos + proc_shift, para_env_sub%num_pe)
            IF (map_send_size(proc_send) > 0) THEN
               send_counter = send_counter + 1
               IF (proc_send == para_env_sub%mepos) THEN
                  buffer_rec(send_counter)%msg(:) = buffer_send(send_counter)%msg
               ELSE
                  CALL para_env_sub%isend(buffer_send(send_counter)%msg, proc_send, &
                                          buffer_send(send_counter)%msg_req)
                  req_send(send_counter) = buffer_send(send_counter)%msg_req
               END IF
            END IF
         END DO
         DEALLOCATE (map_send_size)
         CALL timestop(handle2)

         ! 4) (if memory is a problem we should move this part after point 5)
         !    Here we create the new buffer for cyclic(ring) communication and
         !    we fill it with the data received from the other member of the
         !    subgroup
         CALL timeset(routineN//"_Bcyclic", handle2)
         ! first allocata new structure
         ALLOCATE (buffer_cyclic(0:para_env_exchange%num_pe - 1))
         DO iproc = 0, para_env_exchange%num_pe - 1
            rec_row_size = sizes(1, iproc)
            rec_col_size = sizes(2, iproc)
            ALLOCATE (buffer_cyclic(iproc)%msg(rec_row_size, rec_col_size))
            buffer_cyclic(iproc)%msg = 0.0_dp
         END DO
         ! now collect data from other member of the subgroup and fill
         ! buffer_cyclic
         rec_counter = 0
         DO proc_shift = 0, para_env_sub%num_pe - 1
            proc_receive = MODULO(para_env_sub%mepos - proc_shift, para_env_sub%num_pe)
            size_rec_buffer = map_rec_size(proc_receive)
            IF (map_rec_size(proc_receive) > 0) THEN
               rec_counter = rec_counter + 1

               ! wait for the message
               IF (proc_receive /= para_env_sub%mepos) CALL buffer_rec(rec_counter)%msg_req%wait()

               CALL timeset(routineN//"_fill", handle3)
               iii = 0
               DO jjB = 1, sizes_1i(2, proc_receive)
                  send_pcol = col_indeces_info_1i(1, jjB, proc_receive)
                  j_local = col_indeces_info_1i(2, jjB, proc_receive)
                  DO iiB = 1, sizes_1i(1, proc_receive)
                     send_prow = row_indeces_info_1i(1, iiB, proc_receive)
                     proc_send = grid_2_mepos(send_prow, send_pcol)
                     proc_send_sub = pos_info(proc_send)
                     IF (proc_send_sub /= para_env_sub%mepos) CYCLE
                     iii = iii + 1
                     i_local = row_indeces_info_1i(2, iiB, proc_receive)
                     proc_send_ex = pos_info_ex(proc_send)
                     buffer_cyclic(proc_send_ex)%msg(i_local, j_local) = &
                        buffer_cyclic(proc_send_ex)%msg(i_local, j_local) &
                        + buffer_rec(rec_counter)%msg(iii)
                  END DO
               END DO
               ! and the same for L2_nu_a
               DO jjB = 1, sizes_2a(2, proc_receive)
                  send_pcol = col_indeces_info_2a(1, jjB, proc_receive)
                  j_local = col_indeces_info_2a(2, jjB, proc_receive)
                  DO iiB = 1, sizes_2a(1, proc_receive)
                     send_prow = row_indeces_info_2a(1, iiB, proc_receive)
                     proc_send = grid_2_mepos(send_prow, send_pcol)
                     proc_send_sub = pos_info(proc_send)
                     IF (proc_send_sub /= para_env_sub%mepos) CYCLE
                     iii = iii + 1
                     i_local = row_indeces_info_2a(2, iiB, proc_receive)
                     proc_send_ex = pos_info_ex(proc_send)
                     buffer_cyclic(proc_send_ex)%msg(i_local, j_local) = &
                        buffer_cyclic(proc_send_ex)%msg(i_local, j_local) &
                        + buffer_rec(rec_counter)%msg(iii)
                  END DO
               END DO
               CALL timestop(handle3)

               ! deallocate the received message
               DEALLOCATE (buffer_rec(rec_counter)%msg)
            END IF
         END DO
         DEALLOCATE (row_indeces_info_1i)
         DEALLOCATE (col_indeces_info_1i)
         DEALLOCATE (row_indeces_info_2a)
         DEALLOCATE (col_indeces_info_2a)
         DEALLOCATE (buffer_rec)
         DEALLOCATE (map_rec_size)
         CALL timestop(handle2)

         ! 5)  Wait for all messeges to be sent in the subgroup
         CALL timeset(routineN//"_sub_waitall", handle2)
         CALL mp_waitall(req_send(:))
         DO send_counter = 1, number_of_send
            DEALLOCATE (buffer_send(send_counter)%msg)
         END DO
         DEALLOCATE (buffer_send)
         DEALLOCATE (req_send)
         CALL timestop(handle2)

         ! 6) Start with ring communication
         CALL timeset(routineN//"_ring", handle2)
         proc_send_static = MODULO(para_env_exchange%mepos + 1, para_env_exchange%num_pe)
         proc_receive_static = MODULO(para_env_exchange%mepos - 1, para_env_exchange%num_pe)
         max_row_size = MAXVAL(sizes(1, :))
         max_col_size = MAXVAL(sizes(2, :))
         ALLOCATE (mat_send(max_row_size, max_col_size))
         ALLOCATE (mat_rec(max_row_size, max_col_size))
         mat_send = 0.0_dp
         mat_send(1:nrow_local, 1:ncol_local) = buffer_cyclic(para_env_exchange%mepos)%msg(:, :)
         DEALLOCATE (buffer_cyclic(para_env_exchange%mepos)%msg)
         DO proc_shift = 1, para_env_exchange%num_pe - 1
            proc_receive = MODULO(para_env_exchange%mepos - proc_shift, para_env_exchange%num_pe)

            rec_row_size = sizes(1, proc_receive)
            rec_col_size = sizes(2, proc_receive)

            mat_rec = 0.0_dp
            CALL para_env_exchange%sendrecv(mat_send, proc_send_static, &
                                            mat_rec, proc_receive_static)

            mat_send = 0.0_dp
            mat_send(1:rec_row_size, 1:rec_col_size) = mat_rec(1:rec_row_size, 1:rec_col_size) + &
                                                       buffer_cyclic(proc_receive)%msg(:, :)

            DEALLOCATE (buffer_cyclic(proc_receive)%msg)
         END DO
         ! and finally
         CALL para_env_exchange%sendrecv(mat_send, proc_send_static, &
                                         mat_rec, proc_receive_static)
         L_mu_q(ispin)%local_data(1:nrow_local, 1:ncol_local) = mat_rec(1:nrow_local, 1:ncol_local)
         DEALLOCATE (buffer_cyclic)
         DEALLOCATE (mat_send)
         DEALLOCATE (mat_rec)
         CALL timestop(handle2)

         ! release para_env_exchange
         CALL mp_para_env_release(para_env_exchange)

         CALL cp_fm_release(L1_mu_i(ispin))
         CALL cp_fm_release(L2_nu_a(ispin))
         DEALLOCATE (pos_info_ex)
         DEALLOCATE (grid_2_mepos)
         DEALLOCATE (sizes)
         DEALLOCATE (sizes_1i)
         DEALLOCATE (sizes_2a)

         DEALLOCATE (pos_info)
      END DO

      CALL timestop(handle)

   END SUBROUTINE merge_Lagrangian_matrices

END MODULE mp2_ri_grad
