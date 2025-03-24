/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "common/grid_common.h"
#include "common/grid_mpi.h"
#include "grid_fft_grid.h"
#include "grid_multigrid.h"
#include "grid_multigrid_test.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Function to test the Multigrid backend.
 * \author Frederick Stein
 ******************************************************************************/
int multigrid_test() {
  const int npts_global[2][3] = {{4, 4, 4}, {2, 2, 2}};
  const int npts_local[2][3] = {{4, 4, 4}, {2, 2, 2}};
  const int shift_local[2][3] = {{-2, -2, -2}, {-1, -1, -1}};
  const int border_width[2][3] = {{2, 2, 2}, {1, 1, 1}};
  const double dh[2][3][3] = {
      {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
      {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
  const double dh_inv[2][3][3] = {
      {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
      {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
  const int pgrid_dims[2][3] = {{1, 1, 1}, {1, 1, 1}};
  grid_multigrid *multigrid = NULL;
  grid_create_multigrid(true, 2, npts_global, npts_local, shift_local,
                        border_width, dh, dh_inv, pgrid_dims,
                        grid_mpi_comm_self, &multigrid);
  for (int level = 0; level < multigrid->nlevels; level++) {
    assert(multigrid->fft_grid_layouts[level]->grid_id > 0);
  }

  double grid[64];
  memset(grid, 0, 64 * sizeof(double));

  // grid_copy_to_multigrid_single(multigrid, grid);
  // grid_copy_from_multigrid_single(multigrid, grid);

  grid_free_multigrid(multigrid);

  return 0;
}

// EOF
