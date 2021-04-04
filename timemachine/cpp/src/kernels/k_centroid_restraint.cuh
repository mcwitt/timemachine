
#include "k_fixed_point.hpp"


__device__ const double PI = 3.14159265358979323846;

template<typename RealType>
void __global__ k_centroid_restraint(
    const int N,     // number of bonds
    const double *coords,  // [n, 3]
    const int *group_a_idxs,
    const int *group_b_idxs,
    const int N_A,
    const int N_B,
    // const double *masses, // ignore masses for now
    const double kb,
    const double b0,
    unsigned long long *grad_coords,
    double *energy) {

    // (ytz): ultra shitty inefficient algorithm, optimize later
    const auto t_idx = blockDim.x*blockIdx.x + threadIdx.x;

    if(t_idx != 0) {
        return;
    }

    double group_a_ctr[3] = {0};
    // double mass_a_sums[3] = {0};
    for(int d=0; d < 3; d++) {
        for(int i=0; i < N_A; i++) {
            // double mass_i = masses[group_a_idxs[i]];
            group_a_ctr[d] += coords[group_a_idxs[i]*3+d];
            // mass_a_sums[d] += mass_i;
        }
        group_a_ctr[d] /= N_A[d];
    }

    double group_b_ctr[3] = {0};
    double mass_b_sums[3] = {0};
    for(int d=0; d < 3; d++) {

        for(int i=0; i < N_B; i++) {
            // double mass_i = masses[group_b_idxs[i]];
            group_b_ctr[d] += coords[group_b_idxs[i]*3+d];
            // mass_b_sums[d] += mass_i;
        }
        group_b_ctr[d] /= N_B;
    }

    double dx = group_a_ctr[0] - group_b_ctr[0];
    double dy = group_a_ctr[1] - group_b_ctr[1];
    double dz = group_a_ctr[2] - group_b_ctr[2];
    double dij = sqrt(dx*dx + dy*dy + dz*dz);
    double nrg = kb*(dij-b0)*(dij-b0);

    if(energy) {
        atomicAdd(energy, FLOAT_TO_FIXED<RealType>(nrg));
    }

    double du_ddij = 2*kb*(dij-b0);

    // grads
    if(grad_coords) {
        for(int d=0; d < 3; d++) {
            double ddij_dxi = (group_a_ctr[d] - group_b_ctr[d])/dij;

            for(int i=0; i < N_A; i++) {
                // double mass_i = masses[group_a_idxs[i]];
                double dx = du_ddij*ddij_dxi;
                atomicAdd(grad_coords + group_a_idxs[i]*3 + d, FLOAT_TO_FIXED<RealType>(dx));
            }
            for(int i=0; i < N_B; i++) {
                // double mass_i = masses[group_b_idxs[i]];
                double dx = -du_ddij*ddij_dxi;
                atomicAdd(grad_coords + group_b_idxs[i]*3 + d, FLOAT_TO_FIXED<RealType>(dx));
            }
        }
    }

}
