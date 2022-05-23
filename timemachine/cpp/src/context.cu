#include "context.hpp"
#include "fanout_summed_potential.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "neighborlist.hpp"
#include "nonbonded_all_pairs.hpp"
#include "set_utils.hpp"
#include "summed_potential.hpp"
#include <cub/cub.cuh>
#include <memory>
#include <typeinfo>

namespace timemachine {

struct LessThan {
    int compare;
    CUB_RUNTIME_FUNCTION __device__ __forceinline__ explicit LessThan(int compare) : compare(compare) {}
    CUB_RUNTIME_FUNCTION __device__ __forceinline__ bool operator()(const int &a) const { return (a < compare); }
};

Context::Context(
    int N,
    const double *x_0,
    const double *v_0,
    const double *box_0,
    Integrator *intg,
    std::vector<BoundPotential *> bps,
    MonteCarloBarostat *barostat)
    : N_(N), barostat_(barostat), step_(0), d_sum_storage_(nullptr), d_sum_storage_bytes_(0), intg_(intg), bps_(bps) {
    d_x_t_ = gpuErrchkCudaMallocAndCopy(x_0, N * 3);
    d_v_t_ = gpuErrchkCudaMallocAndCopy(v_0, N * 3);
    d_box_t_ = gpuErrchkCudaMallocAndCopy(box_0, 3 * 3);
    gpuErrchk(cudaMalloc(&d_du_dx_t_, N * 3 * sizeof(*d_du_dx_t_)));
    gpuErrchk(cudaMalloc(&d_du_dl_buffer_, N * sizeof(*d_du_dl_buffer_)));
    gpuErrchk(cudaMalloc(&d_u_buffer_, N * sizeof(*d_u_buffer_)));

    unsigned long long *d_in_tmp = nullptr;  // dummy
    unsigned long long *d_out_tmp = nullptr; // dummy

    // Compute the storage size necessary to reduce du_dl
    cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_in_tmp, d_out_tmp, N_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMalloc(&d_sum_storage_, d_sum_storage_bytes_));

    // for(int i=0; i < bps.size(); i++) {
    // cudaStream_t stream;
    // gpuErrchk(cudaStreamCreate(&stream));
    // streams_.push_back(stream);
    // }
};

Context::~Context() {
    gpuErrchk(cudaFree(d_x_t_));
    gpuErrchk(cudaFree(d_v_t_));
    gpuErrchk(cudaFree(d_box_t_));
    gpuErrchk(cudaFree(d_du_dx_t_));
    gpuErrchk(cudaFree(d_du_dl_buffer_));
    gpuErrchk(cudaFree(d_u_buffer_));
    gpuErrchk(cudaFree(d_sum_storage_));

    // for(int i=0; i < streams_.size(); i++) {
    // gpuErrchk(cudaStreamDestroy(streams_[i]));
    // }
};

void __global__ k_flatten_nblist(
    const int N, // Number of values in src
    const int K, // Number of values in dest
    const unsigned int *__restrict__ src,
    unsigned int *__restrict__ dest) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    const unsigned int val = src[idx];
    if (val > K) {
        return;
    }
    dest[val] = val;
    // flag_map[val] = 1;
}

// Any value that is >=N becomes the idx and any value that is an idx becomes N
void __global__ k_invert_indices(const int N, unsigned int *__restrict__ arr) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    arr[idx] = arr[idx] >= N ? idx : N;
}

bool is_nonbonded_potential(std::shared_ptr<Potential> pot) {
    if (std::shared_ptr<NonbondedAllPairs<float, true>> nb_pot =
            std::dynamic_pointer_cast<NonbondedAllPairs<float, true>>(pot);
        nb_pot) {
        return true;
    } else if (std::shared_ptr<NonbondedAllPairs<float, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<float, false>>(pot);
               nb_pot) {
        return true;
    } else if (std::shared_ptr<NonbondedAllPairs<double, true>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, true>>(pot);
               nb_pot) {
        return true;
    } else if (std::shared_ptr<NonbondedAllPairs<double, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, false>>(pot);
               nb_pot) {
        return true;
    }
    return false;
}

void set_nonbonded_potential_idxs(
    std::shared_ptr<Potential> pot, const int num_idxs, const unsigned int *d_idxs, const cudaStream_t stream) {
    if (std::shared_ptr<NonbondedAllPairs<float, true>> nb_pot =
            std::dynamic_pointer_cast<NonbondedAllPairs<float, true>>(pot);
        nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else if (std::shared_ptr<NonbondedAllPairs<float, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<float, false>>(pot);
               nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else if (std::shared_ptr<NonbondedAllPairs<double, true>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, true>>(pot);
               nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else if (std::shared_ptr<NonbondedAllPairs<double, false>> nb_pot =
                   std::dynamic_pointer_cast<NonbondedAllPairs<double, false>>(pot);
               nb_pot) {
        nb_pot->set_atom_idxs_device(num_idxs, d_idxs, stream);
    } else {
        throw std::runtime_error("Unable to cast potential to NonbondedAllPairs");
    }
}

// TODO document this regarding store_x_interval being based on iterations
std::array<std::vector<double>, 2> Context::local_md(
    const std::vector<double> &lambda_schedule,
    const int iterations,
    const int global_steps,
    const int local_steps,
    const int store_x_interval,
    const std::vector<unsigned int> &local_idxs,
    const double cutoff) {
    if (store_x_interval <= 0) {
        throw std::runtime_error("store_x_interval <= 0");
    }
    const int x_buffer_size = iterations / store_x_interval;

    const int box_buffer_size = x_buffer_size * 3 * 3;
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    // Construct neighborlist to find the inner and outer sphere
    Neighborlist<float> nblist(N_);

    std::set<unsigned int> unique_idxs(local_idxs.begin(), local_idxs.end());
    std::vector<unsigned int> non_local_idxs = get_indices_difference(static_cast<size_t>(N_), unique_idxs);

    int NR = local_idxs.size();
    int NC = non_local_idxs.size();

    const size_t tpb = 32;
    int max_interactions = nblist.num_column_blocks() * nblist.num_row_blocks() * tpb;

    DeviceBuffer<unsigned int> d_sphere_idxs_inner(N_);
    DeviceBuffer<unsigned int> d_sphere_idxs_outer(N_);

    // Store coordinates in host memory as it can be very large
    std::vector<double> h_x_buffer(x_buffer_size * N_ * 3);
    // Store boxes on GPU as boxes are a constant size and relatively small1
    DeviceBuffer<double> d_box_buffer(box_buffer_size);

    DeviceBuffer<unsigned int> d_init_row_idxs(local_idxs.size());
    d_init_row_idxs.copy_from(&local_idxs[0]);
    DeviceBuffer<unsigned int> d_init_col_idxs(non_local_idxs.size());
    d_init_col_idxs.copy_from(&non_local_idxs[0]);

    DeviceBuffer<unsigned int> d_row_idxs(N_);
    DeviceBuffer<unsigned int> d_col_idxs(N_);

    std::vector<unsigned int> atom_idxs_h = std::vector<unsigned int>(N_);
    std::iota(atom_idxs_h.begin(), atom_idxs_h.end(), 0);
    DeviceBuffer<unsigned int> d_identity_idxs(N_);
    d_identity_idxs.copy_from(&atom_idxs_h[0]);

    // Potentials we need
    // Bonded/torsions
    // All exclusions
    // Nonbonded all pairs within inner
    // Ixn group with inner set

    std::vector<BoundPotential *> local_bps;

    // Copy assignment, so we can modify it
    local_bps = bps_;

    // Need to keep references to the bound potential
    std::vector<BoundPotential> ref_bps;

    std::shared_ptr<Potential> nonbonded_potential;

    // Flatten out potentials into bound potentials for local_bps when applicable
    // Traverse backways to pop off the potentials that are fanned out
    for (int i = bps_.size() - 1; i >= 0; i--) {
        std::shared_ptr<FanoutSummedPotential> fanned_potential =
            std::dynamic_pointer_cast<FanoutSummedPotential>(bps_[i]->potential);
        if (fanned_potential != nullptr) {
            std::vector<std::shared_ptr<Potential>> potentials = fanned_potential->get_potentials();
            for (std::shared_ptr<Potential> pot : fanned_potential->get_potentials()) {
                if (is_nonbonded_potential(pot)) {
                    nonbonded_potential = pot;
                    break;
                }
            }
            continue;
        }
        std::shared_ptr<SummedPotential> summed_potential =
            std::dynamic_pointer_cast<SummedPotential>(bps_[i]->potential);
        if (summed_potential != nullptr) {
            for (std::shared_ptr<Potential> pot : summed_potential->get_potentials()) {
                if (is_nonbonded_potential(pot)) {
                    nonbonded_potential = pot;
                    break;
                }
            }
            continue;
        }
    }
    if (!nonbonded_potential) {
        throw std::runtime_error("Unable to find a NonbondedAllPairs potential");
    }

    int *p_num_selected;
    gpuErrchk(cudaMallocHost(&p_num_selected, 1 * sizeof(*p_num_selected)));
    DeviceBuffer<int> num_selected_buffer(1);
    LessThan select_op(N_);

    std::size_t temp_storage_bytes = 0;
    cub::DevicePartition::If(
        nullptr,
        temp_storage_bytes,
        d_sphere_idxs_inner.data,
        d_row_idxs.data,
        num_selected_buffer.data,
        N_,
        select_op);
    DeviceBuffer<char> d_temp_storage_buffer(temp_storage_bytes);

    for (int i = 1; i <= iterations; i++) {

        // Use the provided bound potentials as is
        for (int j = 0; j < global_steps; j++) {
            this->_step(bps_, lambda_schedule[j], nullptr, nullptr, stream);
        }
        NR = local_idxs.size();
        NC = non_local_idxs.size();

        nblist.set_idxs_device(NC, NR, d_init_col_idxs.data, d_init_row_idxs.data, stream);

        max_interactions = nblist.num_column_blocks() * nblist.num_row_blocks() * tpb;
        // Build the neighborlist around the idxs to get the atoms that interact
        nblist.build_nblist_device(N_, d_x_t_, d_box_t_, cutoff, stream);

        // Set the array to all N, which means it will be ignored as an idx
        k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_sphere_idxs_inner.data, N_);
        gpuErrchk(cudaPeekAtLastError());
        // Fill inner sphere with ixn atoms + atoms used to build nblist
        k_flatten_nblist<<<ceil_divide(max_interactions, tpb), tpb, 0, stream>>>(
            max_interactions, N_, nblist.get_ixn_atoms(), d_sphere_idxs_inner.data);
        gpuErrchk(cudaPeekAtLastError());

        // Add the row indices to the indices that will be modified
        k_flatten_nblist<<<ceil_divide(NR, tpb), tpb, 0, stream>>>(
            NR, N_, d_init_row_idxs.data, d_sphere_idxs_inner.data);
        gpuErrchk(cudaPeekAtLastError());

        // Partition the valid row indices to the front of the array
        cub::DevicePartition::If(
            d_temp_storage_buffer.data,
            temp_storage_bytes,
            d_sphere_idxs_inner.data,
            d_row_idxs.data,
            num_selected_buffer.data,
            N_,
            select_op,
            stream);
        gpuErrchk(cudaPeekAtLastError());

        // Copy the num out, that is the new NR, NC == N_ - NR
        gpuErrchk(cudaMemcpyAsync(
            p_num_selected, num_selected_buffer.data, 1 * sizeof(*p_num_selected), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));

        NR = p_num_selected[0];
        NC = N_ - NR;

        k_invert_indices<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_sphere_idxs_inner.data);
        gpuErrchk(cudaPeekAtLastError());

        // Partition the col idx to the front
        cub::DevicePartition::If(
            d_temp_storage_buffer.data,
            temp_storage_bytes,
            d_sphere_idxs_inner.data,
            d_col_idxs.data,
            num_selected_buffer.data,
            N_,
            select_op,
            stream);
        gpuErrchk(cudaPeekAtLastError());

        k_invert_indices<<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_sphere_idxs_inner.data);
        gpuErrchk(cudaPeekAtLastError());

        nblist.set_idxs_device(NC, NR, d_col_idxs.data, d_row_idxs.data, stream);
        max_interactions = nblist.num_column_blocks() * nblist.num_row_blocks() * tpb;
        // Build the neighborlist around the idxs to get the atoms that interact
        nblist.build_nblist_device(N_, d_x_t_, d_box_t_, cutoff, stream);

        k_initialize_array<unsigned int><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(N_, d_sphere_idxs_outer.data, N_);
        gpuErrchk(cudaPeekAtLastError());

        k_flatten_nblist<<<ceil_divide(max_interactions, tpb), tpb, 0, stream>>>(
            max_interactions, N_, nblist.get_ixn_atoms(), d_sphere_idxs_outer.data);
        gpuErrchk(cudaPeekAtLastError());

        // Add the row indices to the indices that will be modified
        k_flatten_nblist<<<ceil_divide(NR, tpb), tpb, 0, stream>>>(NR, N_, d_row_idxs.data, d_sphere_idxs_outer.data);
        gpuErrchk(cudaPeekAtLastError());

        // Partition all the indices that make up the the inner and outer sphere, reuse the d_row_Idxs
        cub::DevicePartition::If(
            d_temp_storage_buffer.data,
            temp_storage_bytes,
            d_sphere_idxs_outer.data,
            d_row_idxs.data,
            num_selected_buffer.data,
            N_,
            select_op,
            stream);
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaMemcpyAsync(
            p_num_selected, num_selected_buffer.data, 1 * sizeof(*p_num_selected), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));

        // Copy value from partition into potential idxs
        set_nonbonded_potential_idxs(nonbonded_potential, p_num_selected[0], d_row_idxs.data, stream);

        for (int j = 0; j < local_steps; j++) {
            this->_step(bps_, lambda_schedule[global_steps + j], nullptr, d_sphere_idxs_inner.data, stream);
        }
        // Set back to the full system, for when we go back to global or end the loop
        set_nonbonded_potential_idxs(nonbonded_potential, N_, d_identity_idxs.data, stream);
        if (i % store_x_interval == 0) {
            gpuErrchk(cudaMemcpyAsync(
                d_box_buffer.data + ((i / store_x_interval) - 1) * 3 * 3,
                d_box_t_,
                3 * 3 * sizeof(*d_box_buffer.data),
                cudaMemcpyDeviceToDevice,
                stream));
            gpuErrchk(cudaMemcpy(
                &h_x_buffer[0] + ((i / store_x_interval) - 1) * N_ * 3,
                d_x_t_,
                N_ * 3 * sizeof(*d_x_t_),
                cudaMemcpyDeviceToHost));
        }
    }

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaStreamDestroy(stream));

    std::vector<double> h_box_buffer(box_buffer_size);
    d_box_buffer.copy_to(&h_box_buffer[0]);
    return std::array<std::vector<double>, 2>({h_x_buffer, h_box_buffer});
}

std::array<std::vector<double>, 3>
Context::multiple_steps(const std::vector<double> &lambda_schedule, int store_du_dl_interval, int store_x_interval) {
    unsigned long long *d_du_dl_buffer = nullptr;
    double *d_box_buffer = nullptr;
    // try catch block is to deal with leaks in d_du_dl_buffer
    if (store_du_dl_interval <= 0) {
        throw std::runtime_error("store_du_dl_interval <= 0");
    }
    if (store_x_interval <= 0) {
        throw std::runtime_error("store_x_interval <= 0");
    }
    if (lambda_schedule.size() % store_x_interval != 0) {
        std::cout << "warning:: length of lambda_schedule modulo store_x_interval does not equal zero" << std::endl;
    }

    if (lambda_schedule.size() % store_du_dl_interval != 0) {
        std::cout << "warning:: length of lambda_schedule modulo store_du_dl_interval does not equal zero" << std::endl;
    }

    int du_dl_buffer_size = lambda_schedule.size() / store_du_dl_interval;
    int x_buffer_size = lambda_schedule.size() / store_x_interval;
    int box_buffer_size = x_buffer_size * 3 * 3;

    std::vector<double> h_x_buffer(x_buffer_size * N_ * 3);
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    try {
        gpuErrchk(cudaMalloc(&d_box_buffer, box_buffer_size * sizeof(*d_box_buffer)));
        // indicator so we can set it to a default arg.
        gpuErrchk(cudaMalloc(&d_du_dl_buffer, du_dl_buffer_size * sizeof(*d_du_dl_buffer)));
        gpuErrchk(cudaMemset(d_du_dl_buffer, 0, du_dl_buffer_size * sizeof(*d_du_dl_buffer)));

        for (int i = 1; i <= lambda_schedule.size(); i++) {
            // decide if we need to store the du_dl for this step
            unsigned long long *du_dl_ptr = nullptr;
            if (i % store_du_dl_interval == 0) {
                // pemdas but just to make it clear we're doing pointer arithmetic
                du_dl_ptr = d_du_dl_buffer + ((i / store_du_dl_interval) - 1);
            }

            double lambda = lambda_schedule[i - 1];
            this->_step(bps_, lambda, du_dl_ptr, nullptr, stream);

            if (i % store_x_interval == 0) {
                gpuErrchk(cudaMemcpy(
                    &h_x_buffer[0] + ((i / store_x_interval) - 1) * N_ * 3,
                    d_x_t_,
                    N_ * 3 * sizeof(double),
                    cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(
                    &d_box_buffer[0] + ((i / store_x_interval) - 1) * 3 * 3,
                    d_box_t_,
                    3 * 3 * sizeof(*d_box_buffer),
                    cudaMemcpyDeviceToDevice));
            }
        }

        gpuErrchk(cudaDeviceSynchronize());

        std::vector<unsigned long long> h_du_dl_buffer_ull(du_dl_buffer_size);
        gpuErrchk(cudaMemcpy(
            &h_du_dl_buffer_ull[0],
            d_du_dl_buffer,
            du_dl_buffer_size * sizeof(*d_du_dl_buffer),
            cudaMemcpyDeviceToHost));

        std::vector<double> h_du_dl_buffer_double(du_dl_buffer_size);
        for (int i = 0; i < h_du_dl_buffer_ull.size(); i++) {
            h_du_dl_buffer_double[i] = FIXED_TO_FLOAT<double>(h_du_dl_buffer_ull[i]);
        }
        std::vector<double> h_box_buffer(box_buffer_size);
        gpuErrchk(cudaMemcpy(
            &h_box_buffer[0], d_box_buffer, box_buffer_size * sizeof(*d_box_buffer), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_du_dl_buffer));
        gpuErrchk(cudaFree(d_box_buffer));
        return std::array<std::vector<double>, 3>({h_du_dl_buffer_double, h_x_buffer, h_box_buffer});

    } catch (...) {
        gpuErrchk(cudaFree(d_du_dl_buffer));
        gpuErrchk(cudaFree(d_box_buffer));
        throw;
    }
}

std::array<std::vector<double>, 3> Context::multiple_steps_U(
    const double lambda, // which lambda window we run the integrator over
    const int n_steps,
    const std::vector<double> &lambda_windows, // which lambda windows we wish to evaluate U at
    int store_u_interval,
    int store_x_interval) {

    unsigned long long *d_u_traj = nullptr;
    double *d_box_traj = nullptr;

    if (store_u_interval <= 0) {
        throw std::runtime_error("store_u_interval <= 0");
    }

    if (store_x_interval <= 0) {
        throw std::runtime_error("store_x_interval <= 0");
    }

    if (n_steps % store_x_interval != 0) {
        std::cout << "warning:: n_steps modulo store_x_interval does not equal zero" << std::endl;
    }

    if (n_steps % store_u_interval != 0) {
        std::cout << "warning:: n_steps modulo store_u_interval does not equal zero" << std::endl;
    }

    int n_windows = lambda_windows.size();
    int u_traj_size = (n_steps / store_u_interval) * n_windows;
    int x_traj_size = n_steps / store_x_interval;
    int box_traj_size = x_traj_size * 3 * 3;

    std::vector<double> h_x_traj(x_traj_size * N_ * 3);

    // try catch block is to deal with leaks in d_u_buffer
    try {
        gpuErrchk(cudaMalloc(&d_box_traj, box_traj_size * sizeof(*d_box_traj)));
        gpuErrchk(cudaMalloc(&d_u_traj, u_traj_size * sizeof(*d_u_traj)));
        gpuErrchk(cudaMemset(d_u_traj, 0, u_traj_size * sizeof(*d_u_traj)));

        cudaStream_t stream = static_cast<cudaStream_t>(0);
        for (int step = 1; step <= n_steps; step++) {

            this->_step(bps_, lambda, nullptr, nullptr, stream);

            if (step % store_x_interval == 0) {
                gpuErrchk(cudaMemcpy(
                    &h_x_traj[0] + ((step / store_x_interval) - 1) * N_ * 3,
                    d_x_t_,
                    N_ * 3 * sizeof(double),
                    cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(
                    &d_box_traj[0] + ((step / store_x_interval) - 1) * 3 * 3,
                    d_box_t_,
                    3 * 3 * sizeof(*d_box_traj),
                    cudaMemcpyDeviceToDevice));
            }

            // we need to compute aggregate energies
            if (step % store_u_interval == 0) {
                unsigned long long *u_ptr = d_u_traj + ((step / store_u_interval) - 1) * n_windows;
                for (int w = 0; w < n_windows; w++) {
                    // reset buffers on each pass.
                    gpuErrchk(cudaMemsetAsync(d_u_buffer_, 0, N_ * sizeof(*d_u_buffer_), stream));
                    for (int i = 0; i < bps_.size(); i++) {
                        bps_[i]->execute_device(
                            N_, d_x_t_, d_box_t_, lambda_windows[w], nullptr, nullptr, nullptr, d_u_buffer_, stream);
                    }
                    cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_u_buffer_, u_ptr + w, N_, stream);
                    gpuErrchk(cudaPeekAtLastError());
                }
            }
        }

        gpuErrchk(cudaDeviceSynchronize());

        std::vector<unsigned long long> h_u_traj_ull(u_traj_size);
        gpuErrchk(cudaMemcpy(&h_u_traj_ull[0], d_u_traj, u_traj_size * sizeof(*d_u_traj), cudaMemcpyDeviceToHost));

        std::vector<double> h_u_traj_double(u_traj_size);
        for (int i = 0; i < h_u_traj_ull.size(); i++) {
            h_u_traj_double[i] = FIXED_TO_FLOAT<double>(h_u_traj_ull[i]);
        }
        std::vector<double> h_box_traj(box_traj_size);
        gpuErrchk(cudaMemcpy(&h_box_traj[0], d_box_traj, box_traj_size * sizeof(*d_box_traj), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(d_u_traj));
        gpuErrchk(cudaFree(d_box_traj));
        return std::array<std::vector<double>, 3>({h_u_traj_double, h_x_traj, h_box_traj});

    } catch (...) {
        gpuErrchk(cudaFree(d_u_traj));
        gpuErrchk(cudaFree(d_box_traj));
        throw;
    }
}

void Context::step(double lambda) {
    cudaStream_t stream = static_cast<cudaStream_t>(0);
    this->_step(bps_, lambda, nullptr, nullptr, stream);
    gpuErrchk(cudaDeviceSynchronize());
}

void Context::_step(
    const std::vector<BoundPotential *> &bps,
    const double lambda,
    unsigned long long *du_dl_out,
    unsigned int *atom_idxs,
    const cudaStream_t stream) {

    gpuErrchk(cudaMemsetAsync(d_du_dx_t_, 0, N_ * 3 * sizeof(*d_du_dx_t_), stream));

    if (du_dl_out) {
        gpuErrchk(cudaMemsetAsync(d_du_dl_buffer_, 0, N_ * sizeof(*d_du_dl_buffer_), stream));
    }

    for (int i = 0; i < bps_.size(); i++) {
        bps[i]->execute_device(
            N_,
            d_x_t_,
            d_box_t_,
            lambda,
            d_du_dx_t_, // we only need the forces
            nullptr,
            du_dl_out ? d_du_dl_buffer_ : nullptr,
            nullptr,
            stream);
    }

    // compute du_dl
    if (du_dl_out) {
        cub::DeviceReduce::Sum(d_sum_storage_, d_sum_storage_bytes_, d_du_dl_buffer_, du_dl_out, N_, stream);
        gpuErrchk(cudaPeekAtLastError());
    }

    // for(int i=0; i < streams_.size(); i++) {
    // gpuErrchk(cudaStreamSynchronize(streams_[i]));
    // }
    intg_->step_fwd(d_x_t_, d_v_t_, d_du_dx_t_, d_box_t_, atom_idxs, stream);

    if (barostat_) {
        // May modify coords, du_dx and box size
        barostat_->inplace_move(d_x_t_, d_box_t_, lambda, stream);
    }

    step_ += 1;
};

int Context::num_atoms() const { return N_; }

void Context::set_x_t(const double *in_buffer) {
    gpuErrchk(cudaMemcpy(d_x_t_, in_buffer, N_ * 3 * sizeof(*in_buffer), cudaMemcpyHostToDevice));
}

void Context::set_v_t(const double *in_buffer) {
    gpuErrchk(cudaMemcpy(d_v_t_, in_buffer, N_ * 3 * sizeof(*in_buffer), cudaMemcpyHostToDevice));
}

void Context::get_du_dx_t_minus_1(unsigned long long *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_du_dx_t_, N_ * 3 * sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void Context::get_x_t(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_x_t_, N_ * 3 * sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void Context::get_v_t(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_v_t_, N_ * 3 * sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

void Context::get_box(double *out_buffer) const {
    gpuErrchk(cudaMemcpy(out_buffer, d_box_t_, 3 * 3 * sizeof(*out_buffer), cudaMemcpyDeviceToHost));
}

} // namespace timemachine
