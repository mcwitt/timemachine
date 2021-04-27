jit_program

#include "timemachine/cpp/src/kernels/surreal.cuh"

#define PI 3.141592653589793115997963468544185161

template <typename NumericType>
NumericType __device__ __forceinline__ transform_lambda_charge(NumericType lambda) {
    return CUSTOM_EXPRESSION_CHARGE;
}

template <typename NumericType>
NumericType __device__ __forceinline__ transform_lambda_sigma(NumericType lambda) {
    return CUSTOM_EXPRESSION_SIGMA;
}

template <typename NumericType>
NumericType __device__ __forceinline__ transform_lambda_epsilon(NumericType lambda) {
    return CUSTOM_EXPRESSION_EPSILON;
}

void __global__ k_permute_interpolated(
    const double lambda,
    const int N,
    const unsigned int * __restrict__ perm,
    const double * __restrict__ d_p,
    double * __restrict__ d_sorted_p,
    double * __restrict__ d_sorted_dp_dl) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.y;
    int stride_idx = blockIdx.y;

    if(idx >= N) {
        return;
    }

    int size = N*stride;

    int source_idx = idx*stride+stride_idx;
    int target_idx = perm[idx]*stride+stride_idx;

    double eps = 1e-7;
    Surreal<double> lambda_surreal(lambda, eps);

    double f_lambda;
    double f_lambda_grad;

    if(stride_idx == 0) {
        f_lambda = transform_lambda_charge(lambda);
        f_lambda_grad = (transform_lambda_charge(lambda_surreal).imag)/eps;
    }
    if(stride_idx == 1) {
        f_lambda = transform_lambda_sigma(lambda);
        f_lambda_grad = (transform_lambda_sigma(lambda_surreal).imag)/eps;
    }
    if(stride_idx == 2) {
        f_lambda = transform_lambda_epsilon(lambda);
        f_lambda_grad = (transform_lambda_epsilon(lambda_surreal).imag)/eps;
    }

    d_sorted_p[source_idx] = (1-f_lambda)*d_p[target_idx] + f_lambda*d_p[size+target_idx];
    d_sorted_dp_dl[source_idx] = f_lambda_grad*(d_p[size+target_idx] - d_p[target_idx]);

}
