
#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <thrust/scan.h>
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace xstd {
  namespace alpaka {
    namespace internal {

      template <typename InputIterator, typename OutputIterator>
      ALPAKA_FN_HOST inline constexpr void inclusive_scan(InputIterator first,
                                                          InputIterator last,
                                                          OutputIterator result) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        thrust::inclusive_scan(thrust::device, first, last, result);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        thrust::inclusive_scan(thrust::hip::par, first, last, result);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        oneapi::dpl::inclusive_scan(oneapi::dpl::execution::dpcpp_default, first, last, result);
#else
        std::inclusive_scan(first, last, result);
#endif
      }

      template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
      ALPAKA_FN_HOST inline constexpr void inclusive_scan(ExecutionPolicy&& policy,
                                                          InputIterator first,
                                                          InputIterator last,
                                                          OutputIterator result) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        oneapi::dpl::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result);
#else
        std::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result);
#endif
      }

      template <typename InputIterator, typename OutputIterator, typename AssociativeOperator>
      ALPAKA_FN_HOST inline constexpr void inclusive_scan(InputIterator first,
                                                          InputIterator last,
                                                          OutputIterator result,
                                                          AssociativeOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        thrust::inclusive_scan(thrust::device, first, last, result, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        thrust::inclusive_scan(thrust::hip::par, first, last, result, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        oneapi::dpl::inclusive_scan(oneapi::dpl::execution::dpcpp_default, first, last, result, op);
#else
        std::inclusive_scan(first, last, result, op);
#endif
      }

      template <typename ExecutionPolicy,
                typename InputIterator,
                typename OutputIterator,
                typename AssociativeOperator>
      ALPAKA_FN_HOST inline constexpr void inclusive_scan(ExecutionPolicy&& policy,
                                                          InputIterator first,
                                                          InputIterator last,
                                                          OutputIterator result,
                                                          AssociativeOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        thrust::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        oneapi::dpl::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result, op);
#else
        std::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result, op);
#endif
      }

      template <typename InputIterator,
                typename OutputIterator,
                typename T,
                typename AssociativeOperator>
      ALPAKA_FN_HOST inline constexpr void inclusive_scan(InputIterator first,
                                                          InputIterator last,
                                                          OutputIterator result,
                                                          T init,
                                                          AssociativeOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        thrust::inclusive_scan(thrust::device, first, last, result, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        thrust::inclusive_scan(thrust::hip::par, first, last, result, init, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        oneapi::dpl::inclusive_scan(
            oneapi::dpl::execution::dpcpp_default, first, last, result, init, op);
#else
        std::inclusive_scan(first, last, result, init, op);
#endif
      }

      template <typename ExecutionPolicy,
                typename InputIterator,
                typename OutputIterator,
                typename T,
                typename AssociativeOperator>
      ALPAKA_FN_HOST inline constexpr void inclusive_scan(ExecutionPolicy&& policy,
                                                          InputIterator first,
                                                          InputIterator last,
                                                          OutputIterator result,
                                                          T init,
                                                          AssociativeOperator op) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        thrust::inclusive_scan(
            std::forward<ExecutionPolicy>(policy), first, last, result, init, op);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        thrust::inclusive_scan(
            std::forward<ExecutionPolicy>(policy), first, last, result, init, op);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
        oneapi::dpl::inclusive_scan(
            std::forward<ExecutionPolicy>(policy), first, last, result, init, op);
#else
        std::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result, init, op);
#endif
      }

    }  // namespace internal
  }  // namespace alpaka
}  // namespace xstd
