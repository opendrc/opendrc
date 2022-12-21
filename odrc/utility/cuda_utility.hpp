#pragma once

namespace odrc::util {

template <typename check_result, typename index_type>
__device__ void write_to_global_memory(check_result* global_address,
                                       index_type*   global_index,
                                       check_result* result,
                                       index_type    size) {
  // this is a temporary solution, we may change it to tune for the realistic
  // workload
  index_type index = atomicAdd(global_index, size);
  memcpy(global_address + index, result, size * sizeof(check_result));
}

}  // namespace odrc::util
