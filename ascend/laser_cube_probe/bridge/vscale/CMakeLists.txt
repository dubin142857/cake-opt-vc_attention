cmake_minimum_required(VERSION 3.18)
project(vc_laser_cube_vscale_probe_bridge LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

foreach(required CANN_ROOT TORCH_ROOT TORCH_NPU_ROOT TORCH_ABI VC_LASER_ADAPTER_DIR)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing required CMake definition ${required}")
  endif()
endforeach()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Separate host source identity keeps legacy int8/fp16 provider caches stable.
set(target vc_laser_cube_vscale_probe_bridge)
add_library(${target} SHARED register.cpp "${VC_LASER_ADAPTER_DIR}/find_op_path.cpp")
target_include_directories(${target} PRIVATE
  "${VC_LASER_ADAPTER_DIR}" "${TORCH_ROOT}/include"
  "${TORCH_ROOT}/include/torch/csrc/api/include" "${TORCH_NPU_ROOT}/include"
  "${TORCH_NPU_ROOT}/include/third_party/acl/inc"
  "${TORCH_NPU_ROOT}/include/third_party/hccl/inc"
  "${CANN_ROOT}/include" "${CANN_ROOT}/include/aclnn" ${Python3_INCLUDE_DIRS})
target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI})
target_compile_options(${target} PRIVATE -O3)
target_link_directories(${target} PRIVATE "${TORCH_ROOT}/lib" "${TORCH_NPU_ROOT}/lib" "${CANN_ROOT}/lib64")
target_link_libraries(${target} PRIVATE torch torch_cpu c10 torch_npu ascendcl nnopbase opapi Python3::Python dl)
target_link_options(${target} PRIVATE -Wl,-Bsymbolic -Wl,--no-undefined -Wl,-z,relro,-z,now)
set_target_properties(${target} PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${TORCH_NPU_ROOT}/lib;${CANN_ROOT}/lib64")
