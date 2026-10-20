cmake_minimum_required(VERSION 3.18)
project(vc_laser_quant_vscale_bulk_bridge LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

foreach(required CANN_ROOT TORCH_ROOT TORCH_NPU_ROOT TORCH_ABI VC_LASER_ADAPTER_DIR)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing required CMake definition ${required}")
  endif()
endforeach()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_library(vc_laser_quant_vscale_bulk_bridge SHARED
  register.cpp
  "${VC_LASER_ADAPTER_DIR}/find_op_path.cpp")
target_include_directories(vc_laser_quant_vscale_bulk_bridge PRIVATE
  "${VC_LASER_ADAPTER_DIR}"
  "${TORCH_ROOT}/include"
  "${TORCH_ROOT}/include/torch/csrc/api/include"
  "${TORCH_NPU_ROOT}/include"
  "${TORCH_NPU_ROOT}/include/third_party/acl/inc"
  "${TORCH_NPU_ROOT}/include/third_party/hccl/inc"
  "${CANN_ROOT}/include"
  "${CANN_ROOT}/include/aclnn"
  ${Python3_INCLUDE_DIRS})
target_compile_definitions(vc_laser_quant_vscale_bulk_bridge PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI})
target_compile_options(vc_laser_quant_vscale_bulk_bridge PRIVATE -O3)
target_link_directories(vc_laser_quant_vscale_bulk_bridge PRIVATE
  "${TORCH_ROOT}/lib" "${TORCH_NPU_ROOT}/lib" "${CANN_ROOT}/lib64")
target_link_libraries(vc_laser_quant_vscale_bulk_bridge PRIVATE
  torch torch_cpu c10 torch_npu ascendcl nnopbase opapi Python3::Python dl)
target_link_options(vc_laser_quant_vscale_bulk_bridge PRIVATE
  -Wl,-Bsymbolic -Wl,--no-undefined -Wl,-z,relro,-z,now)
set_target_properties(vc_laser_quant_vscale_bulk_bridge PROPERTIES
  BUILD_RPATH "${TORCH_ROOT}/lib;${TORCH_NPU_ROOT}/lib;${CANN_ROOT}/lib64")
