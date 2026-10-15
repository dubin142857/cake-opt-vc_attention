# CANN ops-transformer v9.0.1 custom FlashAttentionScore package only.
# The source preparation step verifies the upstream commit before configuration.
# Inject with -DCMAKE_PROJECT_INCLUDE=/absolute/path/to/this/file.  Defer until
# all upstream targets have been declared, including when root CMakeLists uses
# return().  Nothing is removed from the source tree or dependency graph.
if(CMAKE_VERSION VERSION_LESS "3.19")
    message(FATAL_ERROR "The CANN FAS incremental build requires CMake >= 3.19")
endif()

# CMAKE_PROJECT_INCLUDE may also run for dependency project() calls.
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    return()
endif()
include_guard(GLOBAL)

function(vc_cann_fas_acl_only)
    if(NOT PROJECT_NAME STREQUAL "cann_ops-transformer"
       OR NOT BUILD_OPEN_PROJECT
       OR ENABLE_BUILT_IN
       OR ENABLE_TEST
       OR ENABLE_STATIC
       OR BUILD_OPS_RTY_KERNEL
       OR NOT ASCEND_OP_NAME STREQUAL "flash_attention_score")
        message(FATAL_ERROR
            "CANN FAS ACL-only build is restricted to the pinned custom "
            "flash_attention_score package (no built-in/test/static/RTY mode)")
    endif()
    foreach(required_target IN ITEMS cust_opapi cust_proto cust_opmaster
                                     op_transformer_onnx_plugin_obj)
        if(NOT TARGET ${required_target})
            message(FATAL_ERROR
                "CANN FAS upstream target layout changed: missing ${required_target}")
        endif()
    endforeach()
    if(TARGET op_transformer_onnx_plugin)
        message(FATAL_ERROR
            "CANN FAS refuses to skip ONNX when an ONNX shared library is present")
    endif()

    # In this upstream custom package the ONNX OBJECT library is not installed
    # or linked into the ACLNN, op-proto, or tiling libraries. Its default ALL
    # membership nevertheless pulls protoc/Protobuf/Abseil into the build.
    # Those ExternalProjects already use EXCLUDE_FROM_ALL upstream. Excluding
    # just the orphan ONNX object target retains every explicit dependency:
    # if a required core target needs any of these targets, CMake still builds it.
    set_property(TARGET op_transformer_onnx_plugin_obj PROPERTY EXCLUDE_FROM_ALL TRUE)
    message(STATUS
        "CANN FAS ACL-only: skip standalone ONNX objects in ALL; "
        "retain ACLNN/proto/tiling/device targets and explicit dependencies")
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL vc_cann_fas_acl_only)
