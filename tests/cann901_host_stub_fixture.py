# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# See ../ascend/laser_quant_preprocess/LICENSE-CANN.txt for the full license.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
"""Small, offline regression fixture from the *actual* CANN 9.0.1 parser.

Upstream: https://gitcode.com/cann/asc-devkit/tree/v9.0.1
Commit: c3e271c396ca036aa76be5745b4b4c242eeb06a2
cmake/asc/legacy_modules/util/constants.py: FUNC_SIGNATURE, FUNC_PARAMS
cmake/asc/legacy_modules/util/extract_host_stub.py: parse_func_signature_parts
cmake/asc/legacy_modules/util/host_stub_util.py: exception names (bases simplified)

The two expressions and parsing function below are unchanged. This exercises
signature parsing only, not preprocessing, CCE compilation or device execution.
"""
import re
from typing import Tuple

FUNC_SIGNATURE = re.compile(
    r'(?:template\s*<\s*([^>]+?)\s*>\s*\n?\s*)?__attribute__\(\(cce_kernel\)\)\s*\[aicore\]\s*(.+?)\s*\{', re.DOTALL
)
FUNC_PARAMS = re.compile(r'\((.+)\)', re.DOTALL)


class FuncNameNotFound(Exception):
    """Function name not found."""


class MultiFuncNameFound(Exception):
    """Multiple function name not found."""


def parse_func_signature_parts(func_sign: str) -> Tuple[str, str, str]:
    """Parse function signature.
    Return a tuple of return type, function name and function parameters.
    """
    params_obj = FUNC_PARAMS.search(func_sign)
    if params_obj is None:
        # in no param case, to convert "void xxx()" into "void xxx", use [:-2]
        func_sign_without_params = func_sign[:-2]
        func_params = ""
    else:
        func_sign_without_params = func_sign[:params_obj.start(0)]
        func_params = params_obj.group(1).strip()

    # convert params like "xxx(void)", "xxx(  void  )" into "xxx()"
    if func_params == "void":
        func_params = ""
    parts = re.split(r'\s+', func_sign_without_params.strip())
    if len(parts) <= 1:
        raise FuncNameNotFound()
    if len(parts) > 2:
        raise MultiFuncNameFound()
    return_type, func_name = parts

    return return_type, func_name, func_params
