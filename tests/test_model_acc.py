#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import shutil
import os
import unittest

class TestLLMRUNTIME(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("./runtime_outs", ignore_errors=True)

    def test_ns_acc(self):
        os.system("numactl -l -C 0-55 python ../scripts/cal_acc.py --model_name /tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf --weight_dtype int4 --group_size 32 --compute_dtype int8 --tasks piqa")
        


if __name__ == "__main__":
    unittest.main()
