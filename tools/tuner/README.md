XeTLA Tuner
==================================

# Introduction
XeTLA Tuner is a tool used to search the optimal combination of tuning parameters supported by a kernel written in XeTLA.<br>
The tuner will automatically complete the following work:<br>
* Iterate combinations of tuning parameters, such as wg, sg, shared memory size and so on<br>
* Run kernel with different combinations of tuning parameters in real HW<br>
* Evaluate performance and choose the best tuning parameters combination<br>


# Build
        
## Fetch Source Code

        $bash
        
        $mkdir <yourworkdir>
        
        $cd <yourworkdir>
   
        $git clone https://github.com/intel-innersource/libraries.gpu.xetla
   
        $cd <libraries.gpu.xetla>
   

  Currently the tuner tool is not open source and is placed on XeTLA main branch. As we know, XeTLA is open source, and we may use different versions of the open source XeTLA to write our kernels, so we may also need to clone the XeTLA used by our kernel and tell Tuner tool which version of XeTLA to use to compile our kernel. <br>
## Configuration
If we want to tune a kernel, we need to provide the micro-kernel template that conforms to a specific format, and we also need to provide a yaml file to define the tuning parameters. We need to make a new directory under the corresponding directory, such as ./tools/tuner/resource_file/micro_kernel_repo/gemm/gptj_gemm, and gptj_gemm is the new directory used to place kernel files. In the new directory, we need to provide at least the following files:<br>
    xxxx.yaml, CMakeLists.txt, test.hpp and main.cpp<br>
Below we will introduce the purpose of each file in detail:<br>
* xxxx.yaml<br>
   This file is used to define the tuning parameters. Tuner will load this file and get the value set of each tuning parameter. The yaml configuration file looks like below:
```
file_type: micro_kernel_attri_cfg 
micro_kernel_type: GEMM
micro_kernel_name: kernel_func.hpp
tune_parameter: 
    wg_m: 
        start_value: 8
        end_value: 256
        stride: 8
```
"file_type" is used to describe the purpose of this file, and we can describe it according to our need. "micro_kernel_type" is used to define the type of the kernel, such as GEMM, current the type can be GEMM. "micro_kernel_name" is used to describe the kernel name, we recommend using a unique name. "tune_parameter" is used to define the tuning parameters, one tuning parameter has a name and a tuple. We can define the range of one tuning parameter with a tuple (start_value, end_value, stride), in which start_value is the minimum value of the parameter, end_value is the maximum value of the parameter, and stride is the step length, e.g. [start, start+stride, start+stride*2..., end].
* CMakeLists.txt<br>
   This file is the cmake file of the kernel, which is used to build the kernel.
* test.hpp<br>
   This file is used to define the test case of the kernel.<br>
   We need to provide "class TestBase" which is the base class of each test case, e.g:<br>
    ```
    class TestBase {
    public:
        static constexpr size_t batch_size = 1;
    };
    class test_case_4096x4096x4096 : public TestBase {
    public:
        static constexpr size_t mat_m = 4096;
        static constexpr size_t mat_k = 4096;
        static constexpr size_t mat_n = 4096;
        static constexpr size_t wg_m = 16;
        static constexpr size_t wg_n = 64;
        static constexpr size_t sg_m = 16;
        static constexpr size_t sg_n = 16;
        static constexpr size_t sg_k = 16;
        static constexpr uint32_t local_kslicing = 8;
        static constexpr uint32_t global_kslicing = 1;

        static constexpr mem_layout layout_a = mem_layout::row_major;
        static constexpr mem_layout layout_b = mem_layout::row_major;
        using data_type_a = bf16;
        using data_type_b = bf16;
        using data_type_c = bf16;
    };
    ```
    We also need to provide below code which is used to define the test scenario.<br>
    ```
    using tests = ::testing::Types<test_case_4096x4096x4096>;
    ```
* main.cpp<br>
   This file is used to run the kernel and output the preformance data.
   ```
    ……
    template <class Test>
    void gemm_run(int iter){

    }
    ……
    template <typename T>
    class tuner_kernel_gtpj_gemm : public ::testing::Test {};
    TYPED_TEST_SUITE_P(tuner_kernel_gtpj_gemm);
    TYPED_TEST_P(tuner_kernel_gtpj_gemm, esimd) {
        gemm_run<TypeParam>(ITER);
    }

    REGISTER_TYPED_TEST_SUITE_P(tuner_kernel_gtpj_gemm, esimd);
    INSTANTIATE_TYPED_TEST_SUITE_P(
        tuner_kernel_gtpj_gemm_test_suite, tuner_kernel_gtpj_gemm, tests);
    
    int main(int argc, char **argv) {
        if (argc > 1) {
            string arg = argv[1];
            if (arg == "1") { enable_validation = true; }
        }
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
   ```
We can see that the execution of the kernel is based on gtest framework. We need to get a value from the input parameter of the main function. This value(stored in enable_validation) is used to control whether it is necessary to verify the correctness of the kernel's output results. We can also add a function can_implement, which is used to check if current configuration is supported by our kernel, if not, we can skip the execution of the kernel with current configuration, of cource the can_implement is optional. 

## Build tuner
Before building tuner, please correctly set the environment variables used to build XeTLA, we can use different compilers and drivers we need. No matter what compiler and driver we use, we need to ensure that the XeTLA on which the kernel is written can be compiled and run successfully, and the kernel will be tuned can run normally. If you are using XeTLA for the first time, please refer to [README](https://github.com/intel-innersource/libraries.gpu.xetla/blob/main/README.md).<br>
In "<libraries.gpu.xetla>/tools/tuner/**set_tuner_env.sh**", we may need to change two environment variables. If our kernel is not wrote in XeTLA main branch, we need to clone the specified XeTLA branch we used, and then set the environment variable "**xetla_src_dir**" in "./libraries.gpu.xetla/tools/tuner/**set_tuner_env.sh**", the "libraries.gpu.xetla" mentioned here refers to XeTLA main branch. If the execution time of our kernel is larger than the value of "**XETLA_TUNER_CMD_TIME_OUT**" in "**set_tuner_env.sh**", we need to enlarge the "**XETLA_TUNER_CMD_TIME_OUT**".<br>
Please follow the steps below to build tuner:

        $bash
      
        $cd <yourworkdir>

        $source <libraries.gpu.xetla>/tools/scripts/env.sh

        $cd <libraries.gpu.xetla/tools/tuner>

        $source <libraries.gpu.xetla>/tools/tuner/set_tuner_env.sh

        $mkdir build 
   
        $cd build
   
        $cmake .. 
   
        $make  
   
# Run Tuner
## Get tuner help
        $bash
   
        $cd <libraries.gpu.xetla>/tools/tuner/build
        
        $./tuner --help
        ============Tuner supports cmd=========
        ./tuner --operation=GEMM  --m=4096 --n=4096 --k=2048 --A-shape=bf16:row --B-shape=bf16:row --C-shape=f32:row --output=tune_output.csv --verification-enabled=true
        =======================================
        The parameters "m", "n" and "k" are used to describe the matrix shape. In "A-shape=bf16:row", "bf16" is the data type of Matrix A, and "row" is memory layout.
        The parameter "output" is optional, the value of "output" is a file name, which ends with .csv, the file is used to record the performance of every available combination of tunable parameters. If the format of the value of "output" is not correct, the tuner will exit and give error information "the value of output can only be filename.csv" to user. If the user does not specify this option, the tuner will use a default file name to "output" the detailed result of every available tunable combination.
        The parameter "verification-enabled" is optional, the value of "verification-enabled" can be true or false, "true" means that the kernel caller needs to verify the correctness of the kernel running results, "false" means that the kernel caller does not need to verify the correctness of the kernel running results. If the value of "verification-enabled" is not true or false, the tuner will exit and give error information "the value of verification-enabled can only be true or false" to user. If user does not specify this option, the default value of "verification-enabled" is true.         
## Tune GEMM Kernel    
        $bash
   
        $cd <libraries.gpu.xetla>/tools/tuner/build
        
        $./tuner --operation=GEMM  --m=4096 --n=4096 --k=2048 --A-shape=bf16:row --B-shape=bf16:row --C-shape=bf16:row --output=tune_output.csv --verification-enabled=true

        

## Tune result:
        After running tuner successfully, the tuner will output logs like below:
        ======================================================
        -----Top kernel information:-----
        -----kernnel time:0.00191515
        -----kernnel path:./tune_detail_2023_10_22_23_42_42/xetla_tune_case_kernel_func_2023_10_22_23_43_59_2303_3093952/
        -----configuration: wg_m: 16, wg_n: 16, sg_m: 8, sg_n: 16, sg_k: 16, local_kslicing: 1, global_kslicing: 1


        -----kernnel time:0.00199919
        -----kernnel path:./tune_detail_2023_10_22_23_42_42/xetla_tune_case_kernel_func_2023_10_22_23_43_59_6772_3093952/
        -----configuration: wg_m: 8, wg_n: 16, sg_m: 8, sg_n: 16, sg_k: 16, local_kslicing: 1, global_kslicing: 1


        -----kernnel time:0.00207354
        -----kernnel path:./tune_detail_2023_10_22_23_42_42/xetla_tune_case_kernel_func_2023_10_22_23_43_59_43644_3093952/
        -----configuration: wg_m: 16, wg_n: 16, sg_m: 16, sg_n: 16, sg_k: 16, local_kslicing: 1, global_kslicing: 1


        ======================================================



# License
This is an internal tool developed by XeTLA team, do not spread without permission.<br>
Now we use the following third-party components:<br>
* yaml-cpp
* googletest



