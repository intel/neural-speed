# Docker
Follow these instructions to set up and run our provided Docker image.

## Set Up Docker Image
Build or Pull the provided docker images.

### Build Docker Image
```bash
git clone https://github.com/intel/neural-speed.git neuralspeed
cd neuralspeed
docker build -f docker/DockerFile neuralspeed:latest .
```
If you need to use proxy, please use the following command
```bash
docker build --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f docker/DockerFile -t neuralspeed:latest .
```

### Pull From Docker Hub


## Use Docker Image
Utilize the docker container based on docker image.
```bash
docker run -itd --name="neural-speed-docker" neuralspeed:latest /bin/bash
docker exec -it neural-speed-docker /bin/bash
```

## Run Simple Test
```bash
docker exec -it <container_name> /bin/bash
cd /neural_speed/neural_speed
## convert to model.bin
python scripts/convert.py --outtype f32 --outfile llama-fp32.bin ${input_model_path}
## quantize to Q4 with groupsize=128
./build/bin/quant_llama --model_file llama-fp32.bin --out_file llama-q4_j_i8_g128.bin --weight_dtype int4 --group_size 128 --compute_dtype int8 --scale_dtype fp32 --alg sym
## inference
./build/bin/run_llama --seed 1234 -b 2047 -c 64 -n 32 -m llama-q4_j_i8_g128.bin -p "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
```
