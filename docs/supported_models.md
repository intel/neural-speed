## Supported Models
 
Neural Speed supports the following models:
### Text Generation

<table>
<thead>
  <tr>
    <th rowspan="2">Model Name</th>
    <th colspan="4">INT8</th>
    <th colspan="4">INT4</th>
    <th rowspan="2">Transformer Version</th>
    <th rowspan="2">Max tokens length</th>
  </tr>
  <tr>
    <th>RTN</th>
    <th>GPTQ</th>
    <th>AWQ</th>
    <th>AutoRound</th>
    <th>RTN</th>
    <th>GPTQ</th>
    <th>AWQ</th>
    <th>AutoRound</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf" target="_blank" rel="noopener noreferrer">LLaMA2-7B</a>,
    <a href="https://huggingface.co/meta-llama/Llama-2-13b-chat-hf" target="_blank" rel="noopener noreferrer">LLaMA2-13B</a>,
    <a href="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf" target="_blank" rel="noopener noreferrer">LLaMA2-70B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
    <td>4096</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/decapoda-research/llama-7b-hf" target="_blank" rel="noopener noreferrer">LLaMA-7B</a>,
    <a href="https://huggingface.co/decapoda-research/llama-13b-hf" target="_blank" rel="noopener noreferrer">LLaMA-13B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
    <td>2048</td>
  </tr>
    <td><a href="https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf" target="_blank" rel="noopener noreferrer">CodeLlama-7b</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
    <td>16384</td>
  </tr>
  </tr>
    <td><a href="https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0" target="_blank" rel="noopener noreferrer">Solar-10.7B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
    <td>4096</td>
  </tr>
    <tr>
    <td><a href="https://huggingface.co/Intel/neural-chat-7b-v3-1" target="_blank" rel="noopener noreferrer">Neural-Chat-7B-v3-1</a>,
    <a href="https://huggingface.co/Intel/neural-chat-7b-v3-2" target="_blank" rel="noopener noreferrer">Neural-Chat-7B-v3-2</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
    <td>32768</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/mistralai/Mistral-7B-v0.1" target="_blank" rel="noopener noreferrer">Mistral-7B</a>,
     <a href="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1" target="_blank" rel="noopener noreferrer">Mixtral-8x7B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>4.36.0 or newer</td>
    <td>32768</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Qwen/Qwen-7B-Chat" target="_blank" rel="noopener noreferrer">Qwen-7B</a>,
    <a href="https://huggingface.co/Qwen/Qwen-14B-Chat" target="_blank" rel="noopener noreferrer">Qwen-14B</a>,
    <a href="https://huggingface.co/Qwen/Qwen1.5-7B-Chat" target="_blank" rel="noopener noreferrer">Qwen1.5-7B</a>,
    <a href="https://huggingface.co/Qwen/Qwen1.5-0.5B" target="_blank" rel="noopener noreferrer">Qwen1.5-0.5B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
    <td>8192 / 32768</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/EleutherAI/gpt-j-6b" target="_blank" rel="noopener noreferrer">GPT-J-6B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
    <td>2048</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/EleutherAI/gpt-neox-20b" target="_blank" rel="noopener noreferrer">GPT-NeoX-20B</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
    <td>2048</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/databricks/dolly-v2-3b" target="_blank" rel="noopener noreferrer">Dolly-v2-3B</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>4.28.1 or newer</td>
    <td>2048</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/mosaicml/mpt-7b" target="_blank" rel="noopener noreferrer">MPT-7B</a>,
    <a href="https://huggingface.co/mosaicml/mpt-30b" target="_blank" rel="noopener noreferrer">MPT-30B</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
    <td>2048</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/tiiuae/falcon-7b" target="_blank" rel="noopener noreferrer">Falcon-7B</a>,
    <a href="https://huggingface.co/tiiuae/falcon-40b" target="_blank" rel="noopener noreferrer">Falcon-40B</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
    <td>2048</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bigscience/bloomz-7b1" target="_blank" rel="noopener noreferrer">BLOOM-7B</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
    <td>2048</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/facebook/opt-125m" target="_blank" rel="noopener noreferrer">OPT-125m</a>,
    <a href="https://huggingface.co/facebook/opt-1.3b" target="_blank" rel="noopener noreferrer">OPT-1.3B</a>,
    <a href="https://huggingface.co/facebook/opt-13b" target="_blank" rel="noopener noreferrer">OPT-13B</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
    <td>2048</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/THUDM/chatglm-6b" target="_blank" rel="noopener noreferrer">ChatGLM-6B</a>,
    <a href="https://huggingface.co/THUDM/chatglm2-6b" target="_blank" rel="noopener noreferrer">ChatGLM2-6B</a>,
    <a href="https://huggingface.co/THUDM/chatglm3-6b" target="_blank" rel="noopener noreferrer">ChatGLM3-6B</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>4.33.1</td>
    <td>2048 / 32768</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/baichuan-inc/Baichuan-13B-Chat" target="_blank" rel="noopener noreferrer">Baichuan-13B-Chat</a>,
    <a href="https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat" target="_blank" rel="noopener noreferrer">Baichuan2-13B-Chat</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>4.33.1</td>
    <td>4096</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/microsoft/phi-2" target="_blank" rel="noopener noreferrer">phi-2</a>,
    <a href="https://huggingface.co/microsoft/phi-1_5" target="_blank" rel="noopener noreferrer">phi-1_5</a>
    <a href="https://huggingface.co/microsoft/phi-1" target="_blank" rel="noopener noreferrer">phi-1</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
    <td>2048</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/stabilityai/stablelm-3b-4e1t" target="_blank" rel="noopener noreferrer">StableLM-3B</a>,
    <a href="https://huggingface.co/stabilityai/stablelm-2-1_6b" target="_blank" rel="noopener noreferrer">StableLM2-1_6B</a>
    <a href="https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b" target="_blank" rel="noopener noreferrer">StableLM2-Zephyr-1_6B</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
    <td>2048</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/google/gemma-2b-it" target="_blank" rel="noopener noreferrer">gemma-2b-it </a>,
    <a href="https://huggingface.co/google/gemma-7b" target="_blank" rel="noopener noreferrer">gemma-7b</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
    <td>8192</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/openai/whisper-tiny" target="_blank" rel="noopener noreferrer">Whisper-tiny</a>,
    <a href="https://huggingface.co/openai/whisper-base" target="_blank" rel="noopener noreferrer">Whisper-base</a>
    <a href="https://huggingface.co/openai/whisper-small" target="_blank" rel="noopener noreferrer">Whisper-small</a>
    <a href="https://huggingface.co/openai/whisper-medium" target="_blank" rel="noopener noreferrer">Whisper-medium</a>
    <a href="https://huggingface.co/openai/whisper-large" target="_blank" rel="noopener noreferrer">Whisper-large</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
    <td>448</td>
  </tr>
</tbody>
</table>

### Code Generation

<table>
<thead>
  <tr>
    <th rowspan="2">Model Name</th>
    <th colspan="4">INT8</th>
    <th colspan="4">INT4</th>
    <th rowspan="2">Transformer Version</th>
  </tr>
  <tr>
    <th>RTN</th>
    <th>GPTQ</th>
    <th>AWQ</th>
    <th>AutoRound</th>
    <th>RTN</th>
    <th>GPTQ</th>
    <th>AWQ</th>
    <th>AutoRound</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://huggingface.co/codellama/CodeLlama-7b-hf" target="_blank" rel="noopener noreferrer">Code-LLaMA-7B</a>,
    <a href="https://huggingface.co/codellama/CodeLlama-13b-hf" target="_blank" rel="noopener noreferrer">Code-LLaMA-13B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
  </tr>
    <tr>
    <td><a href="https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B" target="_blank" rel="noopener noreferrer">Magicoder-6.7B</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>Latest</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bigcode/starcoderbase-1b" target="_blank" rel="noopener noreferrer">StarCoder-1B</a>,
    <a href="https://huggingface.co/bigcode/starcoderbase-3b" target="_blank" rel="noopener noreferrer">StarCoder-3B</a>,
    <a href="https://huggingface.co/bigcode/starcoder" target="_blank" rel="noopener noreferrer">StarCoder-15.5B</a></td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>✅</td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>Latest</td>
  </tr>
</tbody>
</table>

### Validated GGUF Models

<table>
<thead>
  <tr>
    <th rowspan="2">Model Name</th>
    <!-- <th colspan="2">HF</th>
    <th colspan="2">Llama.cpp</th> -->

  </tr>
  <tr>
    <th>F32</th>
    <th>F16</th>
    <th>Q4_0</th>
    <th>Q8_0</th>
    <th>BTLA</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF" target="_blank" rel="noopener noreferrer">TheBloke/Llama-2-7B-Chat-GGUF</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF" target="_blank" rel="noopener noreferrer">TheBloke/Mistral-7B-v0.1-GGUF</a>,
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF" target="_blank" rel="noopener noreferrer">TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUFF</a>,
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF" target="_blank" rel="noopener noreferrer">TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td></td>
  </tr>
    </tr>
    <tr>
    <td><a href="https://huggingface.co/codellama/CodeLlama-7b-hf" target="_blank" rel="noopener noreferrer">TheBloke/CodeLlama-7B-GGUF</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td></td>
  </tr>
    </tr>
    <tr>
    <td><a href="https://huggingface.co/codellama/CodeLlama-13b-hf" target="_blank" rel="noopener noreferrer">TheBloke/CodeLlama-13B-GGUF</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/codellama/CodeLlama-7b-hf" target="_blank" rel="noopener noreferrer">Code-LLaMA-7B</a>,
    <a href="https://huggingface.co/codellama/CodeLlama-13b-hf" target="_blank" rel="noopener noreferrer">Code-LLaMA-13B</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    <tr>
    <td><a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf" target="_blank" rel="noopener noreferrer">meta-llama/Llama-2-7b-chat-hf</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
   <tr>
    <td><a href="https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0" target="_blank" rel="noopener noreferrer">upstage/SOLAR-10.7B-Instruct-v1.0</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Qwen/Qwen-7B-Chat" target="_blank" rel="noopener noreferrer">Qwen-7B-Chat</a>,
    <a href="https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF" target="_blank" rel="noopener noreferrer">Qwen1.5-7B-Chat-GGUF</a></td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    <tr>
    <td><a href="https://huggingface.co/tiiuae/falcon-7b/tree/main" target="_blank" rel="noopener noreferrer">tiiuae/falcon-7</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    </tr>
    <tr>
    <td><a href="https://huggingface.co/tiiuae/falcon-40b" target="_blank" rel="noopener noreferrer">tiiuae/falcon-40b</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    </tr>
    <tr>
    <td><a href="https://huggingface.co/mosaicml/mpt-7b" target="_blank" rel="noopener noreferrer">mpt-7b</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    </tr>
    <tr>
    <td><a href="https://huggingface.co/mosaicml/mpt-30b" target="_blank" rel="noopener noreferrer">mpt-30b</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
    </tr>
    </tr>
    <tr>
    <td><a href="https://huggingface.co/bigscience/bloomz-7b1" target="_blank" rel="noopener noreferrer">bloomz-7b1</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
</tbody>
</table>
