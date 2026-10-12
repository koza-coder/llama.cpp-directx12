# Running llama.cpp on any GPU with Direct3D 12

This fork adds a native **Direct3D 12** backend to llama.cpp. It needs no CUDA, no ROCm, no
vendor SDK and no Vulkan driver - only a GPU whose Windows driver supports D3D12 feature level
11_0, which covers nearly every graphics card sold since about 2015, including ones no other
llama.cpp backend supports.

- Backend source: [ggml/src/ggml-d3d12](ggml/src/ggml-d3d12) (host code `ggml-d3d12.cpp`, HLSL compute shaders in `hlsl/`)
- Device name in llama.cpp: `D3D120`, `D3D121`, ...
- Platform: Windows 10/11 x64
- Other versions of this fork: [DirectX 11](https://github.com/koza-coder/llama.cpp-directx11) for older GPUs and
  drivers, [OpenGL](https://github.com/koza-coder/llama.cpp-opengl) - same llama.cpp, only the GPU backend differs

You do not need to build anything. Download the zip, download a model, run one command.
Copy and paste each block below into a Command Prompt or PowerShell window. `curl` and `tar` are
part of Windows 10 and 11, so nothing else has to be installed.

---

## 1. Get the release

Download `llama-d3d12-<version>-win-x64.zip` from the
[latest release](https://github.com/koza-coder/llama.cpp-directx12/releases/latest) and unpack it
anywhere (right click, "Extract All", or `tar -xf <zip file>` in a terminal). The same release has a
smaller `.7z` with the same files, and `SHA256SUMS.txt`. To check the download is intact, compare
this number with the line for your file in `SHA256SUMS.txt`:

```
certutil -hashfile llama-d3d12-<version>-win-x64.zip SHA256
```

Inside you will find:

| file | what it is |
|---|---|
| `chat.bat` | double-click: chat in your browser with the first `.gguf` model in the folder |
| `llama-cli.exe` | chat with a model in the terminal |
| `llama-server.exe` | OpenAI-compatible server with a web chat page (what `chat.bat` starts) |
| `llama-mtmd-cli.exe` | describe a picture or transcribe speech |
| `llama-completion.exe` | plain text completion, no chat template |
| `llama-bench.exe` | measure speed |
| `llama-perplexity.exe` | measure model quality on a text file |
| `test-backend-ops.exe` | check the GPU produces correct results |
| `dxcompiler.dll`, `dxil.dll` | **required** - the backend compiles its shaders with these |
| `d3d12-shader-cache/` | precompiled shaders, so the first start is quick |

Keep all files together. The programs are otherwise self-contained: no runtime to install,
no other DLLs, nothing added to your system.

---

## 2. Check your GPU is found

```
llama-bench.exe --list-devices
```

You should see a line naming your graphics card with `D3D120` in front of it. If you have two
GPUs you will see `D3D120` and `D3D121`.

If nothing is listed, your driver does not expose a D3D12 device. Updating the graphics driver is
the first thing to try.

---

## 3. Get a model

Pick one. The first is small enough to try in a minute; the others are better at actually holding
a conversation. Each is one file, and the `curl` line puts it in the folder you are already in.

**Tiny, 105 MB** - SmolLM2 135M. Fast to download, talks nonsense but proves the whole path works:

```
curl -L -o model.gguf https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf
```

**Small, 429 MB** - Qwen2.5 0.5B. A reasonable first real model:

```
curl -L -o model.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
```

**Mixture of experts, 822 MB** - Granite 3.0 1B A400M. Only 400M of its parameters run per token, so
it is quick, and it exercises a different path in the backend than the two above:

```
curl -L -o model.gguf https://huggingface.co/bartowski/granite-3.0-1b-a400m-instruct-GGUF/resolve/main/granite-3.0-1b-a400m-instruct-Q4_K_M.gguf
```

Check the download is complete (`certutil -hashfile model.gguf SHA256`):

| file | bytes | SHA-256 |
|---|---:|---|
| `SmolLM2-135M-Instruct-Q4_K_M.gguf` | 105454432 | `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d` |
| `qwen2.5-0.5b-instruct-q4_0.gguf` | 428730208 | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` |
| `granite-3.0-1b-a400m-instruct-Q4_K_M.gguf` | 821845024 | `074f09e13484e54e73c93830d34e9fa9917a6319fb8bae762a22594b9b4da0dc` |

Any `.gguf` file from Hugging Face works, not only these three. Larger models need more video
memory; a rough rule is that the file has to fit in your card's VRAM, plus about 1 GB for the
context.

---

## 4. Talk to it

Double-click `chat.bat`: it starts `llama-server.exe` with the model in the folder and opens the chat
page in your browser. Close its window to stop it. Or, in the terminal:

```
llama-cli.exe -m model.gguf -ngl 99
```

`-ngl 99` means "put every layer on the GPU". Without it the model runs on your processor and the
GPU is never used, which is a common reason for a disappointing first run.

Type a message, press Enter. Ctrl+C quits.

For a single answer instead of a chat:

```
llama-completion.exe -m model.gguf -ngl 99 -p "The capital of France is" -n 32
```

---

## 5. Measure it

```
llama-bench.exe -m model.gguf -ngl 99
```

Two numbers come out. `pp512` is prompt processing, how fast it reads your input. `tg128` is text
generation, how fast it writes the answer. Both are tokens per second, higher is better.

To see what difference the GPU makes, run it again with `-ngl 0` and compare.

---

## 6. Check the GPU is correct, not just fast

This runs every operation on your GPU and compares the result against the same operation on your
processor, so it tells you whether this backend is right on hardware nobody has tested it on.

Start small:

```
test-backend-ops.exe test -b D3D120 -o GET_ROWS
```

Then the big one, which takes a few minutes:

```
test-backend-ops.exe test -b D3D120 -o MUL_MAT
```

Every line should end in `OK`. A line ending in `FAIL` is a real result worth reporting - see
section 8.

> **A warning worth reading.** A GPU compute program that misbehaves can freeze your screen for a
> few seconds while Windows resets the display driver, and on some drivers it can take the desktop
> down with it. Save your work before running the tests for the first time. If the screen goes
> black and comes back, that is Windows recovering and nothing is damaged. Run the op groups one at
> a time (`-o <op>`) rather than all at once, and do not run `perf` mode until the normal tests pass.

---

## 7. Pictures and sound

`llama-mtmd-cli.exe` runs multimodal models. These need **two** files: the model itself and a
matching "mmproj" projector that turns a picture or a sound into something the model understands.
Both come from the same Hugging Face repository, and they must be a matching pair.

### Describe a picture

```
curl -L -o vision.gguf https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf
```

```
curl -L -o vision-mmproj.gguf https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
```

```
llama-mtmd-cli.exe -m vision.gguf --mmproj vision-mmproj.gguf --image picture.jpg -ngl 99 -p "Describe this picture."
```

`.jpg` and `.png` both work.

### Transcribe speech

```
curl -L -o asr.gguf https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B-Q8_0.gguf
```

```
curl -L -o asr-mmproj.gguf https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf
```

```
llama-mtmd-cli.exe -m asr.gguf --mmproj asr-mmproj.gguf --audio speech.mp3 -ngl 99 -p "Transcribe this audio."
```

`.mp3`, `.wav` and `.flac` all work. `--image` and `--audio` are in fact the same option, so
either name accepts either kind of file.

| file | bytes | SHA-256 |
|---|---:|---|
| `SmolVLM-500M-Instruct-Q8_0.gguf` | 436806912 | `9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4` |
| `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` | 108783360 | `d1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150` |
| `Qwen3-ASR-0.6B-Q8_0.gguf` | 804749248 | `bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971` |
| `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf` | 214392480 | `41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d` |

---

## 8. If something goes wrong

**"dxcompiler.dll was not found" or no shaders compile**
You moved the exe out of the folder. All files have to stay together.

**No device listed by `--list-devices`**
Update your graphics driver. If it still shows nothing, your GPU or driver does not do D3D12
feature level 11_0 - try the [DirectX 11 version](https://github.com/koza-coder/llama.cpp-directx11).

**It runs but is slower than the processor**
Check you passed `-ngl 99`. Also check `--list-devices` is showing your real GPU and not a
software adapter called "Microsoft Basic Render Driver".

**A test fails, or the program stops with "device removed"**
That is a genuine bug report. Set this first and run it again, so the crash names the shader that
caused it instead of only printing an error code:

```
set GGML_D3D12_DRED=1
```

In PowerShell that is `$env:GGML_D3D12_DRED=1` instead.

Then [open an issue](https://github.com/koza-coder/llama.cpp-directx12/issues) with your GPU model,
your driver version, the zip version, the exact command you ran, and the output.

---

## 9. What has been tested

Qwen2.5-0.5B-Instruct Q4_0, `llama-bench` (tokens per second, mean of 2 runs; pp512 = reading a
512-token prompt, tg64 = writing 64 tokens) and `llama-perplexity` (wikitext-2, 8 chunks, lower is
better). The release is named per row, because not every GPU has been re-measured on the newest zip:

| GPU | Release | Prompt (pp512) | Generation (tg64) | Perplexity |
|---|---|---:|---:|---:|
| AMD Radeon AI PRO R9700 | v0.3.0 | 7280 (+/- 633) | 173.5 (+/- 1.1) | 17.2885 |
| Intel Iris Xe | v0.2.3 | 282 | 18.4 | 17.2876 |
| Moore Threads MTT S80 | v0.2.3 | 823 | 62.0 | 17.2886 |

Short prompts (2 to 15 tokens: chat turns, speculative decoding) got much faster in v0.3.0 (prompt
tokens per second, before -> after, mean of 2 runs): R9700 LFM2.5-1.2B Q8_0 pp4 142.7 -> 288.5 and
pp15 167.1 -> 518.5; R9700 Ministral-3-3B Q6_K pp15 64.7 -> 278.5; Intel Iris Xe qwen2.5-0.5b Q4_0
pp15 50.9 -> 124.8. Prompts of 16 tokens and longer are unchanged.

Moore Threads MTT S80: all 19 op groups passed with v0.2.0; granite-3.0-1b-a400m Q4_K_M ran at
pp512 537 / tg64 36.8 tok/s.

**No NVIDIA GPU has been tested.** Zhaoxin / Glenfly and Lisuan GPUs have not been tested either. If
you run this on one, the result is useful information either way - please report it.

Operations the backend does not support run on the processor (the ggml scheduler does this
automatically; `GGML_D3D12_STATS=1` lists them). The authoritative list of supported ops and types
is `ggml_backend_d3d12_device_supports_op` in `ggml-d3d12.cpp`.

---

## 10. Build it yourself

The tested way is to cross-compile on Linux (Debian/Ubuntu, x86-64 or arm64) with MinGW-w64:

```
sudo apt install cmake python3 gcc-mingw-w64-x86-64-posix g++-mingw-w64-x86-64-posix binutils-mingw-w64-x86-64
```

```
git clone https://github.com/koza-coder/llama.cpp-directx12
cd llama.cpp-directx12
```

```
cmake -B build-win \
    -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-w64-mingw32.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_D3D12=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-static -static-libgcc -static-libstdc++"
```

```
cmake --build build-win -j 4 --target llama-cli llama-server llama-bench test-backend-ops
```

The .exe files are in `build-win/bin/`. Python 3 is needed at build time only: `hlsl/embed_hlsl.py`
embeds the HLSL into the binary, and the shaders are compiled at runtime. Copy `dxcompiler.dll` and
`dxil.dll` from a [DirectXShaderCompiler release](https://github.com/microsoft/DirectXShaderCompiler/releases)
(`bin/x64/`) next to the executables; without `dxil.dll` the shaders are unsigned and most drivers
reject them. A native Windows build (MSYS2 MinGW or MSVC) has not been tested.

---

## 11. Environment variables

Set them with `set NAME=value` (Command Prompt) or `$env:NAME="value"` (PowerShell). All are off by
default.

| Variable | Effect |
|---|---|
| `GGML_D3D12_DRED=1` | on a device removal, report which shader was running |
| `GGML_D3D12_STATS=1` | print dispatch counts, timings, and every op that fell back to the processor |
| `GGML_D3D12_PROFILE=1` | per-shader GPU timing |
| `GGML_D3D12_ONLY=<text>` | use only the GPUs whose name contains `<text>` (e.g. with an iGPU and a dGPU) |
| `GGML_D3D12_DISABLE=1` | hide the GPU entirely, so you can compare against the processor |
| `GGML_D3D12_DISABLE_OPS=<op>,<op>` | run these ops on the processor (`ALL` = every op) |
| `GGML_D3D12_MAX_ALLOC_MB=<n>` | cap the size of one GPU buffer |
| `GGML_D3D12_TILED=<n>` | use the tiled prompt matmul from `n` columns up (0 = off). Default: 16, except Intel (5 columns for Q4_0, Q8_0, Q4_K, Q6_K) and AMD (5 columns for Q6_K); setting it switches that table off |
| `GGML_D3D12_TILED_MMID=<n>` | the same for mixture-of-experts layers (default 16) |
| `GGML_D3D12_MV_MULTI=0` | turn off the multi-column mat-vec for 2 to 15 prompt tokens (Q4_0, Q8_0) |
| `GGML_D3D12_SM=<60..67>` | cap the shader model (60 = 6.0) |
| `GGML_D3D12_NO_FUSE=1` | do not fuse neighbouring ops into one shader |
| `GGML_D3D12_ALL_BARRIERS=1` | a barrier after every dispatch (slower; for checking a suspected sync bug) |
| `GGML_D3D12_NO_SHADER_CACHE=1` | do not read or write the compiled shader cache (`d3d12-shader-cache`) |
| `GGML_D3D12_DEBUG=1` | enable the D3D12 debug layer (needs the Graphics Tools Windows feature) |
| `GGML_D3D12_WARP=1` | also list the WARP software adapter |

---

## Licence

llama.cpp is MIT. The DirectX Shader Compiler DLLs are Microsoft's, under the licences included
in the zip.
