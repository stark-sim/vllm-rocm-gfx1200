# vllm-rocm-gfx1200

在 **RX 9060 XT(gfx1200, RDNA4)** 上运行 vLLM 的适配补丁集与构建/运行脚本。

vLLM 官方不为 ROCm 分发 vendored flash-attention,RDNA 消费卡的官方 attention 路径是
Triton 实现(TRITON_ATTN);本适配解决的是更底层的问题:让 vLLM 在这张卡上**编译通过、
能跑**。

## 兼容性矩阵(已验证)

| 组件 | 版本 |
|------|------|
| GPU | RX 9060 XT(gfx1200, RDNA4) |
| ROCm | 7.13(theRock / _rocm_sdk  wheels) |
| PyTorch | 2.13.0a0+rocm7.13.0a20260416 |
| vLLM | 0.23.1rc1.dev905+g3f99883d9(commit `3f99883d9`) |
| Python | 3.11(conda env `vllm-rocm`) |

## 补丁清单(`patches/`)

| 补丁 | 文件 | 解决什么 |
|------|------|----------|
| 0001 | `csrc/spinloop.cpp` | ROCm Clang 23 不能直接包含 `<mwaitxintrin.h>`,改 `<x86intrin.h>` |
| 0002 | `CMakeLists.txt`、`torch_bindings.cpp`、`q_gemm.cu` | 禁用 GPTQ 路径(HIP 缺 half2 atomicAdd,编译失败) |
| 0003 | `vllm/platforms/__init__.py` | amdsmi 在本环境不可用时,用 `torch.version.hip` 兜底识别 ROCm 平台 |
| 0004 | `vllm/platforms/rocm.py` | `_get_gcn_arch` 走 torch.cuda,支持 `HCP_ROCM_GCN_ARCH` 环境变量覆盖 |
| 0005 | `pyproject.toml` | 解除 `torch==2.11.0` 构建钉版,用环境里的 torch 2.13 |

## 构建

```bash
bash scripts/build_vllm_rocm_gfx1200.sh   # clone -> checkout -> apply patches -> pip install -e
```

前提:conda env 里有 torch 2.13+rocm7.13 与编译工具链。构建产物是 editable 安装,
import 指向 `vllm/` 源码树。

## 运行

vLLM 的 ROCm 库依赖一串 `_rocm_sdk*` 目录,用 wrapper 跑(**必须先 cd 到不含 vllm/
目录的路径**,否则 `import vllm` 会变成 namespace 包):

```bash
cd /tmp && ./scripts/run_vllm_rocm.sh python your_script.py
```

## 已知限制

- amdsmi 在此环境不可用(补丁 3/4 即为此兜底);GPU 显存查询走 torch.cuda。
- AITER 未安装;attention 走 TRITON_ATTN(RDNA 的官方路径)。
- 无 vendored flash-attention(那是 CUDA/XPU 专属)。

## 验证记录

- Qwen2-0.5B prefill/decode(TRITON_ATTN)。
- [hcp-vllm-plugin](https://github.com/stark-sim/hcp-vllm-plugin) 全部验证套件在本环境通过
  (ring connector/backend、多请求连续批、16k 长上下文、跨节点 CUDA↔ROCm CP)。
