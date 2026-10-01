# Third-party components

This repository contains only project-authored source code (Apache-2.0, see `LICENSE`).
No third-party source code, model weights, or datasets are included.

| Component | Use | License | Included here |
| --- | --- | --- | --- |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Qwen team, Alibaba Cloud) | Base model, downloaded at runtime | Apache-2.0 | No |
| [sennaLLMLearner/qwen3-4b-system-one-lora](https://huggingface.co/sennaLLMLearner/qwen3-4b-system-one-lora) | LoRA adapter, downloaded at runtime | Apache-2.0 | No |
| [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Runtime dependencies (imported) | BSD-3-Clause / Apache-2.0 | No |
| [MLX](https://github.com/ml-explore/mlx), [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple Silicon runtime dependencies (imported) | MIT | No |

The adapter was trained by distillation from DeepSeek API outputs. Teacher outputs and training texts are not distributed.
This project is not affiliated with or endorsed by Alibaba Cloud/Qwen, DeepSeek, or TypeSafe.
