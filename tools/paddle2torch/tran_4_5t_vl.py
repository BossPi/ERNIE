# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from safetensors.numpy import load_file
from safetensors.torch import save_file as save_safetensors
import torch
import numpy as np
import json
import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert Paddle safetensors model to PyTorch format."
    )

    parser.add_argument(
        "--src_dir",
        type=str,
        default="paddle_models/ERNIE-4.5-vl-28B",
        help="Path to destination PaddlePaddle model directory",
    )

    parser.add_argument(
        "--dest_dir",
        type=str,
        default="torch_models/ERNIE-4.5-vl-28B",
        help="Path to source PyTorch model directory",
    )

    return parser.parse_args()


def convert_pdparams_to_safetensors(pdparams_path, safetensors_path, config):
    """
    Convert a PaddlePaddle .pdparams file to a .safetensors file, transposing .weight tensors.
    Args:
        pdparams_path (str): Path to the input .pdparams file.
        safetensors_path (str): Path to the output .safetensors file.
    """
    print("----------------------------------------------------------------")
    print("pdparams_path:", pdparams_path)
    # Load the PaddlePaddle model state dictionary
    torch_state_dict = {}
    pd_tensors = load_file(pdparams_path)
    for key, param in pd_tensors.items():
        if param.dtype != "float32":
            param = (param.astype(np.uint32) << 16).view(np.float32)
        if (
            (key.endswith(".weight") or key.endswith(".weight_1"))
            and "embed_tokens" not in key
            and param.ndim == 2
        ):
            param = param.T  # Transpose the parameter
        tensor = torch.from_numpy(param)
        if (
            "mlp.gate.weight" not in key
            and "mlp.moe_statics.e_score_correction_bias" not in key
        ):
            tensor = tensor.to(torch.bfloat16)

        key = re.sub("^vision_model", "vision_tower", key)
        key = re.sub("^ernie", "language_model", key)
        key = re.sub("^language_model.resampler_model", "resampler_model", key)
        key = "model." + key
        if "vision_tower" not in key:
            if "qkv_proj" in key:
                N, D = tensor.shape
                num_heads = config["num_attention_heads"]
                kv_heads = config["num_key_value_heads"]
                head_dim = D // num_heads

                hidden_size = head_dim * num_heads
                kv_hidden_size = head_dim * kv_heads

                assert (
                    N == hidden_size + 2 * kv_hidden_size
                ), f"{N} != {hidden_size} + {2*kv_hidden_size}, \
                    qkv_proj.weight dimensions mismatch the model configuration, \
                    please check config.json"
                q_proj, k_proj, v_proj = torch.split(
                    tensor, [hidden_size, kv_hidden_size, kv_hidden_size], dim=0
                )
                torch_state_dict[key.replace("qkv_proj", "q_proj")] = (
                    q_proj.contiguous()
                )
                torch_state_dict[key.replace("qkv_proj", "k_proj")] = (
                    k_proj.contiguous()
                )
                torch_state_dict[key.replace("qkv_proj", "v_proj")] = (
                    v_proj.contiguous()
                )
            elif "mlp" in key:
                if "moe_statics" in key:
                    suffix = "moe_statics.e_score_correction_bias"
                    converted_key = key.removesuffix(suffix)
                    # splitting param (2, ...) to 2 * (1, ...)
                    torch_state_dict[converted_key + "text_moe." + suffix] = tensor[0][
                        None, :
                    ].contiguous()
                    torch_state_dict[converted_key + "vision_moe." + suffix] = tensor[
                        1
                    ][None, :].contiguous()
                elif "gate.weight" in key:
                    moe_type = "text_moe"
                    if "weight_1" in key:
                        moe_type = "vision_moe"
                    suffix = "gate.weight"
                    converted_key = key.removesuffix("_1")  # vision
                    converted_key = converted_key.removesuffix("gate.weight")
                    torch_state_dict[converted_key + f"{moe_type}." + suffix] = (
                        tensor.contiguous()
                    )
                elif ".experts" in key:
                    moe_type = "text_moe"
                    expert_number = int(re.findall(r"\d+", key)[-1])
                    # 128 experts split into 64 each (text, vision)
                    if expert_number >= 64:
                        moe_type = "vision_moe"
                        expert_number -= 64
                    # avoid subbing the layer idx + experts twice
                    prefix = re.findall(
                        r"model.language_model.layers.\d+.mlp.experts.", key
                    )[0]
                    converted_key = re.sub(
                        r"\d+",
                        f"{moe_type}.experts.{expert_number}",
                        key.removeprefix(prefix),
                    )
                    full_key = re.sub(".experts", "", prefix) + converted_key
                    if "up_gate_proj" in full_key:
                        fused_dim, hidden_size = tensor.shape
                        # auto infer intermediate_size
                        assert (
                            fused_dim % 2 == 0
                        ), f"Cannot split tensor with odd dimension {fused_dim}. \
                            Specify intermediate_size."
                        intermediate_size = fused_dim // 2

                        assert (
                            fused_dim == 2 * intermediate_size
                        ), f"Tensor shape {tensor.shape} does not match 2 * intermediate_size {2 * intermediate_size}"

                        gate_proj, up_proj = torch.split(
                            tensor, [intermediate_size, intermediate_size], dim=0
                        )
                        torch_state_dict[
                            full_key.replace("up_gate_proj", "gate_proj")
                        ] = gate_proj.contiguous()
                        torch_state_dict[
                            full_key.replace("up_gate_proj", "up_proj")
                        ] = up_proj.contiguous()
                    else:
                        torch_state_dict[full_key] = tensor.contiguous()
                elif "up_gate_proj" in key:
                    fused_dim, hidden_size = tensor.shape
                    # auto infer intermediate_size
                    assert (
                        fused_dim % 2 == 0
                    ), f"Cannot split tensor with odd dimension {fused_dim}. \
                        Specify intermediate_size."
                    intermediate_size = fused_dim // 2

                    assert (
                        fused_dim == 2 * intermediate_size
                    ), f"Tensor shape {tensor.shape} does not match 2 * intermediate_size {2 * intermediate_size}"

                    gate_proj, up_proj = torch.split(
                        tensor, [intermediate_size, intermediate_size], dim=0
                    )
                    torch_state_dict[key.replace("up_gate_proj", "gate_proj")] = (
                        gate_proj.contiguous()
                    )
                    torch_state_dict[key.replace("up_gate_proj", "up_proj")] = (
                        up_proj.contiguous()
                    )
                else:
                    torch_state_dict[key] = tensor.contiguous()
            elif "lm_head" in key:
                torch_state_dict["lm_head"] = tensor.contiguous()
            elif "spatial_linear" in key or "temporal_linear" in key:
                sequential_number = int(re.findall(r"\d+", key)[-1])

                if sequential_number == 0:
                    converted_key = re.sub(r"(?<=\.)\d+(?=\.)", "fc1", key)
                elif sequential_number == 2:
                    converted_key = re.sub(r"(?<=\.)\d+(?=\.)", "fc2", key)
                elif sequential_number == 3:
                    converted_key = re.sub(r"(?<=\.)\d+(?=\.)", "ln", key)
                else:
                    converted_key = key

                torch_state_dict[converted_key] = tensor.contiguous()
            else:
                torch_state_dict[key] = tensor.contiguous()
        else:
            torch_state_dict[key] = tensor.contiguous()
    return torch_state_dict


def convert_multiple_pdparams_to_safetensors(input_dir, output_dir):
    """
    Convert all .pdparams files in a directory to .safetensors files.
    Args:
        input_dir (str): Directory containing .pdparams files.
        output_dir (str): Directory to save .safetensors files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # indexing base dict
    index_dict = {"metadata": {"total_size": 0}, "weight_map": {}}

    with open(os.path.join(input_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(
        os.path.join(input_dir, "model.safetensors.index.json"), "r", encoding="utf-8"
    ) as f:
        index = json.load(f)
    index_dict["metadata"] = index["metadata"]

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".safetensors"):
            pdparams_path = os.path.join(input_dir, filename)
            safetensors_path = os.path.join(output_dir, filename)
            torch_state_dict = convert_pdparams_to_safetensors(
                pdparams_path, safetensors_path, config
            )
            save_safetensors(torch_state_dict, safetensors_path)

            # remap namings in index
            for k in torch_state_dict.keys():
                index_dict["weight_map"][k] = filename

            print(f"Converted {filename} to {safetensors_path}")

    # save index
    with open(
        os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(index_dict, f, indent=2)


# Example usage
if __name__ == "__main__":
    args = parse_arguments()

    convert_multiple_pdparams_to_safetensors(args.src_dir, args.dest_dir)
