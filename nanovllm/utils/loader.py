import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
import re


from nanovllm.layers.fused_moe import FusedMoE


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def get_all_fused_moe_layers(module):
    moes = []
    for child in module.children():
        if isinstance(child, FusedMoE):
            moes.append(child)
        else:
            moes.extend(get_all_fused_moe_layers(child))
    return moes


def find_fused_moe_layer(model, layer_idx, prefix=""):
    layer = model
    parts = []
    if prefix:
        parts.extend(prefix.split("."))
    parts.extend(["model", "layers", str(layer_idx), "mlp", "experts"])

    for part in parts:
        if hasattr(layer, part):
            layer = getattr(layer, part)
        elif isinstance(layer, nn.ModuleList) and part.isdigit():
            layer = layer[int(part)]
        else:
            return None
    if isinstance(layer, FusedMoE):
        return layer
    return None




deepseek_routed_expert_pattern = re.compile(
    r"^(?:language|language_model)\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|up_proj|gate_proj)\.weight$"
)

deepseek_shared_expert_pattern = re.compile(
    r"^(?:language|language_model)\.model\.layers\.(\d+)\.mlp\.shared_experts\."
    r"(down_proj|up_proj|gate_proj)\.weight$"
)

deepseek_v2_routed_expert_pattern = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|up_proj|gate_proj)\.weight$"
)
deepseek_v2_shared_expert_pattern = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\."
    r"(down_proj|up_proj|gate_proj)\.weight$"
)

proj_to_param_shard = {
    "down_proj": ("w2_weight", "w2"),
    "gate_proj": ("w13_weight", "w1"),
    "up_proj":   ("w13_weight", "w3"),
    "gate_up_proj_bias": ("w13_bias", "w13"),
    "down_proj_bias": ("w2_bias", "w2"),
    "gate_up_proj_blocks": ("w13_weight", "w13"),
    "down_proj_blocks": ("w2_weight", "w2"),
}



def _is_deepseek_vl2(model: nn.Module) -> bool:
    return all(hasattr(model, a) for a in ("language", "projector", "vision"))


def _is_kimi_vl(model: nn.Module) -> bool:
    return all(
        hasattr(model, attr)
        for attr in ("language", "vision_tower", "multi_modal_projector")
    )


def _is_deepseek_v2_pure(model: nn.Module) -> bool:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        try:
            first_layer = model.model.layers[0]
        except Exception:
            return False
        
        return hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "kv_a_proj_with_mqa")
    return False


def load_model(model: nn.Module, path: str):
    is_kimi = _is_kimi_vl(model)
    is_vl2 = _is_deepseek_vl2(model)
    is_dsv2 = _is_deepseek_v2_pure(model) and not is_vl2 and not is_kimi

    if is_kimi:
        _load_kimi_vl(model, path)
    elif is_vl2:
        _load_deepseek_vl2(model, path)
    elif is_dsv2:
        _load_deepseek_v2_pure(model, path)
    else:
        raise ValueError("Unsupported model architecture; only DeepSeek-VL2 and Kimi-VL are supported")


def _load_deepseek_v2_pure(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                
                if weight_name.startswith(("vision.", "projector.")):
                    continue

                
                stripped = weight_name
                if stripped.startswith("language."):
                    stripped = stripped[len("language."):]

                _load_one_dsv2_weight(f, model, weight_name, stripped, packed_modules_mapping)


def _load_deepseek_vl2(model: nn.Module, path: str):
    
    packed_modules_mapping = getattr(model.language, "packed_modules_mapping",
                                     getattr(model, "packed_modules_mapping", {}))

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                
                if weight_name.startswith("vision."):
                    
                    
                    
                    sub_name = weight_name[len("vision."):]
                    try:
                        param = model.vision.vit.get_parameter(sub_name)
                        param.data.copy_(f.get_tensor(weight_name))
                    except AttributeError:
                        print(f"[Warning] vision param not found: {sub_name}")
                    continue

                
                if weight_name.startswith("projector."):
                    sub_name = weight_name[len("projector."):]
                    try:
                        param = model.projector.proj.get_parameter(sub_name)
                        param.data.copy_(f.get_tensor(weight_name))
                    except AttributeError:
                        print(f"[Warning] projector param not found: {sub_name}")
                    continue

                
                if weight_name in ("image_newline", "view_seperator", "tile_indicators"):
                    try:
                        param = model.get_parameter(weight_name)
                        param.data.copy_(f.get_tensor(weight_name))
                    except AttributeError:
                        print(f"[Warning] top-level param not found: {weight_name}")
                    continue

                
                if weight_name.startswith("language."):
                    stripped = weight_name[len("language."):]
                    _load_one_dsv2_weight(f, model.language, weight_name, stripped,
                                          packed_modules_mapping)
                    continue

                
                try:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                except AttributeError:
                    print(f"[Warning] top-level param not found: {weight_name}")


def _load_kimi_vl(model: nn.Module, path: str):
    """Load Kimi-VL checkpoint names into the shared NanoVLLM text runtime."""
    packed_modules_mapping = getattr(
        model.language,
        "packed_modules_mapping",
        getattr(model, "packed_modules_mapping", {}),
    )
    correction_pattern = re.compile(
        r"^language_model\.model\.layers\.(\d+)\.mlp\.gate\."
        r"e_score_correction_bias$"
    )

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name.startswith("vision_tower."):
                    sub_name = weight_name[len("vision_tower."):]
                    try:
                        param = model.vision_tower.get_parameter(sub_name)
                        param.data.copy_(f.get_tensor(weight_name))
                    except AttributeError:
                        print(f"[Warning] Kimi vision param not found: {sub_name}")
                    continue

                if weight_name.startswith("multi_modal_projector."):
                    sub_name = weight_name[len("multi_modal_projector."):]
                    try:
                        param = model.multi_modal_projector.get_parameter(sub_name)
                        param.data.copy_(f.get_tensor(weight_name))
                    except AttributeError:
                        print(f"[Warning] Kimi projector param not found: {sub_name}")
                    continue

                correction_match = correction_pattern.fullmatch(weight_name)
                if correction_match:
                    layer_idx = int(correction_match.group(1))
                    param_path = (
                        f"model.layers.{layer_idx}.mlp.experts."
                        "e_score_correction_bias"
                    )
                    param = model.language.get_parameter(param_path)
                    param.data.copy_(f.get_tensor(weight_name))
                    continue

                if weight_name.startswith("language_model."):
                    stripped = weight_name[len("language_model."):]
                    _load_one_dsv2_weight(
                        f,
                        model.language,
                        weight_name,
                        stripped,
                        packed_modules_mapping,
                    )
                    continue

                print(f"[Warning] unsupported Kimi weight: {weight_name}")


def _load_one_dsv2_weight(safe_file, model_for_param, original_name, stripped_name,
                          packed_modules_mapping):
    
    m = deepseek_routed_expert_pattern.fullmatch(original_name)
    if m:
        layer_idx = int(m.group(1))
        expert_id = int(m.group(2))
        proj = m.group(3)
        param_name, shard_id = proj_to_param_shard[proj]
        param_path = f"model.layers.{layer_idx}.mlp.experts.{param_name}"
        moe_layer = find_fused_moe_layer(model_for_param, layer_idx)
        assert moe_layer is not None, f"FusedMoE not found for layer {layer_idx}"
        param = model_for_param.get_parameter(param_path)
        loaded_weight = safe_file.get_tensor(original_name)
        moe_layer.weight_loader(param, loaded_weight, original_name, shard_id, expert_id)
        return

    
    m = deepseek_v2_routed_expert_pattern.fullmatch(original_name)
    if m:
        layer_idx = int(m.group(1))
        expert_id = int(m.group(2))
        proj = m.group(3)
        param_name, shard_id = proj_to_param_shard[proj]
        param_path = f"model.layers.{layer_idx}.mlp.experts.{param_name}"
        moe_layer = find_fused_moe_layer(model_for_param, layer_idx)
        assert moe_layer is not None, f"FusedMoE not found for layer {layer_idx}"
        param = model_for_param.get_parameter(param_path)
        loaded_weight = safe_file.get_tensor(original_name)
        moe_layer.weight_loader(param, loaded_weight, original_name, shard_id, expert_id)
        return

    
    m = deepseek_shared_expert_pattern.fullmatch(original_name) or \
        deepseek_v2_shared_expert_pattern.fullmatch(original_name)
    if m:
        layer_idx = int(m.group(1))
        proj = m.group(2)
        if proj in ("gate_proj", "up_proj"):
            packed = "gate_up_proj"
            shard_id = 0 if proj == "gate_proj" else 1
            param_path = f"model.layers.{layer_idx}.mlp.shared_experts.{packed}.weight"
            try:
                param = model_for_param.get_parameter(param_path)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, safe_file.get_tensor(original_name), shard_id)
            except AttributeError:
                print(f"[Warning] shared_experts gate_up not found: {param_path}")
        else:  # down_proj
            param_path = f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"
            try:
                param = model_for_param.get_parameter(param_path)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, safe_file.get_tensor(original_name))
            except AttributeError:
                print(f"[Warning] shared_experts down_proj not found: {param_path}")
        return

    
    for k in packed_modules_mapping:
        if k in stripped_name:
            v, shard_id = packed_modules_mapping[k]
            param_name = stripped_name.replace(k, v)
            try:
                param = model_for_param.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, safe_file.get_tensor(original_name), shard_id)
                return
            except AttributeError:
                continue

    try:
        param = model_for_param.get_parameter(stripped_name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, safe_file.get_tensor(original_name))
    except AttributeError:
        print(f"[Warning] param not found: {stripped_name}  (orig: {original_name})")


# ============================================================================

# ============================================================================
def setup_offload_for_model(model, offload_strategy: str = "none",
                            gpu_expert_cache_slots: int = -1,
                            host_pinned: bool = True,
                            physical_expert_cache: bool = False):
    if offload_strategy == "none":
        return

    from nanovllm.layers.fused_moe import (
        FusedMoE, configure_moe_offload,
    )

    moe_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, FusedMoE):
            moe_layers.append((name, mod))
    contract = (
        "physical-cpu-first"
        if physical_expert_cache else "logical-persistent-v1"
    )
    allocated_before = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )
    print(f"[offload] configuring {offload_strategy} cache on {len(moe_layers)} MoE layers, "
          f"slots={gpu_expert_cache_slots}, pinned={host_pinned}, contract={contract}")

    host_master_bytes = 0
    physical_gpu_bytes = 0
    full_gpu_bytes_equivalent = 0
    for name, mod in moe_layers:
        if physical_expert_cache:
            if not getattr(mod, "_physical_host_load_enabled", False):
                raise RuntimeError(
                    f"{name}: missing CPU-first expert construction state"
                )
            w13_host = mod._physical_w13_host
            w2_host = mod._physical_w2_host
            w13_bias_host = mod._physical_w13_bias_host
            w2_bias_host = mod._physical_w2_bias_host
            expected = set(range(int(mod.global_num_experts)))
            loaded = mod._physical_loaded_expert_shards
            incomplete = {
                shard: sorted(expected.difference(loaded.get(shard, set())))
                for shard in ("w1", "w2", "w3")
            }
            incomplete = {
                shard: ids for shard, ids in incomplete.items() if ids
            }
            if incomplete:
                raise RuntimeError(
                    f"{name}: incomplete CPU-first expert checkpoint load: "
                    f"{incomplete}"
                )
        else:
            w13_gpu = mod.w13_weight.data
            w2_gpu = mod.w2_weight.data
            w13_host = w13_gpu.detach().to('cpu', copy=True)
            w2_host = w2_gpu.detach().to('cpu', copy=True)
            w13_bias_host = None
            w2_bias_host = None
            if hasattr(mod, "w13_bias"):
                w13_bias_host = mod.w13_bias.data.detach().to('cpu', copy=True)
            if hasattr(mod, "w2_bias"):
                w2_bias_host = mod.w2_bias.data.detach().to('cpu', copy=True)
            if host_pinned:
                try:
                    w13_host = w13_host.pin_memory()
                    w2_host = w2_host.pin_memory()
                    if w13_bias_host is not None:
                        w13_bias_host = w13_bias_host.pin_memory()
                    if w2_bias_host is not None:
                        w2_bias_host = w2_bias_host.pin_memory()
                except RuntimeError as e:
                    print(f"  WARN: pin_memory failed for {name}: {e}")

        layer_host_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                w13_host, w2_host, w13_bias_host, w2_bias_host
            )
            if tensor is not None
        )
        layer_gpu_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                mod.w13_weight, mod.w2_weight,
                getattr(mod, "w13_bias", None),
                getattr(mod, "w2_bias", None),
            )
            if tensor is not None
        )
        host_master_bytes += layer_host_bytes
        full_gpu_bytes_equivalent += layer_host_bytes
        physical_gpu_bytes += layer_gpu_bytes

        configure_moe_offload(
            mod, w13_host=w13_host, w2_host=w2_host,
            strategy=offload_strategy,
            capacity=gpu_expert_cache_slots,
            physical=physical_expert_cache,
            w13_bias_host=w13_bias_host,
            w2_bias_host=w2_bias_host,
        )

    if physical_expert_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
    allocated_after = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )
    print(
        f"[offload] all {len(moe_layers)} MoE layers configured; "
        f"contract={contract}, allocated_before={allocated_before}, "
        f"allocated_after={allocated_after}, "
        f"allocated_saved={max(0, allocated_before - allocated_after)}, "
        f"host_master_bytes={host_master_bytes}, "
        f"physical_gpu_bytes={physical_gpu_bytes}, "
        f"full_gpu_bytes_equivalent={full_gpu_bytes_equivalent}, "
        f"expert_gpu_bytes_saved={max(0, full_gpu_bytes_equivalent - physical_gpu_bytes)}"
    )
