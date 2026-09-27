"""Multi-event autoregressive inference for pretrained and distilled weights.

This entry point uses ``wan.image2video_ar.WanI2V_AR`` with an explicit
multi-event ``navigation/editing_N`` annotation contract and per-chunk labels:

.. code-block:: json

    {
      "editmeta": {
        "edit_instruction_1": "Change the world for the first event.",
        "edit_instruction_2": "Change the world for the second event."
      },
      "editing_annotation": {
        "0": {"state": "navigation"},
        "1": {"state": "navigation"},
        "4": {"state": "editing_1", "reference_image": "../ref/example.png"},
        "7": {"state": "navigation"},
        "8": {"state": "editing_2"},
        "11": {"state": "navigation"}
      }
    }

Every chunk must be labelled ``navigation`` or ``editing_N``.
Navigation chunks receive an empty local prompt (the
scene is always carried by ``a_B``); an ``editing_N`` chunk receives
``editmeta.edit_instruction_N`` as its exact ``[Instruction:]`` prompt. Each chunk p
uses the model-native GCA context ``a_B + a_p + a_{p-1} + a_{p-2}``.
No ``[State:]`` tag is emitted. A chunk attends only to the reference image
specified on that chunk in ``editing_annotation``; chunks without that field
do not attend to a reference image.
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from wan.commons.parallel_states import initialize_parallel_state
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.image2video_ar import WanI2V_AR
from wan.utils.prompt_template import (
    compose_scene_text_condition,
    compose_chunk_text_condition,
)
from wan.utils.utils import save_video

DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parent / "ckpt"


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def validate_checkpoint_dir(checkpoint_dir: Path, mode: str) -> None:
    required_files = (
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        f"{mode}/low_noise_model/config.json",
        f"{mode}/low_noise_model/diffusion_pytorch_model.safetensors",
        f"{mode}/high_noise_model/config.json",
        f"{mode}/high_noise_model/diffusion_pytorch_model.safetensors",
    )
    missing = [name for name in required_files if not (checkpoint_dir / name).is_file()]
    if not (checkpoint_dir / "google" / "umt5-xxl").is_dir():
        missing.append("google/umt5-xxl/")
    if missing:
        raise FileNotFoundError(
            f"Incomplete model directory: {checkpoint_dir}\nMissing: "
            + ", ".join(missing)
            + "\nPlace the inference weights under the selected checkpoint directory."
        )


def resolve_input_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidate = (base_dir / raw_path).resolve()
    if candidate.exists():
        return candidate

    candidate = (Path.cwd() / raw_path).resolve()
    if candidate.exists():
        return candidate

    return (base_dir / raw_path).resolve()


def load_case_from_json(
    json_path: Path,
    latent_window_size: int,
    temporal_stride: int,
) -> dict:
    """Load one explicitly annotated multi-event case."""
    with json_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    caption = meta.get("caption", {})
    editmeta = meta.get("editmeta", {})
    text = (caption.get("SceneDescription") or caption.get("SceneSummary") or "").strip()
    annotation = meta.get("editing_annotation") or {}
    poses = meta.get("poses_w2c") or []
    total_frames_json = len(poses)
    if total_frames_json <= 0:
        raise ValueError(f"json camera input is empty: {json_path}")
    # Convert raw frames to latent frames (causal VAE:
    # 1 anchor + (T-1)//stride) THEN floor-divide by chunk. The old
    # (T-1)//(chunk*stride) dropped the anchor latent, under-counting by one
    # whenever total_latent is an exact multiple of chunk_size.
    total_latent = (total_frames_json - 1) // temporal_stride + 1
    adaptive_chunks = total_latent // latent_window_size
    if adaptive_chunks < 1:
        raise ValueError(
            f"json camera input only provides {total_frames_json} frames, which is not enough for one AR chunk"
        )
    adaptive_frame_num = (adaptive_chunks * latent_window_size - 1) * temporal_stride + 1
    chunk_labels, chunk_ref_ids, ref_paths = parse_chunk_annotations(
        annotation, json_path, adaptive_chunks,
    )
    edit_instructions = _extract_indexed_prompts(editmeta, "edit_instruction")
    bg_prompt, chunk_prompts, chunk_has_edit = build_chunk_prompts(
        text=text,
        edit_instructions=edit_instructions,
        annotation={str(k): label for k, label in enumerate(chunk_labels)},
        num_chunks=adaptive_chunks,
    )
    return {
        "bg_prompt": bg_prompt,
        "chunk_prompts": chunk_prompts,
        "chunk_has_edit": chunk_has_edit,
        "chunk_ref_ids": chunk_ref_ids,
        "ref_paths": ref_paths,
        "chunk_labels": chunk_labels,
        "text": text,
        "edit_instructions": edit_instructions,
        "adaptive_frame_num": adaptive_frame_num,
        "adaptive_chunks": adaptive_chunks,
        "total_frames_json": total_frames_json,
        "editing_annotation": annotation,
    }


_EDITING_LABEL = re.compile(r"editing_([1-9]\d*)")


def parse_chunk_annotations(annotation: dict, json_path: Path, num_chunks: int):
    if not isinstance(annotation, dict):
        raise ValueError("editing_annotation must be a JSON object")
    expected = {str(i) for i in range(num_chunks)}
    if set(annotation) != expected:
        raise ValueError(
            f"editing_annotation must cover chunks 0..{num_chunks - 1} exactly"
        )
    labels, ref_ids, ref_paths, ref_keys = [], [], [], []
    for index in range(num_chunks):
        item = annotation[str(index)]
        if not isinstance(item, dict) or "state" not in item:
            raise ValueError(f"editing_annotation[{index}] must contain a state")
        if set(item) - {"state", "reference_image"}:
            raise ValueError(f"editing_annotation[{index}] has unsupported fields")
        state = item["state"]
        if not isinstance(state, str) or not (state == "navigation" or _EDITING_LABEL.fullmatch(state)):
            raise ValueError(f"editing_annotation[{index}].state must be navigation or editing_N")
        has_ref_field = "reference_image" in item
        raw_ref = item.get("reference_image")
        if has_ref_field:
            if not state.startswith("editing_") or not isinstance(raw_ref, str) or not raw_ref.strip():
                raise ValueError(
                    f"editing_annotation[{index}].reference_image requires editing_N and a path"
                )
            path = (json_path.parent / raw_ref).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Reference image not found for chunk {index}: {path}")
            ref_key = (path, state)
            if ref_key not in ref_keys:
                ref_keys.append(ref_key)
                ref_paths.append(path)
            ref_id = ref_keys.index(ref_key)
        else:
            ref_id = None
        labels.append(state)
        ref_ids.append(ref_id)
    return labels, ref_ids, ref_paths


def _extract_indexed_prompts(editmeta, prefix):
    pattern = re.compile(rf"{re.escape(prefix)}_([1-9]\d*)")
    result = {}
    for raw_key, raw_prompt in editmeta.items():
        match = pattern.fullmatch(str(raw_key).strip())
        if not match:
            continue
        index = int(match.group(1))
        prompt = str(raw_prompt or "").strip()
        if not prompt:
            raise ValueError(f"editmeta.{prefix}_{index} is empty")
        result[index] = prompt
    return result


def build_chunk_prompts(
    *,
    text,
    edit_instructions,
    annotation,
    num_chunks,
):
    """Build one local prompt ``a_i`` for every labelled AR chunk.

    The annotation is intentionally strict: all chunk indices must be present,
    and every value must be ``navigation`` or ``editing_N``.
    This prevents a missing annotation from silently applying one
    combined event to the entire video.
    """
    bg_prompt = compose_scene_text_condition(text)
    if not isinstance(annotation, dict) or not annotation:
        raise ValueError(
            "multi-event AR requires a non-empty editing_annotation with one "
            "navigation/editing_N label per chunk"
        )

    expected_keys = {str(k) for k in range(num_chunks)}
    actual_keys = {str(key) for key in annotation}
    missing = sorted(expected_keys - actual_keys, key=int)
    extra = sorted(
        actual_keys - expected_keys,
        key=lambda item: (0, int(item)) if item.isdigit() else (1, item),
    )
    if missing or extra:
        raise ValueError(
            "editing_annotation must cover exactly all AR chunks: "
            f"missing={missing}, extra={extra}, num_chunks={num_chunks}"
        )

    prompts = []
    chunk_has_edit = []
    for k in range(num_chunks):
        label = str(annotation[str(k)]).strip().lower()
        if label == "navigation":
            prompts.append("")
            chunk_has_edit.append(0)
            continue
        match = _EDITING_LABEL.fullmatch(label)
        if not match:
            raise ValueError(
                f"editing_annotation[{k!r}]={annotation[str(k)]!r}; "
                "expected navigation or editing_N"
            )

        event_index = int(match.group(1))
        event_text = edit_instructions.get(event_index)
        if not event_text:
            raise ValueError(
                f"editmeta.edit_instruction_{event_index} is missing or empty"
            )

        prompts.append(
            compose_chunk_text_condition(
                event_text,
            )
        )
        chunk_has_edit.append(1)
    return bg_prompt, prompts, chunk_has_edit


def parse_txt_file(txt_file_path: str):
    samples = []
    txt_path = Path(txt_file_path).resolve()
    txt_dir = txt_path.parent
    with txt_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [item.strip() for item in line.split("@")]
            if len(parts) != 2:
                raise ValueError(
                    f"Manifest line {line_num} must be image@json; "
                    "put reference_image in the JSON chunk annotation"
                )
            image_path, json_path = parts[0], parts[1]
            image_path = resolve_input_path(image_path, txt_dir)
            json_path = resolve_input_path(json_path, txt_dir)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found for line {line_num}: {image_path}")
            if not json_path.exists():
                raise FileNotFoundError(f"Json not found for line {line_num}: {json_path}")
            if json_path.suffix != ".json":
                raise ValueError(
                    f"Batch inference expects image@json lines, got {json_path} on line {line_num}"
                )

            samples.append(
                {
                    "case_id": json_path.stem,
                    "image_path": image_path,
                    "json_path": json_path,
                }
            )
    return samples


def setup_distributed(sp_size: int):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if sp_size > world_size:
        raise ValueError(f"sp_size ({sp_size}) cannot exceed world_size ({world_size})")
    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"
        )

    initialize_parallel_state(sp=sp_size, dp_replicate=max(world_size // sp_size, 1))
    return rank, world_size, local_rank


def build_pipeline(args, rank: int, local_rank: int):
    if args.size not in SUPPORTED_SIZES["i2v-A14B"]:
        raise ValueError(f"Unsupported size: {args.size}")

    config = WAN_CONFIGS["i2v-A14B"]
    enable_fsdp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    pipeline = WanI2V_AR(
        config=config,
        checkpoint_dir=args.checkpoint_root,
        expert_subdir=args.inference_mode,
        control_type=args.control_type,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=enable_fsdp,
        dit_fsdp=enable_fsdp,
        t5_cpu=args.t5_cpu,
        latent_window_size=args.latent_window_size,
        sink_chunks=args.sink_chunks,
        recent_chunks=args.recent_chunks,
    )
    return pipeline


def make_output_path(save_dir: Path, output_stem: str) -> Path:
    return save_dir / f"{output_stem}.mp4"


def save_output(video, save_dir: Path, output_stem: str, fps: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = make_output_path(save_dir, output_stem)
    save_video(video.unsqueeze(0), save_file=str(output_path), fps=fps)
    return output_path


def main():
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--inference_mode", choices=["pretrain", "distilled"],
                             default="pretrain")
    mode, _ = mode_parser.parse_known_args()
    mode = mode.inference_mode
    distilled = mode == "distilled"
    parser = argparse.ArgumentParser(
        description=f"EditWorld multi-event {mode} autoregressive inference"
    )
    parser.add_argument("--inference_mode", choices=["pretrain", "distilled"],
                        default="pretrain")
    parser.add_argument("--control_type", type=str, choices=["cam", "act"], required=True)
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="Shared weight root containing T5, VAE, tokenizer, and the selected "
             "mode's low/high DiT experts (default: <repository>/ckpt).",
    )
    parser.add_argument("--txt_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sp_size", type=int, default=8)
    parser.add_argument("--vis_ui", action="store_true")
    parser.add_argument("--latent_window_size", type=int, default=4)
    parser.add_argument("--recent_chunks", type=int, default=2,
                        help="Recent window in CHUNKS. With sparse memory this is the "
                             "GPU-resident recent set; rolling-fallback capacity is DERIVED = "
                             "(sink + recent + 1) * chunk latents. <0 = unbounded.")
    parser.add_argument("--sink_chunks", type=int, default=1,
                        help="Protected sink in CHUNKS (never evicted).")
    parser.add_argument("--sparse_mem_topk", type=int, default=2,
                        help="Sparse GCA memory: per layer attend "
                             "sink + recent + top-k retrieved middle-history chunks. "
                             "Default 2 selects two middle-history chunks; "
                             "0 = off (rolling sink+window cache). Requires chunk-aligned "
                             "--sink_chunks/--recent_chunks (e.g. 1/2).")
    parser.add_argument("--sparse_mem_offload", type=parse_bool, nargs="?", const=True, default=True,
                        help="Keep middle-history KV blocks on CPU and move retrieved ones "
                             "to device per chunk (default: true; pass false to disable).")
    if not distilled:
        parser.add_argument("--sampling_steps", type=int, default=None)
        parser.add_argument("--shift", type=float, default=None)
        parser.add_argument("--guide_scale", type=float, default=None,
                            help="CFG guidance scale. Default uses config.sample_guide_scale.")
        parser.add_argument("--n_prompt", type=str, default=None,
                            help="Negative prompt for CFG; defaults to the model config.")
    parser.add_argument(
        "--t5_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run T5 text encoder on CPU.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_root)
    validate_checkpoint_dir(checkpoint_dir, mode)
    config = WAN_CONFIGS["i2v-A14B"]
    pcm_phase_sigmas = None
    if distilled:
        from wan.utils.pcm_inference import load_phase_sigmas
        pcm_phase_sigmas = load_phase_sigmas(checkpoint_dir / mode, float(config.boundary))
    rank, world_size, local_rank = setup_distributed(args.sp_size)

    pipeline = build_pipeline(args, rank, local_rank)

    target_size = SIZE_CONFIGS[args.size]
    # Fail-fast: H/W must divide by (vae_spatial_stride * patch) so VAE latents,
    # patchify, and the plucker (h c1)/(w c2) folding all divide cleanly (else it
    # only fails late inside VAE encode / einops.rearrange).
    _mh = int(config.vae_stride[1]) * int(config.patch_size[1])
    _mw = int(config.vae_stride[2]) * int(config.patch_size[2])
    if target_size[0] % _mh != 0 or target_size[1] % _mw != 0:
        raise ValueError(
            f"--size {args.size} -> target_size {target_size} must be divisible "
            f"by (vae_stride*patch)=({_mh}x{_mw})"
        )
    sampling_steps = 4 if distilled else (args.sampling_steps if args.sampling_steps is not None else config.sample_steps)
    shift = config.sample_shift if distilled else (args.shift if args.shift is not None else config.sample_shift)
    guide_scale = None if distilled else (args.guide_scale if args.guide_scale is not None else config.sample_guide_scale)

    all_cases = parse_txt_file(args.txt_file)
    save_dir = Path(args.save_dir)

    if rank == 0 and args.txt_file:
        print(
            f"[infer] found {len(all_cases)} batch samples in {args.txt_file}",
            flush=True,
        )

    for index, case in enumerate(all_cases, 1):
        case_meta = load_case_from_json(
            case["json_path"],
            latent_window_size=args.latent_window_size,
            temporal_stride=int(config.vae_stride[0]),
        )
        chunk_prompts = case_meta["chunk_prompts"]
        chunk_has_edit = case_meta["chunk_has_edit"]
        chunk_ref_ids = case_meta["chunk_ref_ids"]
        chunk_labels = case_meta["chunk_labels"]
        bg_prompt = case_meta["bg_prompt"]
        adaptive_frame_num = case_meta["adaptive_frame_num"]
        num_chunks = case_meta["adaptive_chunks"]
        output_stem = f"case {index}"
        if rank == 0 and args.txt_file:
            print(
                f"[infer] sample {index}/{len(all_cases)} "
                f"case_id={case['case_id']} image={case['image_path']} json={case['json_path']} "
                f"frames={case_meta['total_frames_json']} "
                f"adaptive_frame_num={adaptive_frame_num} chunks={num_chunks} "
                f"annotation_labels={chunk_labels} "
                f"edit_instructions={case_meta['edit_instructions']} "
                f"ref_chunks={[k for k, ref_id in enumerate(chunk_ref_ids) if ref_id is not None]} "
                f"(edit_instruction_N selected by editing_N, no [State:])",
                flush=True,
            )

        expected_output_path = make_output_path(save_dir, output_stem)
        if expected_output_path.exists():
            if rank == 0:
                print(f"[infer] skip existing output: {expected_output_path}", flush=True)
            continue

        try:
            image = Image.open(case["image_path"]).convert("RGB")
            ar_ref_images = [Image.open(rp).convert("RGB") for rp in case_meta["ref_paths"]]
            if ar_ref_images and rank == 0:
                print(f"[infer] {len(ar_ref_images)} reference image(s)", flush=True)
            videos = pipeline.generate(
                chunk_prompts=chunk_prompts,
                img=image,
                json_path=str(case["json_path"]),
                bg_prompt=bg_prompt,
                vis_ui=args.vis_ui,
                target_size=target_size,
                frame_num=adaptive_frame_num,
                shift=shift,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=None if distilled else args.n_prompt,
                seed=args.seed,
                chunk_has_edit=chunk_has_edit,
                chunk_ref_ids=chunk_ref_ids,
                ref_images=ar_ref_images,
                sparse_mem_topk=args.sparse_mem_topk,
                sparse_mem_offload=args.sparse_mem_offload,
                pcm_phase_sigmas=pcm_phase_sigmas,
            )
        except ValueError as exc:
            if rank == 0:
                print(f"[infer] skip case {case['case_id']}: {exc}", flush=True)
            continue

        if rank == 0 and videos is not None:
            output_path = save_output(
                videos,
                save_dir=save_dir,
                output_stem=output_stem,
                fps=config.sample_fps,
            )
            print(f"[infer] saved {output_path}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
