"""Autoregressive inference pipeline.

Builds on the Wan i2v pipeline conventions with:
  * Uses `WanModelAR` with the cond_y-aligned autoregressive path.
  * Generates video chunk by chunk via a persistent KV cache; each chunk attends
    to the committed clean KVs of all prior chunks (uniform chunk-causal AR).
  * Routes cam_injector through fp32 autocast (already inside `WanModelAR`).
"""

import gc
import json
import os
import random
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from tqdm import tqdm

from .commons.parallel_states import get_parallel_state
from .distributed.fsdp import shard_model
from .modules.model_ar import GCAKVCache, GCA_PROMPT_WINDOW, WanModelAR
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.cam_utils import (
    compute_relative_poses,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.infer_data import (
    ResizeCropAspectCenter,
    broadcast_intrinsics_to_length,
    extract_spatialvid_meta,
    extract_translation_wasd,
)
from .utils.ar_geometry import ARGeometry, build_i2v_mask
from .utils.pcm_inference import pcm_step, validate_phase_sigmas


class WanI2V_AR:
    """Autoregressive inference pipeline.

    Chunked autoregressive generation: each forward produces
    `latent_window_size` latent frames of target given the accumulated history.
    """

    def __init__(
        self,
        config,
        checkpoint_dir,
        expert_subdir="pretrain",
        control_type=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        t5_cpu=False,
        latent_window_size=4,
        sink_chunks=0,
        recent_chunks=-1,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        # Decomposed window flags in CHUNK units; frames and the model-internal
        # local_attn_size (V2 CAPACITY semantics = sink + recent + current) are
        # DERIVED here — the rolling fallback (sparse_mem_topk=0) is correct by
        # construction and chunk alignment holds by construction.
        self.sink_chunks = int(sink_chunks)
        self.recent_chunks = int(recent_chunks)
        self.sink_frames = self.sink_chunks * int(latent_window_size)
        self.cache_frames = (
            (self.sink_chunks + self.recent_chunks + 1) * int(latent_window_size)
            if self.recent_chunks >= 0
            else -1
        )

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if control_type is not None:
            self.control_type = control_type
        elif "cam" in checkpoint_dir:
            self.control_type = "cam"
        elif "act" in checkpoint_dir:
            self.control_type = "act"
        else:
            raise ValueError(
                f"Cannot infer control_type from checkpoint path: {checkpoint_dir}. "
                "Pass control_type explicitly."
            )

        shard_fn = None
        if t5_fsdp or dit_fsdp:
            shard_fn = partial(shard_model, device_id=device_id)

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu") if t5_cpu else self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            dtype=torch.float32,
            device=self.device,
        )

        if self.rank == 0:
            print(f"[infer] loading WanModelAR from {checkpoint_dir}/{expert_subdir}", flush=True)
        if expert_subdir not in {"pretrain", "distilled"}:
            raise ValueError(f"Unsupported expert directory: {expert_subdir}")
        required_dit_dirs = (
            f"{expert_subdir}/low_noise_model",
            f"{expert_subdir}/high_noise_model",
        )
        missing_dit_dirs = [
            name
            for name in required_dit_dirs
            if not os.path.isdir(os.path.join(checkpoint_dir, name))
        ]
        if missing_dit_dirs:
            raise FileNotFoundError(
                "Dual-DiT checkpoint is incomplete; missing model directories: "
                + ", ".join(missing_dit_dirs)
            )
        model_overrides = dict(
            control_type=self.control_type,
            local_attn_size=self.cache_frames,
            sink_size=self.sink_frames,
        )
        # Load the low/high DiT experts and route them by diffusion timestep.
        def _build_one(subfolder):
            m = WanModelAR.from_pretrained(
                checkpoint_dir,
                subfolder=subfolder,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                use_safetensors=True,
                local_files_only=True,
                **model_overrides,
            )
            return self._configure_model(
                model=m,
                dit_fsdp=dit_fsdp,
            )

        self.boundary = float(config.boundary)
        if not 0.0 < self.boundary < 1.0:
            raise ValueError(
                f"Dual-DiT timestep boundary must be between 0 and 1, got {self.boundary}"
            )
        self.low_noise_model = _build_one(required_dit_dirs[0])
        self.high_noise_model = _build_one(required_dit_dirs[1])

        parallel_state = get_parallel_state()
        self.sp_size = parallel_state.sp if parallel_state.sp_enabled else 1

        self.sample_neg_prompt = config.sample_neg_prompt

        # Seed-less chunk-by-chunk AR geometry. Inference
        # derives the real chunk count from frame_num; only chunk_size /
        # frame_window_size are used at generation time.
        self.geometry = ARGeometry(
            chunk_size=int(latent_window_size),
            vae_temporal_stride=int(self.vae_stride[0]),
            patch_size=tuple(self.patch_size),
        )
        self.latent_window_size = self.geometry.latent_window_size

    # ---------- model setup ----------

    def _configure_model(self, model, dit_fsdp):
        model.eval().requires_grad_(False)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = self._apply_fsdp2_to_model(model)
        else:
            model.to(self.device)
        return model

    def _apply_fsdp2_to_model(self, model):
        param_dtype = self.param_dtype
        reduce_dtype = torch.float32
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        )
        fsdp_config = {"mp_policy": mp_policy}
        try:
            fsdp_config["mesh"] = get_parallel_state().fsdp_mesh
        except Exception:
            pass
        for block in list(model.blocks):
            fully_shard(block, **fsdp_config)
        fully_shard(model, **fsdp_config)
        return model

    # ---------- window geometry ----------

    @property
    def frame_window_size(self):
        """Number of RGB frames per AR chunk."""
        return self.geometry.frame_window_size

    # ---------- main entry ----------

    @torch.no_grad()
    def generate(
        self,
        chunk_prompts,
        img,
        json_path,
        bg_prompt="",
        vis_ui=False,
        target_size=(480, 832),
        frame_num=161,
        shift=10.0,
        sampling_steps=70,
        guide_scale=5.0,
        n_prompt=None,
        seed=42,
        offload_model=True,
        chunk_has_edit=None,
        chunk_ref_ids=None,
        ref_images=None,
        sparse_mem_topk=2,
        sparse_mem_offload=True,
        pcm_phase_sigmas=None,
    ):
        distilled = pcm_phase_sigmas is not None
        if distilled:
            pcm_phase_sigmas = validate_phase_sigmas(pcm_phase_sigmas, self.boundary)
            if sampling_steps != 4:
                raise ValueError("Distilled inference requires exactly four steps")
        if not isinstance(json_path, str) or not json_path.endswith(".json"):
            raise ValueError(
                f"Inference requires paired JSON control input, got: {json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        parsed = extract_spatialvid_meta(meta)
        json_intrinsics = parsed["intrinsics"]

        total_frames_json = len(parsed["poses"])

        # Seed-less chunk alignment: the latent
        # stream is a whole clip of N*chunk latent frames; frame 0 is the first
        # frame of chunk 0 (its image content comes only from cond_y). Truncate to
        # a whole multiple of chunk_size.
        frame_window = self.frame_window_size
        stride = int(self.vae_stride[0])
        total_latent = (frame_num - 1) // stride + 1
        num_chunks = self.geometry.num_chunks(total_latent)
        if num_chunks < 1:
            raise ValueError(
                f"frame_num={frame_num} too small for at least one AR chunk "
                f"(need >= {frame_window} frames)"
            )
        usable_frames = (num_chunks * self.latent_window_size - 1) * stride + 1
        if total_frames_json < usable_frames:
            raise ValueError(
                f"json camera input only provides {total_frames_json} frames, "
                f"but {usable_frames} are required for {num_chunks} AR chunks: {json_path}"
            )

        c2ws = parsed["poses"][:usable_frames]
        wasd_action = None
        if self.control_type == "act":
            # Compute WASD on the FULL poses then slice, so the last usable frame gets
            # the real displacement to its next frame (compute-then-
            # truncate); truncating poses first zeros the last action (extract_* appends
            # [0,0,0,0] for the frame with no successor).
            wasd_action = extract_translation_wasd(parsed["poses"])[:usable_frames]

        # ---- image + resolution setup ----
        # Build a fixed-target ResizeCropAspectCenter and apply
        # it to the first-frame image. Use the image's actual H/W (NOT json's
        # `resolution` field) as the normalization base for intrinsics, matching
        # The input frame shape is read as `video_np.shape[1:3]`.
        guide_scale = guide_scale[0] if isinstance(guide_scale, (tuple, list)) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        img_h, img_w = int(img.shape[1]), int(img.shape[2])

        target_h, target_w = int(target_size[0]), int(target_size[1])
        if target_h % (self.vae_stride[1] * self.patch_size[1]) != 0:
            raise ValueError(
                f"target_h={target_h} must be divisible by vae_stride[1]*patch_size[1]="
                f"{self.vae_stride[1] * self.patch_size[1]}"
            )
        if target_w % (self.vae_stride[2] * self.patch_size[2]) != 0:
            raise ValueError(
                f"target_w={target_w} must be divisible by vae_stride[2]*patch_size[2]="
                f"{self.vae_stride[2] * self.patch_size[2]}"
            )
        h = target_h
        w = target_w
        lat_h = target_h // self.vae_stride[1]
        lat_w = target_w // self.vae_stride[2]
        resize_crop = ResizeCropAspectCenter(h, w)
        if self.rank == 0:
            print(
                f"[infer] input image=({img_h}, {img_w}) -> target=({h}, {w}), "
                f"latent=({lat_h}, {lat_w})",
                flush=True,
            )

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        # One generator shared across all AR chunks. The RNG state therefore
        # advances continuously chunk-by-chunk: chunk N's initial noise depends
        # on chunk N-1's sampling draws. This is intentional — it gives each
        # `seed` a deterministic, end-to-end video; re-seeding per chunk would
        # produce IID chunk-initial noise but would also make each chunk's
        # sampling reproducible only in isolation.
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # ---- text embeddings for Gated Causal Attention (GCA) ----
        # Each chunk p attends to a_B + a_{p-W}..a_p: the background/scene prompt
        # plus the windowed chunk instructions (no scene or state tag) of chunks
        # p-GCA_PROMPT_WINDOW..p. ``bg_prompt`` = a_B; ``chunk_prompts`` = per-chunk a_i.
        # The unconditional (CFG) branch uses the negative prompt (``--n_prompt`` or
        # the Wan default ``sample_neg_prompt``), NOT an empty string.
        if len(chunk_prompts) != num_chunks:
            raise ValueError(
                f"chunk_prompts must have num_chunks={num_chunks} entries, got {len(chunk_prompts)}"
            )
        if chunk_has_edit is not None and len(chunk_has_edit) != num_chunks:
            raise ValueError(
                f"chunk_has_edit must have num_chunks={num_chunks} entries, got {len(chunk_has_edit)}"
            )
        if chunk_ref_ids is not None and len(chunk_ref_ids) != num_chunks:
            raise ValueError(
                f"chunk_ref_ids must have num_chunks={num_chunks} entries, got {len(chunk_ref_ids)}"
            )
        # Each editing chunk may select its own reference image; chunks without
        # a reference_image field do not attend a reference bank directly.
        selected_ref_ids = list(chunk_ref_ids or [None] * num_chunks)
        if any(ref_id is not None and
               (not isinstance(ref_id, int) or ref_id < 0 or ref_id >= len(ref_images or []))
               for ref_id in selected_ref_ids):
            raise ValueError("chunk_ref_ids contains an invalid reference image index")

        t5_device = self.device if not self.t5_cpu else torch.device("cpu")
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)

        def _enc(prompt):
            out = self.text_encoder([prompt], t5_device)[0]
            return out.to(self.device) if self.t5_cpu else out

        a_b_enc = _enc(bg_prompt)
        prompt_cache = {}
        for chunk_prompt in dict.fromkeys(chunk_prompts):
            prompt_cache[chunk_prompt] = _enc(chunk_prompt)
        # CFG unconditional branch: negative prompt (--n_prompt or sample_neg_prompt).
        context_null = None
        if not distilled:
            neg_prompt = n_prompt if n_prompt else self.sample_neg_prompt
            context_null = [_enc(neg_prompt)]
        if not self.t5_cpu and offload_model:
            self.text_encoder.model.cpu()

        def cumulative_context(p):
            # Window [a_B, a_p, a_{p-1}, a_{p-2}] in importance-DESCENDING order:
            # the model CONCATENATES the items into ONE text_len block (truncating
            # the tail — i.e. the oldest prompt — first, zero-padding the rest).
            # Uses the checkpoint's per-chunk prompt-window layout.
            lo = max(0, p - GCA_PROMPT_WINDOW)
            return [a_b_enc] + [prompt_cache[chunk_prompts[c]] for c in range(p, lo - 1, -1)]

        if self.rank == 0:
            print(
                f"[infer] per-chunk prompts: {len(prompt_cache)} unique / {num_chunks} chunks",
                flush=True,
            )

        # ---- First-frame RGB (the i2v anchor). Encoded ONLY into cond_y below
        # (mask=1 at frame 0); it does NOT occupy a separate seed latent — frame 0
        # is simply the first frame of chunk 0 and is generated like any other. ----
        x0_rgb = resize_crop(img[None].cpu()).transpose(0, 1).to(self.device)  # [1,3,H,W] -> [3,1,H,W]

        # ---- Plucker control for all latent positions ----
        # Seed-less: latent frame i (i = 0 .. N*chunk-1) aligns to RGB frame
        # i*stride (causal window-end; latent 0 == RGB frame 0).
        t_stride = int(self.vae_stride[0])
        total_lat_f = num_chunks * self.latent_window_size
        target_indices = torch.arange(total_lat_f, dtype=torch.long) * t_stride

        Ks_np = resize_crop.transform_intrinsics(
            json_intrinsics, img_h, img_w,
        )
        Ks = torch.from_numpy(Ks_np).float()

        c2ws_infer = interpolate_camera_poses(
            src_indices=np.arange(usable_frames, dtype=np.float64),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=target_indices.cpu().numpy().astype(np.float64),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer.to(self.device), framewise=True)
        Ks = broadcast_intrinsics_to_length(Ks, total_lat_f).to(self.device)

        only_rays_d = False
        action_lat = None
        if wasd_action is not None:
            action_lat = torch.from_numpy(wasd_action).float().to(self.device)
            action_lat = action_lat.index_select(0, target_indices.to(action_lat.device))
            only_rays_d = True

        plucker = get_plucker_embeddings(c2ws_infer, Ks, h, w, only_rays_d=only_rays_d)
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        plucker = plucker[None, ...]
        plucker = rearrange(
            plucker, "b (f h w) c -> b c f h w",
            f=total_lat_f, h=lat_h, w=lat_w,
        ).to(self.param_dtype)
        if action_lat is not None:
            act = action_lat[:, None, None, :].repeat(1, h, w, 1)
            act = rearrange(
                act,
                "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                c1=int(h // lat_h), c2=int(w // lat_w),
            )[None, ...]
            act = rearrange(act, "b (f h w) c -> b c f h w",
                            f=total_lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            plucker = torch.cat([plucker, act], dim=1)

        # ---- scheduler ----
        with torch.amp.autocast("cuda", dtype=self.param_dtype):
            # ---- AR chunk loop ----
            # Seed-less running latent stream: starts empty (or with the GT
            # prefix) and grows one generated chunk at a time. Decode is deferred
            # to a single VAE call on the whole [chunk_0 .. chunk_{N-1}] stream.
            # Frame 0 is chunk 0's first frame; its image content is injected only
            # through cond_y (mask=1 at frame 0), not as a separate seed latent.
            accumulated_future = None

            # ---- KV-cache AR generation (chunk-by-chunk streaming) ----
            # History lives in a persistent per-layer self-attn KV cache: each
            # chunk is denoised attending to the committed clean KVs of all prior
            # chunks, then its own clean x0 KV is committed at t=0. Cross-attn text
            # Text K,V are cached per chunk and reused across sampling steps.
            m0 = self.low_noise_model
            num_layers = len(m0.blocks)
            num_heads = m0.num_heads
            head_dim = m0.dim // num_heads
            # Under sequence parallelism the self-attn KV cache is head-sharded
            # (Ulysses): each rank stores only num_heads // sp_size local heads, so
            # the cache is non-redundant across ranks. sp_size==1 -> all heads.
            if num_heads % self.sp_size != 0:
                raise ValueError(f"num_heads ({num_heads}) must be divisible by sp_size ({self.sp_size})")
            local_num_heads = num_heads // self.sp_size
            chunk = self.latent_window_size
            frame_seqlen = ((lat_h + 1) // 2) * ((lat_w + 1) // 2)  # tokens / latent frame (patch 2x2)

            # Encode each optional reference as an independent single-frame latent.
            # Each image gets its own per-expert KV bank, selected by chunk_ref_ids.
            num_ref = 0
            latents_ref_36 = []
            if ref_images:
                for rimg in ref_images:
                    r = TF.to_tensor(rimg).sub_(0.5).div_(0.5)
                    r = resize_crop(r[None].cpu()).transpose(0, 1).to(self.device)  # [3,1,H,W]
                    ref16 = self.vae.encode([r])[0].detach()[None]  # [1,16,1,H,W]
                    zmask = ref16.new_zeros(1, 4, 1, lat_h, lat_w)
                    latents_ref_36.append(
                        torch.cat([ref16, zmask, ref16], dim=1).to(self.param_dtype)
                    )
                num_ref = len(latents_ref_36)
            ref_offset_frames = 0  # video RoPE starts at t=0; refs sit at negative positions

            # Windowed KV cache sized to `local_attn_size` latent frames when set
            # (the first `sink_size` frames of that window are the protected sink),
            # else keep the full clip cached (unbounded). `max_attention_size` bounds
            # how many cached tokens each query attends; the rolling+sink eviction
            # lives in GatedCausalAttention's KV-cache path.
            if self.cache_frames > 0:
                kv_size = self.cache_frames * frame_seqlen
            else:
                kv_size = num_chunks * chunk * frame_seqlen
            max_attention_size = kv_size
            if sparse_mem_topk > 0:
                # Sparse full-fidelity GCA memory: --sink_chunks/--recent_chunks
                # are reinterpreted as the sink / recent windows IN FRAMES and must be
                # chunk-aligned; middle history stays cached (CPU-offloaded) and is
                # retrieved per layer via pooled-QK top-k. No rolling eviction.
                if self.recent_chunks <= 0:
                    raise ValueError(
                        f"sparse memory requires --recent_chunks > 0, got {self.recent_chunks}"
                    )
                sparse_sink_c = self.sink_chunks
                sparse_recent_c = self.recent_chunks

            def _new_self_kv():
                if sparse_mem_topk > 0:
                    # GCAKVCache is a plain object (pytree LEAF): FSDP's root
                    # pre-forward rebuilds dict/list kwargs, which silently drops
                    # dict-level commits — a class instance passes by reference.
                    # pk = fp32 pre-RoPE pooled-key table, device-resident (a few
                    # MB): each layer's per-forward top-k retrieval scores cosine
                    # against this table only, never the offloaded full-res blocks.
                    return [
                        GCAKVCache(
                            topk=int(sparse_mem_topk),
                            sink_chunks=int(sparse_sink_c),
                            recent_chunks=int(sparse_recent_c),
                            offload=bool(sparse_mem_offload),
                            pk=torch.zeros(1, num_chunks, local_num_heads, head_dim,
                                           dtype=torch.float32, device=self.device),
                        )
                        for _ in range(num_layers)
                    ]
                return [
                    {
                        "k": torch.zeros(1, kv_size, local_num_heads, head_dim, dtype=self.param_dtype, device=self.device),
                        "v": torch.zeros(1, kv_size, local_num_heads, head_dim, dtype=self.param_dtype, device=self.device),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                    }
                    for _ in range(num_layers)
                ]

            def _new_ref_kv():
                # One single-frame bank per reference image.
                R = frame_seqlen
                return [
                    {
                        "k": torch.zeros(1, R, local_num_heads, head_dim, dtype=self.param_dtype, device=self.device),
                        "v": torch.zeros(1, R, local_num_heads, head_dim, dtype=self.param_dtype, device=self.device),
                    }
                    for _ in range(num_layers)
                ]

            # Each expert owns its history and reference caches. Multi-step CFG
            # additionally keeps an independent unconditional stream; distilled
            # sampling needs only the conditional stream.
            def _new_expert(model):
                expert = {
                    "model": model,
                    "kv_cond": _new_self_kv(),
                    "ref_cond": [_new_ref_kv() for _ in range(num_ref)],
                    "cc": [{"is_init": False} for _ in range(num_layers)],
                }
                if not distilled:
                    # CFG branches must keep independent committed history and refs.
                    expert.update(
                        kv_uncond=_new_self_kv(),
                        ref_uncond=[_new_ref_kv() for _ in range(num_ref)],
                        cu=[{"is_init": False} for _ in range(num_layers)],
                    )
                return expert
            experts = [_new_expert(self.low_noise_model), _new_expert(self.high_noise_model)]
            boundary_ts = self.boundary * self.num_train_timesteps

            def _pick_expert(t):
                return experts[1] if float(t.item()) >= boundary_ts else experts[0]

            # i2v cond_y stream = [mask(4) | VAE([first_frame, black×rest])(16)],
            # mask=1 ONLY at frame 0 (the
            # input image); every other frame is mask=0 + VAE(black). Sliced per
            # chunk and used for BOTH denoise and the t=0 KV-commit, so committed
            # frames read as the base's cached non-anchor frames (not mask=1).
            total_gen_lat = num_chunks * chunk
            stride = int(self.vae_stride[0])
            cond_rgb_frames = (total_gen_lat - 1) * stride + 1
            ph, pw = x0_rgb.shape[-2], x0_rgb.shape[-1]
            cond_rgb = torch.cat(
                [x0_rgb, x0_rgb.new_zeros(3, cond_rgb_frames - 1, ph, pw)], dim=1
            )
            cond_latents = self.vae.encode([cond_rgb])[0]
            cond_mask = build_i2v_mask(
                frame_num=cond_rgb_frames,
                lat_h=lat_h,
                lat_w=lat_w,
                device=cond_latents.device,
                dtype=cond_latents.dtype,
                vae_temporal_stride=stride,
            )
            cond_stream = torch.cat([cond_mask, cond_latents], dim=0)  # [20, total_gen_lat, lat_h, lat_w]

            def _plucker_slice(start, end):
                return {"c2ws_plucker_emb": plucker[:, :, start:end].chunk(1, dim=0)}

            def _commit(lat, start_frame, ctx_list, ref_id=None):
                # Write clean KVs for `lat` (frames [start_frame:start_frame+F]) at t=0.
                # Commit each expert's clean history under the active text stream.
                # Multi-step CFG also commits a separate unconditional stream.
                f = lat.shape[2]
                cur = start_frame * frame_seqlen
                for e in experts:
                    streams = [(ctx_list, e["kv_cond"], e["ref_cond"])]
                    if not distilled:
                        streams.append((context_null, e["kv_uncond"], e["ref_uncond"]))
                    for ctx, kvc, ref_banks in streams:
                        e["model"](
                            x=[lat.squeeze(0)],
                            t=torch.tensor([0.0], device=self.device),
                            context=ctx,
                            seq_len=f * frame_seqlen,
                            y=[cond_stream[:, start_frame:start_frame + f].to(self.device)],
                            dit_cond_dict=_plucker_slice(start_frame, start_frame + f),
                            kv_cache=kvc,
                            current_start=cur,
                            max_attention_size=max_attention_size,
                            frame_seqlen=frame_seqlen,
                            ref_kv=ref_banks[ref_id] if ref_id is not None else None,
                            attend_ref=ref_id is not None,
                            rope_offset_frames=ref_offset_frames,
                            # sparse memory: this clean pass appends the chunk's
                            # full-res KV block + pooled key (no-op for the
                            # rolling cache).
                            commit_chunk=True,
                        )

            # Build each reference bank only when its first annotated chunk is
            # reached. Each bank is independent for every expert and active branch.
            ready_refs = set()
            zero_ctrl = (
                torch.zeros(1, plucker.shape[1], 1, lat_h, lat_w,
                            device=self.device, dtype=self.param_dtype)
                if num_ref else None
            )

            def _prepare_reference(ref_id, chunk_idx):
                if ref_id in ready_refs:
                    return
                ref_ctx = [a_b_enc, prompt_cache[chunk_prompts[chunk_idx]]]
                ref_frame = latents_ref_36[ref_id]
                for e in experts:
                    streams = [(ref_ctx, e["ref_cond"][ref_id])]
                    if not distilled:
                        streams.append((context_null, e["ref_uncond"][ref_id]))
                    for rctx, refb in streams:
                        e["model"](
                            x=[ref_frame.squeeze(0)],
                            t=torch.tensor([0.0], device=self.device),
                            context=rctx,
                            seq_len=frame_seqlen,
                            y=None,
                            dit_cond_dict={"c2ws_plucker_emb": zero_ctrl.chunk(1, dim=0)},
                            current_start=0,
                            max_attention_size=max_attention_size,
                            frame_seqlen=frame_seqlen,
                            ref_kv=refb,
                            commit_ref=True,
                            ref_temporal_pos=-256,
                        )
                ready_refs.add(ref_id)

            for chunk_idx in range(num_chunks):
                chunk_context = cumulative_context(chunk_idx)
                start_frame = chunk_idx * chunk
                current_start = start_frame * frame_seqlen
                ref_id = selected_ref_ids[chunk_idx]
                attend_ref = ref_id is not None
                if attend_ref:
                    _prepare_reference(ref_id, chunk_idx)
                chunk_control = _plucker_slice(start_frame, start_frame + chunk)
                # Cumulative context changed vs the previous chunk -> recompute text
                # K/V on this chunk's first denoising step, then reuse for the rest.
                for e in experts:
                    for lc in e["cc"]:
                        lc["is_init"] = False
                    if not distilled:
                        for lc in e["cu"]:
                            lc["is_init"] = False

                target_latent = torch.randn(
                    1, 16, chunk, lat_h, lat_w,
                    dtype=torch.float32, generator=seed_g, device=self.device,
                )
                if distilled:
                    pcm_sigmas = torch.tensor(pcm_phase_sigmas, device=self.device, dtype=torch.float32)
                    timesteps = pcm_sigmas[:-1] * self.num_train_timesteps
                else:
                    sched = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False,
                    )
                    sched.set_timesteps(sampling_steps, device=self.device, shift=shift)
                    timesteps = sched.timesteps

                pbar = tqdm(timesteps, desc=f"[chunk {chunk_idx + 1}/{num_chunks}]", disable=(self.rank != 0))
                latent = target_latent.squeeze(0)
                # Built once per chunk (constant across denoising steps). kv_cache and
                # cross caches are per-expert (picked per timestep), so passed per call.
                common = dict(
                    seq_len=chunk * frame_seqlen,
                    y=[cond_stream[:, start_frame:start_frame + chunk].to(self.device)],
                    dit_cond_dict=chunk_control,
                    current_start=current_start,
                    max_attention_size=max_attention_size,
                    frame_seqlen=frame_seqlen,
                    attend_ref=attend_ref,
                    rope_offset_frames=ref_offset_frames,
                )
                for step_idx, t in enumerate(pbar):
                    timestep = torch.stack([t]).to(self.device)
                    x_in = [latent.to(self.device)]
                    e = _pick_expert(t)
                    noise_pred_cond = e["model"](
                        x=x_in, t=timestep, context=chunk_context,
                        kv_cache=e["kv_cond"], crossattn_cache=e["cc"],
                        ref_kv=e["ref_cond"][ref_id] if attend_ref else None, **common)[0]
                    if distilled:
                        sigma_next = pcm_sigmas[step_idx + 1]
                        step_noise = (
                            torch.randn(latent.shape, dtype=latent.dtype,
                                        device=latent.device, generator=seed_g)
                            if float(sigma_next) > 0 else torch.zeros_like(latent)
                        )
                        latent = pcm_step(
                            latent, noise_pred_cond, pcm_sigmas[step_idx], sigma_next, step_noise,
                        )
                    else:
                        noise_pred_uncond = e["model"](
                            x=x_in, t=timestep, context=context_null,
                            kv_cache=e["kv_uncond"], crossattn_cache=e["cu"],
                            ref_kv=e["ref_uncond"][ref_id] if attend_ref else None, **common)[0]
                        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                        temp_x0 = sched.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0]
                        latent = temp_x0.squeeze(0)

                chunk_latent = latent.unsqueeze(0)
                # Commit the clean chunk KVs so subsequent chunks condition on it.
                _commit(chunk_latent, start_frame, chunk_context, ref_id=ref_id)
                accumulated_future = (
                    chunk_latent
                    if accumulated_future is None
                    else torch.cat([accumulated_future, chunk_latent], dim=2)
                )

                if offload_model:
                    torch.cuda.empty_cache()

            if offload_model:
                torch.cuda.empty_cache()

            # Single full-stream VAE decode after all chunks generated. The
            # stream is the whole clip [chunk_0 .. chunk_{N-1}] = num_chunks *
            # latent_window_size latents; frame 0 (chunk 0's first frame) is the
            # input image, reconstructed from the cond_y anchor.
            if self.rank == 0:
                history_video = self.vae.decode([accumulated_future.squeeze(0)])[0]
                videos = [history_video]

        del target_latent, accumulated_future
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.rank == 0:
            videos = videos[0]
            if vis_ui:
                from .utils.vis_utils import visualize_wasd_and_rotation_ui

                videos_np = videos.detach().cpu().numpy()
                videos_np = np.transpose(videos_np, (1, 2, 3, 0))
                videos_np = (videos_np + 1) / 2
                videos_np = visualize_wasd_and_rotation_ui(
                    videos_np, c2ws, wasd_action,
                )
                videos_np = np.transpose(videos_np, (3, 0, 1, 2))
                videos_np = videos_np * 2 - 1
                videos = torch.from_numpy(videos_np)
            return videos
        return None
