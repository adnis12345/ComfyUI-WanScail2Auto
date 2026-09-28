import torch
from comfy_api.latest import io
import comfy.model_management as mm
import nodes

# 根据你的环境修正导入路径
# from comfy_extras.nodes_scail import WanSCAILToVideo
# from comfy_extras.nodes_custom_sampler import SamplerCustom


def _vae_decode(latent: dict, vae):
    samples = latent["samples"]
    device = mm.get_torch_device()
    vae.to(device)
    img = vae.decode(samples)
    return img


def _split_latent(latent: dict, segment_length: int):
    samples = latent["samples"]
    total_frames = samples.shape[1]
    seg_list = []
    start = 0
    while start < total_frames:
        end = min(start + segment_length, total_frames)
        seg_samples = samples[:, start:end, ...]
        new_latent = latent.copy()
        new_latent["samples"] = seg_samples
        if "noise_mask" in new_latent:
            new_latent["noise_mask"] = new_latent["noise_mask"][:, start:end, ...]
        seg_list.append(new_latent)
        start = end
    return seg_list


def _concat_latent(latent_segments):
    all_samples = []
    mask_list = []
    has_mask = False
    for seg in latent_segments:
        all_samples.append(seg["samples"])
        if "noise_mask" in seg:
            has_mask = True
            mask_list.append(seg["noise_mask"])
    full_samples = torch.cat(all_samples, dim=1)
    full_latent = latent_segments[0].copy()
    full_latent["samples"] = full_samples
    if has_mask:
        full_latent["noise_mask"] = torch.cat(mask_list, dim=1)
    return full_latent

def align_frame_len(target_frames):
    remainder = (target_frames - 1) % 4
    valid = target_frames - remainder
    return max(1, valid)


class WanAutoSegmentSingleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanAutoSegmentSingleNode",
            category="VideoAuto::Experimental",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=512, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=896, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("pose_video", optional=True, tooltip="Video used for pose conditioning. Will be downscaled to half the resolution of the main video."),
                io.Image.Input("pose_video_mask", optional=True, tooltip="SCAIL-2 only. Colored per-identity SAM3 mask video at the same resolution as pose_video."),
                io.Boolean.Input("replacement_mode", default=False, optional=True, tooltip="SCAIL-2 only. False = Animation Mode (pose_video_mask should have black background). True = Replacement Mode (pose_video_mask should have white background)."),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Strength of the pose latent."),
                io.Float.Input("pose_start", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Start step of the pose conditioning."),
                io.Float.Input("pose_end", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="End step of the pose conditioning."),
                io.Image.Input("reference_image", optional=True, tooltip="Reference image. The first image is the primary reference (composite all identities onto it). SCAIL-2: extra batch images are used as additional views (back view, close-up, occluded background), each needing a matching reference_image_mask in that identity's color."),
                io.Image.Input("reference_image_mask", optional=True, tooltip="SCAIL-2 only. Colored reference mask, batch matching reference_image (first = primary reference mask, rest = identity masks for the additional reference_image)."),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True, tooltip="CLIP vision features for conditioning. Model is trained with stretch resize to aspect ratio."),
                io.Int.Input("video_frame_offset", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Cumulative output frame this chunk begins at. Wire from the previous chunk's video_frame_offset output."),
                io.Int.Input("previous_frame_count", default=5, min=1, max=nodes.MAX_RESOLUTION, step=4, tooltip="Tail frames of previous_frames to anchor. SCAIL-2 trained at 5 (81-frame chunks, 76-frame step)."),
                io.Image.Input("previous_frames", optional=True, tooltip="SCAIL-2 only. Full decoded output of the previous chunk. Only the last previous_frame_count are used as the extension anchor."),

                # SamplerCustom 参数
                io.Model.Input("model"),
                io.Boolean.Input("add_noise", default=True, advanced=True),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                # io.Conditioning.Input("positive"),
                # io.Conditioning.Input("negative"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                # io.Latent.Input("latent_image"),
            ],
            outputs=[
                io.Image.Output("full_rgb_video"),
            ]
        )

    @classmethod
    def execute(cls, model, add_noise, noise_seed, cfg, sampler, sigmas, positive, negative, vae, width, height, length, batch_size, pose_strength, pose_start, pose_end,
                video_frame_offset, previous_frame_count, replacement_mode=False, reference_image=None, clip_vision_output=None, pose_video=None,
                pose_video_mask=None, reference_image_mask=None, previous_frames=None) -> io.NodeOutput:

        # ========== 获取pose_video总帧数 ==========
        total_frames = 0
        seg_len = length
        overlap = 5
        step = seg_len - overlap
        rgb_result_list = []

        if pose_video is not None:
            # pose_video shape: [帧数B, H, W, 3]
            total_frames = pose_video.shape[0]
            print(f"pose_video 总帧数 = {total_frames}, tensor shape={pose_video.shape}")
        else:
            print("pose_video 为空输入")

        # ========= 迭代状态变量：每轮会更新 =========
        current_offset = video_frame_offset       # 第一轮=0，后续用上一轮offset_out
        current_prev_frames = previous_frames     # 第一轮=None，后续存重叠latent

        from nodes import NODE_CLASS_MAPPINGS
        WanCls = NODE_CLASS_MAPPINGS["WanSCAILToVideo"]
        SamplerCls = NODE_CLASS_MAPPINGS["SamplerCustom"]

        print("pose_video shape:", pose_video.shape)
        print("pose_video_mask shape:", pose_video_mask.shape)

        start = 0
        while start < total_frames:
            end = min(start + seg_len, total_frames)

            seg_actual_len = end - start
            seg_actual_len = align_frame_len(seg_actual_len)
            print(f"当前分段视频，计划处理的帧数范围： 开始{start} 结束{end}，本段实际帧数 {seg_actual_len}")
            print(f"video_frame_offset:{current_offset}")

            # seg_pose = pose_video[start:end]
            # seg_mask = pose_video_mask[start:end] if pose_video_mask is not None else None

            # print("seg_pose shape:", seg_pose.shape)
            # if seg_mask is not None:
            #     print("seg_mask shape:", seg_mask.shape)


            # ========== 关键字传参，避免位置错乱 ==========
            wan_ret = WanCls.execute(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=seg_actual_len,
                batch_size=batch_size,
                pose_strength=pose_strength,
                pose_start=pose_start,
                pose_end=pose_end,          
                video_frame_offset=current_offset,
                previous_frame_count=overlap,
                replacement_mode=replacement_mode,
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                pose_video=pose_video,
                pose_video_mask=pose_video_mask,
                reference_image_mask=reference_image_mask,
                previous_frames=current_prev_frames
            )
            pos_out = wan_ret[0]    # positive
            neg_out = wan_ret[1]    # negative
            latent_out = wan_ret[2]  # out_latent
            offset_out = wan_ret[3]  # video_frame_offset + length

            # 采样器执行
            samp_ret = SamplerCls.execute(
                model=model,
                add_noise=add_noise,
                noise_seed=noise_seed,
                cfg=cfg,
                positive=pos_out,
                negative=neg_out,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent_out
            )
            samp_latent = samp_ret[1] 

            latent_tensor = samp_latent["samples"]
            print(f"片段latent shape: {latent_tensor.shape}")
            # latent_result_list.append(latent_tensor)

            latent = samp_latent["samples"]
            seg_rgb_frames = vae.decode(latent)
            if len(seg_rgb_frames.shape) == 5:
                seg_rgb_frames = seg_rgb_frames.reshape(-1, seg_rgb_frames.shape[-3], seg_rgb_frames.shape[-2], seg_rgb_frames.shape[-1])

            print(f"转换后seg_rgb_frames shape: {seg_rgb_frames.shape}")
            # 收集RGB片段
            if len(rgb_result_list) == 0:
                rgb_result_list.append(seg_rgb_frames)
            else:
                seg_rgb_cut = seg_rgb_frames[overlap:, :, :, :]
                rgb_result_list.append(seg_rgb_cut)

            current_prev_frames = seg_rgb_frames
            print(f"current_prev_frames shape: {current_prev_frames.shape}")
            current_offset = offset_out
            if start + seg_len >= total_frames:
                break
            start = start + step

        full_rgb = torch.cat(rgb_result_list, dim=0)
        print(f"拼接完成完整latent shape: {full_rgb.shape}")

        return io.NodeOutput(full_rgb)


NODE_CLASS_MAPPINGS = {
    "WanAutoSegmentSingleNode": WanAutoSegmentSingleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAutoSegmentSingleNode": "WanScail2自动分段接力"
}
