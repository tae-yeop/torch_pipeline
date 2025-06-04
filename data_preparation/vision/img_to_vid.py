import glob
import os
import cv2
import numpy as np
from moviepy import *


def images_to_mp4_moviepy(input_pattern, output_mp4, size=(256,256), fps=20):
    """
    input_pattern: '/path/to/real69_flow_*.jpg'
    output_mp4: 'flow_moviepy.mp4'
    size=(W,H)
    fps=20
    """
    # 1) 이미지 경로 수집, 정렬
    img_paths = glob.glob(input_pattern)
    img_paths.sort()
    if len(img_paths) == 0:
        print("No images found")
        return

    # 2) 프레임(이미지) 목록 생성
    frames = []
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        resized = cv2.resize(img, size)
        # moviepy에서는 [H,W,3] RGB float or uint8 형태가 필요
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(rgb) # 저장
    
    if len(frames) == 0:
        print("No valid frames read.")
        return
    
    # 3) ImageSequenceClip 생성
    clip = ImageSequenceClip(frames, fps=fps)  # frames: list of [H,W,3] ndarrays in RGB

    # 4) mp4로 쓰기
    # codec='libx264', audio=False => 무음 영상
    clip.write_videofile(output_mp4, codec='libx264', audio=False)
    print("Saved video:", output_mp4)


# --- Usage ---
if __name__ == "__main__":
    pattern = "/purestorage/project/tyk/3_CUProjects/iBeta/mask*_reflection_mask.jpg"
    out_mp4 = "/purestorage/project/tyk/3_CUProjects/iBeta/mask_reflection_mask.mp4"
    images_to_mp4_moviepy(pattern, out_mp4, size=(256,256), fps=20)

