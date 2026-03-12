# Text-To-Video

The `DreamID-Omni` pipeline generates short videos from text, image and video.

## Local CLI Usage
### Download the Model locally
Since DreamID-Omni combine multiple models, and without any config, so we need to download them locally.

```bash
python download_dreamid_omni.py --output-dir ./dreamid_omni
```

### Run the Inference
python x_to_video_audio.py \
  --model ./dreamid_omni \
  --model-type "dreamid-omni" \
  --prompt "Two people walking together and singing happily" \
  --image-path ./example0.png ./example1.png \
  --audio-path ./example0.wav ./example1.wav \
  --video-negative-prompt "jitter, bad hands, blur, distortion" \
  --audio-negative-prompt "robotic, muffled, echo, distorted" \
  --cfg-parallel-size 2 \
  --num-inference-steps 45 \
  --height 704 \
  --width 1024 \
  --output dreamid_omni.mp4
```

Key arguments:

- `--prompt`: text description (string).
- `--model`: path to the model local directory.
- `--model-type`: model type, now only support "dreamid-omni", we don't have config in huggingface repo, so we need to pass the model type.
- `--height/--width`: output resolution (defaults 704 * 1024).
- `--image-path`: path to the input image list.
- `--audio-path`: path to the input audio list, indicate the timbre of the output video.
- `--cfg-parallel-size`: number of parallel cfg parallel (defaults 1).
- `--num-inference-steps`: number of denoising steps (defaults 45).
- `--video-negative-prompt`: negative prompt for video generation.
- `--audio-negative-prompt`: negative prompt for audio generation.
