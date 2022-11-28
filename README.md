# TTS project

In this project I've implemented FastSpeech2 model to generate voice by text prompts.

WandB report:
https://wandb.ai/mak_corp/tts_project/reports/TTS-project-report--VmlldzozMDUwNzM5

Installation guide:
```shell

python3 -m pip install -r requirements.txt
chmod +x setup.sh
./setup.sh

```

Run train:
```shell

python3 train.py -c hw_tts/configs/fastspeech_2.json

```

Run test:
```shell

python3 test.py \
-c hw_tts/configs/fastspeech_2.json \
-r checkpoints/TTS/best_model/model_best.pth \
-o test_wavs

```
