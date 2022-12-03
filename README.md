# TTS project

In this project I've implemented FastSpeech2 model to generate voice by text prompts.

WandB report:
https://wandb.ai/mak_corp/tts_project/reports/TTS-project-report--VmlldzozMDUwNzM5

WARNING! I've fixed bug on inference after deadline. Here is the code state before deadline:
https://github.com/mak-corp/tts_project/tree/3a9bff448661402f2f436633c770f1c589ecd0b2


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
-c hw_tts/configs/fastspeech_baseline.json \
-r checkpoints/TTS/best_model/model_best.pth \
-o test_wavs

```

Run test of FastSpeech2 fixed after deadline:
```shell

python3 test.py \
-c hw_tts/configs/fastspeech_2.json \
-r checkpoints/TTS/best_model_2/model_best.pth \
-o test_wavs

```
