# Speech Recognition

## Dependency

pip install -r requirements.txt

## Dataset
  1. VOiCES
  2. Mozilla Common Voice [huggingface](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)
  3. Librispeech [huggingface](https://huggingface.co/datasets/openslr/librispeech_asr)
  4. LIUM [huggingface](https://huggingface.co/datasets/LIUM/tedlium)

```bash
# install VOiCES dataset
sudo apt install awscli
aws s3 cp --no-sign-request s3://lab41openaudiocorpus/VOiCES_devkit.tar.gz .
```

## Other
  - chagne huggice-face dataset .cache folder to avoid OS_ERROR
    ```
    EX: put this line in .bashrc 
    export HF_DATASETS_CACHE="/dev/shm/wylin2"
    ```
