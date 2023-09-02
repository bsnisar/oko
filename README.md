## Install 

```
$ source /path/to/venv/bin/activate
(env) $ python -m pip install pip-tools
(env) $ pip-compile
```


## Download

```
TEXT_MODEL_NAME="sentence-transformers/clip-ViT-B-32-multilingual-v1" \
  CLIP_MODEL_NAME="clip-ViT-B-32" \
  ./cicd/build.sh
```