


## Info

In various applications, such as information retrieval, recommendation systems, and semantic search, we often need to find similar items or documents based on their content. Embeddings provide an efficient way to represent these items as dense vectors, where items with similar meanings are located close to each other in the vector space. 

This repository showcases how to create and use embeddings to build a search system for analizing images and building an 
analitics on top of it.


## Install 

```
$ source /path/to/venv/bin/activate
(env) $ python -m pip install pip-tools
(env) $ pip-compile
```
```
docker pull vespaengine/vespa:8.230.17
```


## Download

```
TEXT_MODEL_NAME="sentence-transformers/clip-ViT-B-32-multilingual-v1" CLIP_MODEL_NAME="clip-ViT-B-32" \
  ./cicd/build.sh
```