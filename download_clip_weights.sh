#!/bin/bash

vitPath="modules/ViT-B-32.pt"

if [[ -f "$vitPath" ]]; then
  echo "ViT-B-32 weights have already been downloaded and are in the file $vitPath"
  exit 0
fi

wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
