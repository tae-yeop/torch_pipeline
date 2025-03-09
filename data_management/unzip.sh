#!/bin/bash

# 현재 디렉토리의 모든 images_archive*.zip 파일에 대해 반복
for zipfile in images_archive*.zip; do
  # 압축 파일 이름에서 숫자 추출
  number=$(echo "$zipfile" | grep -o '[0-9]\+')
  # 대상 폴더 이름 설정 (예: unpacked_images34)
  target_folder="unpacked_images$number"
  # 대상 폴더가 없으면 생성
  mkdir -p "$target_folder"
  # 압축 해제 명령 실행
  unzip -o "$zipfile" -d "$target_folder"
done
