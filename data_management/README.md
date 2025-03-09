데이터를 어떻게 활용해야하나
파일을 일일이 하나씩 읽으면 너무 느리다

csv, json 등의 메타데이터 형태로 파일 path와 전처리한 내용을 저장해두는게 빠르다
```
face_name,face_name_align,race,gender,age,race_scores_fair,gender_scores_fair,age_scores_fair
00000.png,/purestorage/datasets/DGM/FFHQ_origin/detected_faces/00000_face_detected.png,White,Male,0-2,"[0.42043263 0.01416711 0.1518024  0.20708567 0.11146507 0.02823909
 0.06680806]",[0.6150983  0.38490167],"[9.9551231e-01 4.3601408e-03 1.3388054e-05 3.5781875e-05 4.6363242e-05
 2.3634186e-05 6.8688073e-06 1.2569327e-06 2.0532821e-07]"

```

이후 해당 메타데이터를 읽자

```python
import glob

def get_dataframe():
    ffhq = glob.glob("/purestorage/datasets/DGM/FFHQ_origin/detected_faces" + "/*.csv")
    utk = glob.glob("/purestorage/datasets/face_recognition/UTKFace" + "/*.csv") # face_detected_face_detected 있는건 제외하기
    rfw = glob.glob("/purestorage/datasets/face_recognition/rfw/test/data/detected_faces" + "/*.csv")
    ylfw = glob.glob("/purestorage/datasets/face_recognition/YLFW_Benchmark/detected_faces" + "/*.csv")
    all_files = ffhq + utk + rfw + ylfw

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    df = frame[~frame['face_name_align'].str.contains('_face_detected.*_face_detected')]
    return df

# 이후 파일을 압축한 뒤에 로컬 서버의 하드로 보내기
all_combinations = all_combinations[args_idx.start_idx:args_idx.end_idx]
idx = args_idx.start_idx
for race, gender, age in all_combinations:
    filtered_df = df[(df['race'] == race) & (df['gender'] == gender) & (df['age'] == age)]
    image_files = filtered_df['face_name_align'].tolist()
    print(len(image_files))
    # 압축 파일 경로
    archive_path = f"/workspace/images_archive{idx}.zip"

    # 이미지 파일을 압축 파일로 만들기
    with zipfile.ZipFile(archive_path, 'w') as archive:
        # 퓨어스토리지 이미지 압축하기
        for file_path in image_files:
            archive.write(file_path, os.path.basename(file_path))

    print(f"Created archive: {archive_path}")

    # /tmp에서 압축 파일 풀기 (이미 /tmp에 있으니 이동 단계는 생략)
    extract_to_path = f"/workspace/unpacked_images{idx}"
    os.makedirs(extract_to_path, exist_ok=True)  # 추출 디렉토리 생성
    with zipfile.ZipFile(archive_path, 'r') as archive:
        archive.extractall(extract_to_path)

    print(f"Extracted files to: {extract_to_path}")
    idx += 1


```
