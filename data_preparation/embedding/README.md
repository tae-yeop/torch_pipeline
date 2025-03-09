### FAISS

- 2017년 페이스북에서 내놓은 Facebook AI Similarity Search (Faiss) 라이브러리
- Billon 스케일 데이터셋까지도 소화한다 (CUDA GPU를 사용)
- 정확도와 속도사이의 trade-off를 고려해 알고리즘을 선택할 수 있다
- 주요 기술
    - 인버스 파일 시스템 (Inverted File System, IVF)
    - Product Quantization (PQ)
    - Flat (exact search)
    - HNSW (Hierarchical Navigable Small World graphs)

```
pip install faiss-gpu
```

- 사용방법
    - `Index`라는 데이터 베이스에 벡터를 등록하도록 한다
    - `Index`는 여러 종류 있음 ⇒ 상황에 맞춰 선택 : `IndexFlatL2`, `IndexIVFFlat` , `IndexIVFPQ`
    - 임베딩을 인덱스에 넣으면 학습(벡터들의 분포를 분석)을 수행함
    - 검색시 쿼리 벡터를 입력받아서 kNN을 이용해서 찾아낸다
    - Index와 Distance를 리턴해줌

- 인덱스 종류
    - `IndexFlatL2`
        - brute force exhaustive search ⇒ 등록된 모든 벡터와 비교를 하게됨
    - `IndexIVFFlat`
        - 클러스터링하는데 k-means 등을 사용해서 유사한 벡터들이 같은 클러스터에 할당
        - Inverted File을 생성하는데 여기에는 역색인 정보를 담고 있다 : 클러스터 정보와 클러스터에 속한 벡터들의 인덱스
        - 쿼리와 정말 유사한게 하필이면 다른 클러스터에 배정받는 바람에 제일 유사한 벡터를 못찾을수도 있다  ⇒ 정확도를 희생하는 대신에 Search space를 줄이는 trade-off
    - `IndexIVFPQ`
        - 벡터를 저장할 때 원본 그자체(flat) 형태로 저장 ⇒ 데이터셋이 엄청 커지면 부담
        - PQ에선 원본 벡터를 서브벡터로 나눈다
        - 각 서브벡터마다 클러스터링을 수행한다 ⇒ 각 서브벡터마다 centroids가 만들어진다
        - 여기서 또 각각 서브벡터마다 개별적인 ID를 부여한다. 
### ScaNN

- 구글에서도 비슷하게 ScaNN(Scalable Nearest Neighbors)을 내놓음
- 벤치마크 상으로는 가장 속도가 좋은 것으로 보인다
- 주요 기술
    - 트리 기반 클러스터링
    - Anisotropic Hashing (AH)
    - Product Quantization (PQ)

```
pip install scann
```

- 동작 과정
(1) 트리 기반 클러스터링

- 트리 구조를 사용하여 여러 클러스터로 분할해두었고 검색시 클러스터 centroid를 이용해서 클러스터 찾음

(2) Anisotropic Hashing (AH) 또는 Product Quantization (PQ)

- 벡터 차원을 줄여서 빠른 검색을 수행

(3) Reordering (재정렬)

- 2번까지 한 뒤에 추가적인 검색을 수행한다
- 이때는 실제 데이터와 유사도를 다시 계산(정확도 ⬆️)