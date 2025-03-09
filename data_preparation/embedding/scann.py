import torch
import scann
import numpy as np
import argparse

def get_data(args):
    if args.data_path == "random":
        data = np.random.rand(500, 512).astype('float32')

    elif '.pt' in args.data_path and os.path.exists(args.data_path):
        data = torch.load(args.data_path)
        data = data.detach().cpu().numpy().astype(np.float32) # float32로 변환 권장됨

    elif '.npy' in args.data_path and os.path.exists(args.data_path):
        data = np.load(args.data_path)

    else:
        raise ValueError(f"Invalid data path: {args.data_path}")

    return data

def get_query_data(args):
    if args.query_data_path == "random":
        dim = 512
        q_num = 10
        query_data = np.random.rand(q_num, dim).astype(np.float32)
    else:
        query_data = np.load(args.query_data_path)

    return query_data

def get_searcher():


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/data.npy") #random, 파일 경로
    args = parser.parse_args()

    data = get_data(args)
    queries = get_query_data(args)

    searcher = scann.scann_ops_pybind.builder(
        data,
        num_neighbor=10, # 결과로 얻고자 하는 인덱스 갯수
        similarity_measure="dot_product" # "dot_product" 또는 "squared_l2” 사용
    ).tree( # Hierarchical Clustering을 설정
        num_leaves=100, # 생성할 클러스터(또는 리프 노드)의 수
        num_leaves_to_search=10, # 검색 시 고려할 클러스터의 수 (적을수록 빠르지만 부정확)
        training_sample_size=25000 # 클러스터링 알고리즘을 훈련용 샘플의 수(정확도 ⬆️)
    ).score_ah( # Anisotropic Hashing(AH)는 벡터를 해싱하여 유사도 검색을 가속화함
        2, # 해싱에 사용할 차원수 (정확도 ⬆️)
        anisotropic_quantization_threshold=0.2 # 이 값이상 유사도를 가진 벡터들을 해싱 (유사도가 낮으면 해싱 제외)
    ).reorder( # AH 등의 해싱 기법으로 빠르게 필터링한 후에 실제 데이터와의 유사도를 다시 계산
        reordering_num_neighbors=100 # 유사도 재계산을 위해 고려할 이웃의 수 (정확도 ⬆️)
    ).build() 

    neighbors, distances = searcher.search_batched(queries)

    print("Neighbors:", neighbors)
    print("Distances:", distances)


    #  Neighbors: [[4784 7876 9561  323 1631 5759 5560 8426 1524 1706]
    # [ 737 2593  323 7876 1765 1524 4784 4397 3340  357]
    # [ 323 7876 3436  524 9291 5493 1524 2593 7479 1825]
    # [ 323 1631 2593 7876 7896 1524  524  568 4758 8115]
    # [1524  323 3383 1631 8970 3325 5475 9490 5560   39]
    # [8970 5475  323 3691 9555 7479 4859 1983 1995  737]
    # [2593  323  524 1631  568 1995 4784 8187 3701 7876]
    # [2593  323 1631 7721 7876  524 1576 7835 5759 8906]
    # [ 323  524 5497 2593 8906 4784 7217 4931 7507 1391]
    # [2593  323 6045  524 7996 7876 2072 2590 1899 3701]]
    # Distances: [[141.94846 141.45673 140.85341 140.43272 140.21185 139.79039 139.75795
    # 139.69138 139.27908 138.8934 ]
    # [135.73688 135.55466 134.78435 134.76509 134.5323  134.44995 134.31204
    # 133.66522 133.56009 133.37633]
    # [145.17993 143.92627 143.26053 142.71298 142.21617 141.936   141.8855
    # 141.6841  141.55838 141.44118]
    # [143.74937 141.53625 141.15004 141.0061  140.50882 139.95264 139.83185
    # 139.06454 138.8533  138.8128 ]
    # [139.70424 139.6432  139.53827 139.47894 139.29916 138.71463 138.47798
    # 138.4502  138.28758 138.20108]
    # [143.18991 141.5333  140.817   140.68095 140.09978 139.4015  139.14334
    # 138.8306  138.76282 138.34735]
    # [141.43015 140.18555 139.99133 139.81447 139.76259 139.62413 139.48204
    # 139.05289 138.97516 138.5133 ]
    # [139.92155 139.87491 137.08524 136.88643 136.64754 136.52008 135.84695
    # 135.81412 135.54132 135.51071]
    # [143.30328 140.83505 140.14433 139.9558  139.30743 139.23976 138.2784
    # 138.17384 138.14197 137.75952]
    # [138.36737 137.9672  137.36467 136.9118  136.88997 136.84763 136.77675
    # 136.69437 136.61513 136.59593]]