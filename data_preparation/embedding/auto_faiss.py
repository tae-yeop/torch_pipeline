import faiss
import numpy as np
import torch
import os
import argparse

def get_data(args):
    if args.data_path == "random":
        data = np.random.rand(1000, 128).astype('float32')
    
    elif args.data_path.endswith(".pt") and os.path.exists(args.data_path):
        data = torch.load(args.data_path)
        data = data.detach().cpu().numpy().astype(np.float32) # float32로 변환 권장됨

    elif args.data_path.endswith(".npy") and os.path.exists(args.data_path):
        data = np.load(args.data_path).astype(np.float32)

    else:
        raise ValueError(f"Invalid data path: {args.data_path}")

    return data

def get_index(args, data=None):
    # 이미 있는 경우 로드하도록 함
    if args.index_path and os.path.exists(args.index_path):
        index = faiss.read_index(args.index_path)
    else:
        # 인덱스 생성시 임베딩 차원을 넣도록 함
        if args.index_type == "Flat":
            index = faiss.IndexFlatL2(data.shape[1])

        elif args.index_type == "IVFFlat":
            nlist = 50 # cell 갯수
            quantizer = faiss.IndexFlatL2(data.shape[1])
            index = faiss.IndexIVFFlat(quantizer, data.shape[1], nlist)
            index.train(data)

        elif args.index_type == "IVFPQ":
            m = 8 # 서브 벡터 갯수
            nlist = 50 # cell 갯수
            bits = 8 # 각 서브 벡터의 cetnroid의 비트수

            quantizer = faiss.IndexFlatL2(data.shape[1])
            index = faiss.IndexIVFPQ(quantizer, data.shape[1], nlist, m, bits)
            index.train(data)

        elif args.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(data.shape[1])
        else:
            raise ValueError(f"Invalid index type: {args.index_type}")

        index.add(data)

    # True, 갯수가 나와야함
    print(index.is_trained, index.ntotal)
    return index

def get_query_data(args, dim):
    if args.query_data_path == "random":
        query_data = np.random.rand(10, dim).astype(np.float32)
    else:
        query_data = np.load(args.query_data_path)

    return query_data

def search_index(args, index, query_data, k):

    if isinstance(index, faiss.IndexIVFFlat):
        # IndexIVFFlat는 근사적인 결과이기 때문에 잘못된 결과일 수 있음 ⇒  정확성을 높이기 위해서 nprobe 값을 늘리도록
        index.nprobe = 10
    elif isinstance(index, faiss.IndexIVFPQ):
        index.nprobe = 10
    # 검색 시 임베딩 차원을 넣도록 함
    distances, ids = index.search(query_data, k)


    return distances, ids

def get_vector_value(ids, index, k, dim):
    num_queries = ids.shape[0]
    vecs = np.zeros((num_queries, k, dim), dtype=np.float32)

    if isinstance(index, faiss.IndexIVFFlat):
        index.make_direct_map()

    
    for q_idx in range(num_queries):
        for i, val in enumerate(ids.tolist()): # Index[0].tolist() => [22, 11, 67] 인덱스 값이 들어있다
            vecs[q_idx, i] = index.reconstruct(val)

    return vecs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_type", type=str, default="Flat") # Flat, IVFFlat, IVFPQ, HNSW
    parser.add_argument("--index_path", type=str, default="./data/index.faiss") # 인덱스 저장 경로
    parser.add_argument("--data_path", type=str, default="./data/data.npy") #random, 파일 경로
    parser.add_argument("--query_data_path", type=str, default="./data/query_data.npy") #random, 파일 경로
    parser.add_argument("--save_index_path", type=str, default="./data/saved_index.faiss") # 인덱스 저장 경로
    parser.add_argument("--k", type=int, default=5) # 검색 결과 상위 k개
    args = parser.parse_args()


    # 데이터 생성
    data = get_data(args)
    dim = data.shape[1]

    # Faiss 인덱스 생성
    index = get_index(args, data)

    # 검색 데이터 생성
    query_data = get_query_data(args, dim)

    # 검색
    distances, ids = search_index(args, index, query_data, k)

    print(ids)      # 찾아낸 이웃의 ID
    print(distances)  # 이웃까지의 거리
    # [num_q, k] 만큼 리턴된다
    # Index: [[22 11 67]
    # [83  9 60]
    # [35 67 79]
    # [40 74 99]
    # [11 14 93]]

    # Distance: [[879.6943  893.8348  893.86285]
    # [944.20715 958.6526  962.0861 ]
    # [931.2137  931.86273 947.1274 ]
    # [911.0959  915.0969  917.58716]
    # [890.5468  943.226   948.8726 ]]
    vecs = get_vector_value(ids, index, args.k, dim)
    print(vecs)

    if args.save_index_path:
        faiss.write_index(index, args.save_index_path)