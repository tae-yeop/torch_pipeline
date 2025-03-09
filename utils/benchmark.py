import time

a = time.perf_counter()
mask, label_names = face_parser(img)
print(f'추론 시간: {time.perf_counter() - a: 0.4f} 초')

