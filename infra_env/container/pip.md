
한번에 버전 맞춰서 설치
```
pip install --upgrade "numpy>=1.23,<2.0a0" "pyarrow<15"
```
디펜던시 없이 설치하기
```
pip install --no-deps biotite==1.0.1
```
호환성 문제 체크하기
```
python -m pip check
```