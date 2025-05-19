크롤링 대표적인 도구 2가지 : Selenium 과 Scrapy

	
Selenium	마치 사용자가 브라우저를 사용하는 것처럼 브라우저를 동작시켜서, 크롤링을 하는 기술
동적 웹페이지를 크롤링할 수 있음

원래 웹 앱을 테스트하는데 사용하는 프레임워크
Scrapy	대량의 데이터 크롤링을 위한 크롤링 프레임워크
대량의 데이터 크롤링을 위해 필요한 다양한 기능 제공

최신 버전에선 크롬드라이버는 따로 내려받아서 설치할 필요는 없어짐
(1) 크롬 설치
- `chromium-browser`은 해봤는데 잘안되는듯

```
wget https://dl-ssl.google.com/linux/linux_signing_key.pub -O /tmp/google.pub
ls -l /tmp/google.pub
mkdir -p /etc/apt/keyrings
gpg --no-default-keyring --keyring /etc/apt/keyrings/google-chrome.gpg --import /tmp/google.pub
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main'   | tee /etc/apt/sources.list.d/google-chrome.list
apt-get update
apt-get install google-chrome-stable
```
(2) selenium 설치
```python    
pip install selenium webdriver_manager
```

---
# 통신

GET method
데이터를 url에 포함 (&로 구분)

POST
데이터를 숨겨서 전송 (url 변화 x)

웹서버의 다양성
- 웹서버 마다 데이터를 요청하는 방식 다름
    - get/post 메소드
    - Headers
    - Parameters

- 웹서버마다 데이터를 반환하는 방식 다름
    - 여러 번에 걸쳐 데이터를 반환
    - 반환하는 데이터의 포맷이 다름 (XML, HTML)
    - 파이썬 dict 형태, 문자열인 JSON반환

Requests : 웹서버와 HTTP 요청 처리
Beautiful Soup :
HTML Tag를 파싱하는 라이브러리, CSS Selector로 원하는 데이터 선택