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