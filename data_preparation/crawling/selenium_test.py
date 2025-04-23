from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

chrome_options = Options()
chrome_options.add_argument("--headless") # GUI 없이 동작
# chrome_options.add_argument("--user-data-dir=/tmp/selenium_chrome_profile")
# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("window-size=1400,1500")
# chrome_options.add_argument("start-maximized")
# chrome_options.add_argument("enable-automation")
# chrome_options.add_argument("--disable-infobars")
# chrome_options.add_argument("--disable-dev-shm-usage")


# 경로 지정 없이 webdriver.Chrome() 만으로 드라이버를 자동 설치/사용
driver = webdriver.Chrome(options=chrome_options)

driver.get("https://www.google.com")
print(driver.page_source)


element = driver.find_element(By.NAME, "q")
element.send_keys("Hello Selenium Manager\n")

driver.quit()