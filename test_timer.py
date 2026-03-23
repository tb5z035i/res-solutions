import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)

try:
    driver.get("http://localhost:7000/ui/")
    print("Waiting 10 seconds for timer to poll workers...")
    time.sleep(10)

    if "mock-worker-1" in driver.page_source:
        print("✓ SUCCESS: Worker found!")
    else:
        print("✗ FAIL: Worker not found")

    driver.save_screenshot("/tmp/ui_final_test.png")

finally:
    driver.quit()
