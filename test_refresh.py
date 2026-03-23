import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)

try:
    driver.get("http://localhost:7000/ui/")
    time.sleep(5)

    # Find and click Refresh Workers button
    buttons = driver.find_elements(By.TAG_NAME, "button")
    for btn in buttons:
        if "Refresh" in btn.text:
            print("Clicking Refresh Workers...")
            btn.click()
            time.sleep(3)
            break

    # Check if worker appears
    if "mock-worker-1" in driver.page_source:
        print("✓ SUCCESS: Worker found after refresh!")
    else:
        print("✗ FAIL: Worker not found")

    driver.save_screenshot("/tmp/ui_after_refresh.png")
    print("Screenshot: /tmp/ui_after_refresh.png")

finally:
    driver.quit()
