import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)

try:
    print("Opening UI...")
    driver.get("http://localhost:7000/ui/")
    time.sleep(5)

    print("Looking for dropdown...")
    dropdowns = driver.find_elements(By.TAG_NAME, "select")
    inputs = driver.find_elements(By.CSS_SELECTOR, "input[role='combobox']")

    print(f"Found {len(dropdowns)} select elements")
    print(f"Found {len(inputs)} combobox inputs")

    # Try to find refresh button
    buttons = driver.find_elements(By.TAG_NAME, "button")
    print(f"Found {len(buttons)} buttons")
    for btn in buttons:
        if "Refresh" in btn.text:
            print("Clicking Refresh Workers button...")
            btn.click()
            time.sleep(2)
            break

    driver.save_screenshot("/tmp/ui_screenshot.png")
    print("Screenshot saved to /tmp/ui_screenshot.png")

    # Check page source for worker name
    if "mock-worker-1" in driver.page_source:
        print("✓ Worker name found in page!")
    else:
        print("✗ Worker name NOT found in page")

finally:
    driver.quit()
