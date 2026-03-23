from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()
driver.get("http://localhost:7000/ui")

try:
    # Wait for page to load
    time.sleep(3)

    # Check if worker dropdown exists and has options
    print("Checking for worker dropdown...")
    dropdown = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "select, input[role='combobox']"))
    )
    print(f"Dropdown found: {dropdown.tag_name}")

    # Try to find refresh button
    refresh_btn = driver.find_elements(By.XPATH, "//button[contains(text(), 'Refresh')]")
    if refresh_btn:
        print("Clicking refresh button...")
        refresh_btn[0].click()
        time.sleep(2)

    # Take screenshot
    driver.save_screenshot("/tmp/ui_test.png")
    print("Screenshot saved to /tmp/ui_test.png")

    time.sleep(5)

except Exception as e:
    print(f"Error: {e}")
    driver.save_screenshot("/tmp/ui_error.png")
finally:
    driver.quit()
