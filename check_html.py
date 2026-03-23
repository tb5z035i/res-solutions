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
    time.sleep(8)

    # Get page source and check for dropdown
    source = driver.page_source

    # Save HTML for inspection
    with open('/tmp/ui_page.html', 'w') as f:
        f.write(source)

    print("HTML saved to /tmp/ui_page.html")

    # Check for worker name
    if "mock-worker-1" in source:
        print("✓ Worker found in HTML")
    else:
        print("✗ Worker NOT in HTML")

    # Check for dropdown element
    if "Select Worker" in source:
        print("✓ Dropdown label found")
    else:
        print("✗ Dropdown label NOT found")

    driver.save_screenshot("/tmp/ui_final.png")

finally:
    driver.quit()
