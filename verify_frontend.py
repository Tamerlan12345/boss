from playwright.sync_api import sync_playwright
import time

def verify_frontend():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Navigate to the app
        page.goto("http://localhost:8081")

        # Wait for initialization
        time.sleep(2)

        # Screenshot
        page.screenshot(path="verification_screenshot.png")

        # Check for Simli video element
        if page.locator("#simli-video").count() > 0:
            print("Simli video element found.")
        else:
            print("Simli video element NOT found.")

        browser.close()

if __name__ == "__main__":
    verify_frontend()
