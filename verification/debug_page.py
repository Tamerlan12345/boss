from playwright.sync_api import sync_playwright

def debug():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:8080/admin")
        page.screenshot(path="verification/debug_admin.png")
        print(page.content())
        browser.close()

if __name__ == "__main__":
    debug()
