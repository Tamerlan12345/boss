from playwright.sync_api import sync_playwright
import time

def verify_admin_tabs():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Go to Admin
        page.goto("http://localhost:8080/admin")

        # Verify Tabs Buttons
        print("Checking Tabs Buttons...")
        assert page.locator("button.tab-btn", has_text="Эфир и Управление").is_visible()
        assert page.locator("button.tab-btn", has_text="Разговор").is_visible()
        assert page.locator("button.tab-btn", has_text="Настройки").is_visible()

        # Screenshot 1: Default View (Broadcast Tab)
        page.screenshot(path="verification/tab_broadcast.png")
        print("Captured Broadcast Tab.")

        # Click Conversation Tab
        page.click("button.tab-btn:has-text('Разговор')")
        time.sleep(0.5) # Wait for potential transition

        # Verify Logs container is visible
        assert page.locator("#logs").is_visible()

        # Screenshot 2: Conversation Tab
        page.screenshot(path="verification/tab_conversation.png")
        print("Captured Conversation Tab.")

        # Click Settings Tab
        page.click("button.tab-btn:has-text('Настройки')")
        time.sleep(0.5)

        # Verify Settings
        assert page.locator("button", has_text="Сжать контекст памяти").is_visible()

        # Screenshot 3: Settings Tab
        page.screenshot(path="verification/tab_settings.png")
        print("Captured Settings Tab.")

        browser.close()

if __name__ == "__main__":
    verify_admin_tabs()
