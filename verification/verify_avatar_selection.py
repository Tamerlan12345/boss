from playwright.sync_api import sync_playwright, expect

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://localhost:8000")

    # Check button is disabled initially
    connect_btn = page.locator("#connectBtn")
    # It might happen very fast, so we check if it has 'disabled' attribute.
    # But since the page loads and JS runs almost immediately, we might miss the initial "true" state if we are not fast enough,
    # OR if the JS runs and enables it very quickly.
    # However, the user wants to see it enabled AFTER the check.

    # We wait for the status to become 'Ready to Connect'
    expect(page.locator("#status")).to_have_text("Ready to Connect", timeout=10000)

    # Check button is enabled now
    expect(connect_btn).to_be_enabled()

    # Check logs for the switch message
    # "Switched to available avatar: mock_avatar_id_123 (Mock Avatar)"
    logs = page.locator("#logs")
    expect(logs).to_contain_text("Switched to available avatar: mock_avatar_id_123")

    # Check console logs? Playwright can capture console, but here we printed to #logs div too in the JS code?
    # No, console.log("Available Avatars:", avatars) goes to browser console.
    # But we also have log() function that writes to #logs div.
    # log(`Switched to available avatar: ${activeAvatarId} (${avatars[0].name})`);

    page.screenshot(path="/home/jules/verification/verification.png")
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
