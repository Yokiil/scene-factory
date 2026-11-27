"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { launchBrowser } = require("./qa_browser.cjs");

const base = process.env.SCENE_FACTORY_QA_URL || "http://127.0.0.1:8765/";
const output = path.resolve(process.env.SCENE_FACTORY_QA_OUTPUT || "outputs/dessert_display/browser");
fs.mkdirSync(output, { recursive: true });

async function main() {
  const browser = await launchBrowser();
  const page = await browser.newPage({ viewport: { width: 1500, height: 1050 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    await page.goto(base, { waitUntil: "domcontentloaded" });
    await page.getByRole("button", { name: "三层甜点架", exact: true }).click();
    const responsePromise = page.waitForResponse(
      (response) => response.url().endsWith("/api/generate")
        && response.request().method() === "POST",
      { timeout: 180000 },
    );
    await page.locator("#generate").click();
    const response = await responsePromise;
    const payload = await response.json();
    assert.equal(response.status(), 200, JSON.stringify(payload));
    const item = payload.items[0];
    assert.equal(item.matched_recipe.name, "three_tier_dessert_display");
    assert.equal(item.validation.valid, true);
    assert.equal(item.scene.objects.length, 43);
    assert.equal(item.appearance.proxy_count, 0);
    assert.ok(Object.values(item.assets).every((asset) => asset.preserve_visual_materials));
    await page.waitForFunction(
      () => document.querySelector("#viewer-status").dataset.state === "ready",
      null, { timeout: 60000 },
    );
    assert.match(await page.locator("#viewer-status").textContent(), /0 个占位/);
    await page.locator("#result-card").screenshot({ path: path.join(output, "desktop.png") });
    await page.getByLabel("Three.js 场景三维预览", { exact: true }).screenshot({ path: path.join(output, "scene.png") });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator("#result-card").screenshot({ path: path.join(output, "mobile.png") });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({
      scene_id: item.scene.scene_id,
      validation: item.validation,
      appearance: item.appearance,
      errors,
    }, null, 2));
    console.log(JSON.stringify({ scene_id: item.scene.scene_id, objects: 43, errors }));
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
