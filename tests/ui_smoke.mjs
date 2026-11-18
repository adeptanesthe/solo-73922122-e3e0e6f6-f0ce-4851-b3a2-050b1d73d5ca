/* Headless smoke test of the UI against a live server using jsdom.
 *
 * Usage: node tests/ui_smoke.mjs http://127.0.0.1:8123
 * Requires jsdom available from the working directory (see README).
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

// jsdom is an optional dev dependency; point JSDOM_DIR at a node_modules
// directory that contains it when it is not installed next to this file.
const require = createRequire(
  process.env.JSDOM_DIR ? path.join(process.env.JSDOM_DIR, "noop.js") : import.meta.url,
);
const { JSDOM, VirtualConsole } = require("jsdom");

const BASE = process.argv[2] || "http://127.0.0.1:8123";
const root = path.resolve(new URL("..", import.meta.url).pathname);

const html = fs.readFileSync(path.join(root, "app/static/index.html"), "utf8");
const script = fs.readFileSync(path.join(root, "app/static/app.js"), "utf8");

const virtualConsole = new VirtualConsole();
const errors = [];
virtualConsole.on("jsdomError", (e) => errors.push(String(e)));
virtualConsole.on("error", (e) => errors.push(String(e)));

const dom = new JSDOM(html, {
  url: BASE + "/",
  runScripts: "outside-only",
  virtualConsole,
  pretendToBeVisual: true,
});
const { window } = dom;
window.fetch = (input, init) => fetch(new URL(input, BASE), init);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, label, timeout = 20000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (fn()) return;
    await sleep(50);
  }
  throw new Error("timeout waiting for " + label);
}
const $ = (id) => window.document.getElementById(id);

window.eval(script);
await sleep(200);

const results = [];
function check(name, condition, detail = "") {
  results.push({ name, ok: !!condition, detail });
  console.log(`${condition ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
}

// 1. samples are listed
await until(() => $("sample-buttons").children.length > 0, "sample buttons");
check("样例按钮已渲染", $("sample-buttons").children.length >= 3,
  $("sample-buttons").textContent.trim());

// 2. load the repeat sample
const repeatBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("sample_repeat"));
repeatBtn.click();
await until(() => !$("workspace").hidden && $("progress-text").textContent !== "0 / 0 帧", "sample loaded");
check("载入样例后显示工作区", !$("workspace").hidden, $("progress-text").textContent);
check("初始无候选字幕", $("confirmed-text").textContent === "", `“${$("confirmed-text").textContent}”`);

// 3. advance one chunk
window.document.getElementById("chunk-size").value = "5";
$("chunk-size").dispatchEvent(new window.Event("input"));
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("5 /"), "advance chunk");
check("分块推进更新进度", $("progress-text").textContent.startsWith("5 /"), $("progress-text").textContent);
check("候选表非空", $("beam-body").children.length > 0, `${$("beam-body").children.length} 束`);
const firstText = $("beam-body").children[0].querySelector(".beam-text").textContent;
check("最佳候选含字符", firstText.length > 0, firstText);

// 4. confirm the best candidate
$("confirm-best").click();
await until(() => $("confirmed-text").textContent.length > 0, "confirm");
const confirmed = $("confirmed-text").textContent;
check("确认后前缀显示", confirmed.length > 0, confirmed);
check("blank 分数显示", $("ps-blank").textContent !== "—", $("ps-blank").textContent);
check("非 blank 分数显示", $("ps-nonblank").textContent !== "—", $("ps-nonblank").textContent);
check("撤销按钮可用", !$("undo-btn").disabled);

// 5. advance after confirmation, then undo
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("10 /"), "advance after confirm");
$("undo-btn").click();
await until(() => $("confirmed-text").textContent === "", "undo");
check("撤销后前缀清空", $("confirmed-text").textContent === "");
check("撤销后进度保留", $("progress-text").textContent.startsWith("10 /"), $("progress-text").textContent);
check("撤销后候选非空", $("beam-body").children.length > 0);

// 6. failure keeps state: confirm text not in beam
$("prefix-input").value = "zzzz";
$("apply-prefix").click();
await until(() => !$("op-error").hidden, "error shown");
const beforeProgress = $("progress-text").textContent;
check("非法确认显示错误", !$("op-error").hidden, $("op-error").textContent);
check("错误后进度不变", $("progress-text").textContent === beforeProgress, $("progress-text").textContent);

// 7. historical frame inspection
$("peek-frame").value = "2";
$("peek-frame").dispatchEvent(new window.Event("input"));
await sleep(400);
check("历史帧提示只读", $("peek-note").textContent.includes("历史回放"), $("peek-note").textContent);
check("历史帧概率已渲染", $("frame-probs").children.length > 0);

// 8. continuous run to the end
$("peek-frame").value = $("peek-frame").max;
$("peek-frame").dispatchEvent(new window.Event("input"));
$("run-btn").click();
await until(() => $("progress-text").textContent.startsWith("15 / 15"), "run to end", 30000);
check("连续推进到结尾", $("progress-text").textContent.startsWith("15 / 15"), $("progress-text").textContent);
await until(() => $("stop-btn").hidden, "run loop finished");
check("运行结束停止按钮隐藏", $("stop-btn").hidden);

// 9. zero-probability sample shows -∞ without NaN
const zeroBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("sparse_zero"));
zeroBtn.click();
await until(() => $("progress-text").textContent.startsWith("0 / 7"), "zero sample loaded");
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("7 / 7") || $("progress-text").textContent.startsWith("5 / 7"), "zero sample advance");
await sleep(300);
const bodyText = $("beam-body").textContent;
check("零概率样例无 NaN", !bodyText.includes("NaN"), bodyText.slice(0, 120));
check("零概率样例显示 -∞", bodyText.includes("-∞") || $("frame-probs").textContent.includes("0.000"));

// 10. no uncaught UI errors
check("无未捕获脚本错误", errors.length === 0, errors.join(" | ").slice(0, 200));

// 11. speaker attribution analysis on the dedicated two-speaker sample
const attrBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("sample_attribution"));
check("归属样例按钮存在", !!attrBtn);
if (attrBtn) {
  attrBtn.click();
  await until(() => $("progress-text").textContent.startsWith("0 / 9"), "attribution sample loaded");
  // advance all frames so the whole window is available
  $("run-btn").click();
  await until(() => $("progress-text").textContent.startsWith("9 / 9"), "attr advance", 30000);
  await until(() => $("stop-btn").hidden, "attr run finished");

  // configure the window and reference
  $("attr-start").value = "0";
  $("attr-end").value = "6";
  $("attr-start").dispatchEvent(new window.Event("input"));
  $("attr-end").dispatchEvent(new window.Event("input"));
  $("attr-rebuild").click();
  await until(() => $("attr-speaker-body").children.length === 7, "speaker table built");
  check("说话人表格按帧区间生成 7 行", $("attr-speaker-body").children.length === 7);
  check("样例填表按钮可见", !$("attr-fill-sample").hidden);
  $("attr-fill-sample").click();
  await sleep(100);
  const firstA = $("attr-speaker-body").children[0].querySelectorAll("input")[0].value;
  check("样例说话人概率已填入", firstA === "0.92", firstA);
  // rows sum to 1 -> run button enabled
  check("行和合法时计算按钮可用", !$("attr-run").disabled);

  $("attr-reference").value = "abc";
  $("attr-reference").dispatchEvent(new window.Event("input"));
  $("attr-run").click();
  await until(() => !$("attr-result").hidden && $("attr-result-body").children.length === 3, "attribution result", 20000);
  check("归属结果逐字显示 3 行", $("attr-result-body").children.length === 3);
  const speakers = [...$("attr-result-body").children].map((tr) => tr.children[1].textContent.trim());
  check("首字归甲、含乙归属", speakers[0].includes("甲") && speakers.some((s) => s.includes("乙")), speakers.join(","));
  check("逐帧证据表已渲染", $("attr-evidence-body").children.length === 7);
  check("成功后无过期横幅", $("attr-stale").hidden);

  // editing an input marks the previous result stale
  $("attr-reference").value = "ab";
  $("attr-reference").dispatchEvent(new window.Event("input"));
  check("更改输入后标记过期", !$("attr-stale").hidden);

  // invalid row sum keeps the previous result and disables run
  const inputs = $("attr-speaker-body").children[0].querySelectorAll("input");
  inputs[0].value = "0.9";
  inputs[1].value = "0.9";
  inputs[1].dispatchEvent(new window.Event("input"));
  check("行和非法时计算按钮禁用", $("attr-run").disabled);
  check("非法输入保留旧结果", !$("attr-result").hidden && $("attr-result-body").children.length === 3);

  // advancing after a fresh result flags it stale
  inputs[0].value = "0.92";
  inputs[1].value = "0.08";
  inputs[1].dispatchEvent(new window.Event("input"));
  $("attr-reference").value = "abc";
  $("attr-reference").dispatchEvent(new window.Event("input"));
  $("attr-run").click();
  await until(() => $("attr-stale").hidden, "recompute fresh");
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
