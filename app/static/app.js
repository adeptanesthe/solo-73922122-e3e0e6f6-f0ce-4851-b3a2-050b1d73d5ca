/* Streaming CTC workbench UI.
 *
 * Every server response carries a monotonically increasing revision.  The UI
 * remembers the revision it currently shows and ignores any response that is
 * older, so a slow advance or undo replay can never overwrite a newer
 * confirm / undo.  Mutating requests are additionally serialised in a small
 * client-side queue, and while a request is in flight the saved subtitle and
 * progress stay on screen until the answer arrives (or an error is shown).
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    connStatus: $("conn-status"),
    setupPanel: $("setup-panel"),
    vocabFile: $("vocab-file"),
    framesFile: $("frames-file"),
    beamWidth: $("beam-width"),
    beamOut: $("beam-out"),
    chunkSize: $("chunk-size"),
    chunkOut: $("chunk-out"),
    loadBtn: $("load-btn"),
    resetBtn: $("reset-btn"),
    setupHint: $("setup-hint"),
    setupError: $("setup-error"),
    sampleButtons: $("sample-buttons"),
    workspace: $("workspace"),
    progressBar: $("progress-bar"),
    progressText: $("progress-text"),
    confirmedText: $("confirmed-text"),
    pendingText: $("pending-text"),
    confirmBest: $("confirm-best"),
    undoBtn: $("undo-btn"),
    stepBtn: $("step-btn"),
    runBtn: $("run-btn"),
    restartBtn: $("restart-btn"),
    stopBtn: $("stop-btn"),
    prefixInput: $("prefix-input"),
    applyPrefix: $("apply-prefix"),
    psText: $("ps-text"),
    psBlank: $("ps-blank"),
    psNonblank: $("ps-nonblank"),
    psTotal: $("ps-total"),
    opError: $("op-error"),
    opNotice: $("op-notice"),
    beamBody: $("beam-body"),
    beamEmpty: $("beam-empty"),
    peekFrame: $("peek-frame"),
    peekFrameOut: $("peek-frame-out"),
    peekNote: $("peek-note"),
    frameProbs: $("frame-probs"),
    historyList: $("history-list"),
    historyEmpty: $("history-empty"),
    busy: $("busy"),
    busyText: $("busy-text"),
    // Speaker attribution panel.
    attrPanel: $("attribution-panel"),
    attrStart: $("attr-start"),
    attrStartOut: $("attr-start-out"),
    attrEnd: $("attr-end"),
    attrEndOut: $("attr-end-out"),
    attrReference: $("attr-reference"),
    attrRebuild: $("attr-rebuild"),
    attrFillSample: $("attr-fill-sample"),
    attrSpeakerBody: $("attr-speaker-body"),
    attrTableEmpty: $("attr-table-empty"),
    attrRun: $("attr-run"),
    attrRowHint: $("attr-row-hint"),
    attrError: $("attr-error"),
    attrResult: $("attr-result"),
    attrStale: $("attr-stale"),
    attrStaleInputs: $("attr-stale-inputs"),
    attrStaleRevision: $("attr-stale-revision"),
    attrMetaText: $("attr-meta-text"),
    attrResultBody: $("attr-result-body"),
    attrEvidenceHead: $("attr-evidence-head"),
    attrEvidenceBody: $("attr-evidence-body"),
  };

  const state = {
    sessionId: null,
    generation: 0,      // bumped whenever a new session replaces the old one
    revision: -1,       // revision of the state currently displayed
    snapshot: null,     // last state object accepted from the server
    busy: false,
    running: false,
    runAbort: false,
    pending: 0,
    samples: [],
    // Behind the scenes: the last shown state is kept while a request is in
    // flight, so a failed operation leaves the subtitles and progress intact.
    peekLive: false,
  };

  // Speaker-analysis state, kept separate from the decoding snapshot so that
  // stale results survive input edits and advances until replaced.
  const attribution = {
    rows: [],            // current editable speaker rows: [{a, b, el...}]
    lastResult: null,    // last successful response payload
    lastSignature: null, // input signature + revision the result was computed for
    requestSeq: 0,       // client-side monotonic id; older replies are dropped
    inFlight: false,
    sampleSpeakers: null,
  };

  // ---------------------------------------------------------------- helpers

  const fmt = (value) => {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number" || Number.isNaN(value)) return "—";
    if (!Number.isFinite(value)) return "-∞";
    return value.toFixed(4);
  };

  const pct = (value) => {
    if (!Number.isFinite(value)) return "0";
    if (value >= 0.01) return (value * 100).toFixed(1) + "%";
    return value.toExponential(2);
  };

  function setConnection(kind, label) {
    els.connStatus.dataset.state = kind;
    els.connStatus.textContent = label;
  }

  function setBusy(on, text = "计算中…") {
    state.busy = on;
    els.busy.hidden = !on;
    els.busyText.textContent = text;
    if (on) setConnection("busy", "计算中");
    else updateStatusFromSnapshot();
    updateButtons();
  }

  // Mark an operation as in flight.  Operations are *not* blocked while
  // another one runs: the server applies them in submission order and each
  // response is revision-guarded, so advancing / undoing during a replay is
  // safe and the late replay result cannot overwrite the newer action.
  function beginOp(label) {
    state.pending += 1;
    setBusy(true, label);
  }

  function endOp() {
    state.pending = Math.max(0, state.pending - 1);
    if (state.pending === 0) setBusy(false);
  }

  function updateStatusFromSnapshot() {
    if (!state.snapshot) { setConnection("idle", "未加载"); return; }
    if (state.running) setConnection("busy", "连续推进中");
    else setConnection("ready", "就绪");
  }

  function showError(message) {
    hideNotice();
    if (!message) { els.opError.hidden = true; els.opError.textContent = ""; return; }
    els.opError.hidden = false;
    els.opError.textContent = message;
  }

  function showNotice(message) {
    if (!message) { els.opNotice.hidden = true; els.opNotice.textContent = ""; return; }
    els.opNotice.hidden = false;
    els.opNotice.textContent = message;
  }

  function hideNotice() { els.opNotice.hidden = true; els.opNotice.textContent = ""; }

  function updateButtons() {
    const ready = !!state.sessionId && !!state.snapshot;
    const atEnd = ready && state.snapshot.next_frame >= state.snapshot.frame_count;
    // Recompute (undo replay / long advance) deliberately does NOT disable
    // advance or undo: those actions are queued by the server behind the
    // running operation and applied to its result, in order.
    els.confirmBest.disabled = !ready;
    els.undoBtn.disabled = !ready || (state.snapshot.confirmed_count || 0) === 0;
    els.stepBtn.disabled = !ready || atEnd || state.running;
    els.runBtn.disabled = !ready || atEnd || state.running;
    els.restartBtn.disabled = !ready || state.running;
    els.applyPrefix.disabled = !ready;
    els.stopBtn.hidden = !state.running;
    els.resetBtn.disabled = !ready;
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let payload = null;
    try { payload = await response.json(); } catch { payload = null; }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = response.status;
      throw err;
    }
    return payload;
  }

  // Apply a server state snapshot, unless it belongs to a replaced session or
  // is older than the state already displayed.  Together with the server's
  // in-order worker this is what makes late results harmless.
  function applyState(payload, { force = false } = {}) {
    if (!payload || typeof payload.revision !== "number") return false;
    if (payload.session_id !== state.sessionId) return false;
    if (!force && payload.revision < state.revision) return false;
    state.revision = payload.revision;
    state.snapshot = payload;
    render();
    return true;
  }

  function startSession(payload, sampleName = null) {
    // Invalidate anything still in flight for the previous session.
    state.generation += 1;
    state.runAbort = true;
    state.running = false;
    state.pending = 0;
    els.stopBtn.hidden = true;
    state.sessionId = payload.session_id;
    state.revision = -1;
    state.snapshot = null;
    state.loadedSample = sampleName;
    state.peekLive = true;

    // Reset the speaker-analysis panel for the new session.
    attribution.requestSeq += 1;
    attribution.inFlight = false;
    attribution.rows = [];
    attribution.lastResult = null;
    attribution.lastSignature = null;
    attribution.sampleSpeakers = null;
    els.attrResult.hidden = true;
    els.attrStale.hidden = true;
    els.attrSpeakerBody.innerHTML = "";
    els.attrTableEmpty.hidden = false;
    els.attrReference.value = "";
    els.attrStart.value = "0";
    els.attrEnd.value = "0";
    els.attrFillSample.hidden = true;
    showAttrError("");

    applyState(payload, { force: true });
    if (state.snapshot) {
      els.peekFrame.value = "0";
      els.peekFrame.max = String(state.snapshot.frame_count);
      els.peekFrameOut.textContent = `0 / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
    }
    syncAttributionBounds();
    updateButtons();
    updateStatusFromSnapshot();
  }

  // ---------------------------------------------------------------- rendering

  function render() {
    const snap = state.snapshot;
    if (!snap) { els.workspace.hidden = true; return; }
    els.workspace.hidden = false;

    const total = snap.frame_count || 0;
    const done = snap.next_frame || 0;
    els.progressBar.style.width = total ? `${(100 * done) / total}%` : "0%";
    els.progressText.textContent = `${done} / ${total} 帧`;

    const best = snap.best_text || "";
    const prefix = snap.prefix || "";
    const pending = best.startsWith(prefix) ? best.slice(prefix.length) : best;
    els.confirmedText.textContent = prefix;
    els.pendingText.textContent = pending;

    const scores = snap.prefix_scores || {};
    els.psText.textContent = scores.text ? `“${scores.text}” → ${scores.last_char}` : "—";
    els.psBlank.textContent = fmt(scores.logp_blank);
    els.psNonblank.textContent = fmt(scores.logp_nonblank);
    els.psBlank.title = "确认时的末字 blank 路径概率（保留值，不随新帧归一化）";
    els.psNonblank.title = "确认时的末字非 blank 路径概率（保留值，不随新帧归一化）";
    els.psTotal.textContent = fmt(scores.logp);
    if (scores.text) {
      els.psTotal.title =
        `确认于第 ${scores.confirmed_at} 帧；此前缀“${scores.text}”的完整前向分数 ` +
        `为 ${fmt(scores.logp)}，其中末字前的前缀分数为 ${fmt(scores.previous_logp)}。` +
        "值保留自确认时刻，不随后续帧或再次确认归一化。";
    } else {
      els.psTotal.title = "尚未确认前缀";
    }

    renderBeams(snap, prefix);

    els.peekFrame.max = String(total);
    if (Number(els.peekFrame.value) > total) els.peekFrame.value = String(total);
    els.peekFrameOut.textContent = `${els.peekFrame.value} / ${total}`;
    els.peekNote.textContent =
      Number(els.peekFrame.value) === done ? "显示当前帧" : "历史回放（只读）";

    renderHistory(snap);
    renderFrameProbs(snap.frame_info);
    els.prefixInput.placeholder = prefix ? `当前前缀：${prefix}` : "例如：啊";
    syncAttributionBounds();
    // Any advance / confirm / undo bumps the revision, which invalidates a
    // previously computed attribution result until it is recomputed.
    markAttributionStale();
    updateButtons();
    updateStatusFromSnapshot();
  }

  function renderBeams(snap, prefix) {
    const beams = snap.beams || [];
    if (typeof snap.revision === "number") state.peekLive = true;
    els.beamBody.innerHTML = "";
    els.beamEmpty.hidden = beams.length > 0;
    beams.forEach((beam, index) => {
      const tr = document.createElement("tr");
      if (index === 0) tr.className = "selected";

      const idx = document.createElement("td");
      idx.textContent = String(index + 1);

      const textTd = document.createElement("td");
      const span = document.createElement("span");
      span.className = "beam-text";
      const rest = beam.text.startsWith(prefix) ? beam.text.slice(prefix.length) : beam.text;
      if (prefix) {
        const pfx = document.createElement("span");
        pfx.className = "pfx";
        pfx.textContent = prefix;
        span.appendChild(pfx);
      }
      const tail = document.createElement("span");
      if (rest) tail.textContent = rest;
      else { tail.className = "empty-tag"; tail.textContent = "∅"; }
      span.appendChild(tail);
      textTd.appendChild(span);
      if (beam.text.startsWith(prefix)) textTd.title = beam.text;

      const logpTd = document.createElement("td");
      logpTd.className = "num";
      logpTd.textContent = fmt(beam.logp);

      const probTd = document.createElement("td");
      probTd.className = "num";
      probTd.textContent = pct(beam.prob);

      const blankTd = document.createElement("td");
      blankTd.className = "num";
      blankTd.textContent = fmt(beam.logp_blank);

      const nbTd = document.createElement("td");
      nbTd.className = "num";
      nbTd.textContent = fmt(beam.logp_nonblank);

      const actTd = document.createElement("td");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "确认此候选";
      btn.title = `确认前缀：${beam.text}`;
      btn.disabled = !state.peekLive;
      btn.addEventListener("click", () => confirmPrefix(beam.text));
      actTd.appendChild(btn);

      tr.append(idx, textTd, logpTd, probTd, blankTd, nbTd, actTd);
      els.beamBody.appendChild(tr);
    });
  }

  function renderHistory(snap) {
    const list = snap.confirmations || [];
    els.historyList.innerHTML = "";
    els.historyEmpty.hidden = list.length > 0;
    list.forEach((entry, index) => {
      const li = document.createElement("li");
      const left = document.createElement("span");
      const badge = document.createElement("span");
      badge.className = "h-meta";
      badge.textContent = `#${entry.seq} @${entry.frame} 帧`;
      const prefixSpan = document.createElement("span");
      prefixSpan.className = "h-prefix";
      prefixSpan.textContent = entry.prefix_after || "(空)";
      left.append(badge, document.createTextNode("  "), prefixSpan);

      const right = document.createElement("span");
      right.className = "h-meta";
      right.textContent =
        `logP ${fmt(entry.confirmed_logp)} · blank ${fmt(entry.confirmed_logp_blank)}` +
        ` · 非 blank ${fmt(entry.confirmed_logp_nonblank)}`;

      li.append(left, right);
      if (index === list.length - 1) li.style.borderColor = "rgba(79, 209, 197, .45)";
      els.historyList.appendChild(li);
    });
  }

  function renderFrameProbs(info) {
    els.frameProbs.innerHTML = "";
    if (!info || !info.probs) return;
    const vocab = info.vocab || [];
    info.probs.forEach((value, index) => {
      const cell = document.createElement("div");
      cell.className = "fp" + (value === 0 ? " zero" : "");
      const tok = document.createElement("span");
      tok.className = "tok";
      tok.textContent = index === 0 ? "<blank>" : vocab[index];
      const val = document.createElement("span");
      val.className = "val";
      val.textContent = "  " + value.toFixed(3);
      cell.append(tok, val);
      cell.title = `log P = ${info.log_probs && info.log_probs[index] !== null ? Number(info.log_probs[index]).toFixed(4) : "-∞"}`;
      els.frameProbs.appendChild(cell);
    });
  }

  // ------------------------------------------------------------- operations

  async function withBusy(label, fn) {
    beginOp(label);
    try {
      return await fn();
    } finally {
      endOp();
    }
  }

  async function createSession(formData) {
    els.setupError.hidden = true;
    setBusy(true, "解析并创建会话…");
    try {
      const payload = await api("/api/sessions/upload", { method: "POST", body: formData, headers: {} });
      startSession(payload);
      showError("");
      hideNotice();
      els.loadBtn.disabled = false;
      setConnection("ready", "就绪");
    } catch (error) {
      els.setupError.hidden = false;
      els.setupError.textContent = error.message;
      setConnection("error", "加载失败");
    } finally {
      setBusy(false);
      els.loadBtn.disabled = false;
    }
  }

  async function advanceTo(target, { silentBusy = false } = {}) {
    if (!state.sessionId) return false;
    const sessionId = state.sessionId;
    const generation = state.generation;
    const run = async () => {
      try {
        const payload = await api("/api/advance", {
          method: "POST",
          body: JSON.stringify({
            session_id: sessionId,
            to_frame: target,
          }),
        });
        if (generation !== state.generation || sessionId !== state.sessionId) return false;
        const applied = applyState(payload);
        if (applied) { showError(""); hideNotice(); }
        return applied;
      } catch (error) {
        if (generation !== state.generation || sessionId !== state.sessionId) return false;
        showError(`推进失败：${error.message}（字幕与进度保持不变）`);
        return false;
      }
    };
    return silentBusy ? run() : withBusy("推进帧…", run);
  }

  async function confirmPrefix(prefix) {
    if (!state.sessionId) return;
    if (!prefix || prefix === (state.snapshot && state.snapshot.prefix)) {
      showError("请输入比当前已确认前缀更长的文本");
      return;
    }
    if (!state.peekLive) {
      showError("正在查看历史帧；请先把查看帧拖回当前帧再确认");
      return;
    }
    const sessionId = state.sessionId;
    const generation = state.generation;
    await withBusy("确认前缀…", async () => {
      try {
        const payload = await api("/api/confirm", {
          method: "POST",
          body: JSON.stringify({
            session_id: sessionId,
            prefix,
          }),
        });
        if (generation !== state.generation) return;
        if (applyState(payload)) {
          showError("");
          showNotice(`已确认前缀「${prefix}」。后续只保留兼容候选。`);
          els.prefixInput.value = "";
        }
      } catch (error) {
        showError(`确认失败：${error.message}（前缀与候选保持不变）`);
      }
    });
  }

  async function undo() {
    if (!state.sessionId) return;
    const sessionId = state.sessionId;
    const generation = state.generation;
    await withBusy("撤销确认并重算…", async () => {
      try {
        const payload = await api("/api/undo", {
          method: "POST",
          body: JSON.stringify({ session_id: sessionId }),
        });
        if (generation !== state.generation) return;
        if (applyState(payload)) {
          showError("");
          showNotice("已撤销上次确认，束与分数恢复到未作该次确认时应有的状态。");
          els.prefixInput.value = "";
        }
      } catch (error) {
        showError(`撤销失败：${error.message}（字幕与进度保持不变）`);
      }
    });
  }

  async function peek(frame) {
    if (!state.sessionId) return;
    try {
      const payload = await api("/api/peek", {
        method: "POST",
        body: JSON.stringify({ session_id: state.sessionId, frame }),
      });
      // Peek responses are informational only; they must not clobber the live
      // beam table while newer state is displayed.
      if (payload.revision !== state.revision) return;
      if (!state.snapshot) return;
      if (frame === state.snapshot.next_frame) {
        // Back on the live boundary: show the live state, not a replay.
        state.peekLive = true;
        renderBeams(state.snapshot, state.snapshot.prefix || "");
        renderFrameProbs(state.snapshot.frame_info);
        els.peekNote.textContent = "显示当前帧";
        return;
      }
      const prefix = payload.prefix || "";
      renderBeams({ beams: payload.beams, prefix }, prefix);
      state.peekLive = false;
      renderFrameProbs(payload.frame_info);
      els.peekNote.textContent = `历史回放：第 ${frame} 帧（只读，确认按钮无效）`;
    } catch (error) {
      if (frame === (state.snapshot ? state.snapshot.next_frame : -1)) {
        showError(`查看帧失败：${error.message}`);
      }
    }
  }

  async function runContinuously() {
    if (!state.sessionId || !state.snapshot || state.running) return;
    state.running = true;
    state.runAbort = false;
    state.busy = false;
    updateButtons();
    setConnection("busy", "连续推进中");
    while (state.running && !state.runAbort) {
      const snap = state.snapshot;
      if (!snap || snap.next_frame >= snap.frame_count) break;
      const chunk = Math.max(1, Number(els.chunkSize.value) || 40);
      const target = Math.min(snap.frame_count, snap.next_frame + chunk);
      // Advance without the modal busy overlay; the status pill shows progress.
      const ok = await advanceTo(target, { silentBusy: true });
      if (!ok) break;
      els.peekFrame.value = String(state.snapshot.next_frame);
      els.peekFrameOut.textContent =
        `${state.snapshot.next_frame} / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
      await new Promise((resolve) => setTimeout(resolve, 60));
    }
    state.running = false;
    updateButtons();
    updateStatusFromSnapshot();
  }

  // -------------------------------------------------------- speaker attribution

  function attrFmt(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    if (!Number.isFinite(value)) return "-∞";
    return Number(value).toFixed(digits);
  }

  function attrCurrentRange() {
    const start = Number(els.attrStart.value);
    const end = Number(els.attrEnd.value);
    return { start, end, count: end - start + 1 };
  }

  // A signature identifies the exact inputs a result was computed for.  Any
  // edit changes it and immediately marks the displayed result stale.
  function attrSignature() {
    const snap = state.snapshot;
    const { start, end } = attrCurrentRange();
    const reference = els.attrReference.value;
    const rows = attribution.rows.map((r) => `${r.aInput.value}|${r.bInput.value}`).join(";");
    return JSON.stringify({
      start,
      end,
      reference,
      rows,
      revision: snap ? snap.revision : -1,
    });
  }

  function showAttrError(message) {
    if (!message) { els.attrError.hidden = true; els.attrError.textContent = ""; return; }
    els.attrError.hidden = false;
    els.attrError.textContent = message;
  }

  function rebuildSpeakerTable(preferOld = true) {
    const { start, end, count } = attrCurrentRange();
    const oldRows = attribution.rows;
    els.attrSpeakerBody.innerHTML = "";
    attribution.rows = [];

    const validRange =
      Number.isInteger(start) && Number.isInteger(end) &&
      start >= 0 && end >= start && count >= 1 && count <= 64;
    els.attrTableEmpty.hidden = validRange;
    if (!validRange) {
      els.attrTableEmpty.textContent = "帧区间非法：需为 1～64 个连续帧，且不超过已推进帧数。";
      updateAttrRowHint();
      return;
    }
    els.attrTableEmpty.hidden = true;

    for (let i = 0; i < count; i += 1) {
      const frame = start + i;
      const tr = document.createElement("tr");
      const frameTd = document.createElement("td");
      frameTd.className = "mono";
      frameTd.textContent = String(frame);

      const mkCell = (which, old) => {
        const td = document.createElement("td");
        td.className = "num";
        const input = document.createElement("input");
        input.type = "number";
        input.min = "0";
        input.max = "1";
        input.step = "0.01";
        const previous = preferOld && oldRows[i] ? oldRows[i][which] : "";
        input.value = previous;
        input.dataset.frame = String(frame);
        input.addEventListener("input", () => {
          updateAttrRowHint();
          markAttributionStale();
        });
        td.appendChild(input);
        return { td, input };
      };
      const a = mkCell("aInput", true);
      const b = mkCell("bInput", true);

      const sumTd = document.createElement("td");
      sumTd.className = "num rowsum mono";
      tr.append(frameTd, a.td, b.td, sumTd);
      els.attrSpeakerBody.appendChild(tr);
      attribution.rows.push({ frame, aInput: a.input, bInput: b.input, sumTd, tr });
    }
    updateAttrRowHint();
  }

  function readRow(row) {
    const a = Number(row.aInput.value);
    const b = Number(row.bInput.value);
    return { a, b };
  }

  function updateAttrRowHint() {
    let invalid = 0;
    for (const row of attribution.rows) {
      const { a, b } = readRow(row);
      const aOk = Number.isFinite(a) && a >= 0 && a <= 1;
      const bOk = Number.isFinite(b) && b >= 0 && b <= 1;
      const sum = aOk && bOk ? a + b : NaN;
      const sumOk = Number.isFinite(sum) && Math.abs(sum - 1) <= 1e-6;
      row.sumTd.dataset.ok = String(sumOk);
      row.sumTd.textContent = Number.isFinite(sum) ? sum.toFixed(3) : "—";
      row.tr.classList.toggle("row-invalid", !(aOk && bOk && sumOk));
      if (!(aOk && bOk && sumOk)) invalid += 1;
    }
    const n = attribution.rows.length;
    els.attrRowHint.textContent = n
      ? (invalid ? `有 ${invalid} 行概率无效或行和不为 1，已保留上次结果。` : `共 ${n} 行，每行甲+乙=1。`)
      : "";
    els.attrRun.disabled = !state.sessionId || n === 0 || invalid > 0;
    markAttributionStale();
  }

  // Mark the displayed result stale relative to the current inputs / revision.
  function markAttributionStale() {
    if (!attribution.lastResult) return;
    const current = attrSignature();
    const stale = current !== attribution.lastSignature;
    els.attrStale.hidden = !stale;
    if (stale) {
      els.attrStaleInputs.textContent = `帧 ${attribution.lastResult.start_frame}–${attribution.lastResult.end_frame}、字幕「${attribution.lastResult.reference}」`;
      els.attrStaleRevision.textContent = `r${attribution.lastResult.revision}`;
    }
  }

  function renderAttributionResult(payload) {
    attribution.lastResult = payload;
    attribution.lastSignature = attrSignature();
    els.attrResult.hidden = false;
    els.attrStale.hidden = true;
    showAttrError("");

    const total = payload.log_total === null ? "—" : attrFmt(payload.log_total, 4);
    const acoustic = payload.log_acoustic_total === null ? "—" : attrFmt(payload.log_acoustic_total, 4);
    if (!payload.reachable) {
      els.attrMetaText.textContent =
        `字幕「${payload.reference}」在所选帧中不可达（路径数 0）。已保留判定所需的输入回显，请修改字幕或帧区间。`;
    } else {
      els.attrMetaText.textContent =
        `字幕「${payload.reference}」· 帧 ${payload.start_frame}–${payload.end_frame} · ` +
        `折叠路径 ${payload.path_count} 条 · 加权对数概率合计 ${total} · 纯声学对数概率 ${acoustic} · ` +
        `基于会话版本 r${payload.revision}`;
    }

    els.attrResultBody.innerHTML = "";
    for (const ch of payload.characters) {
      const tr = document.createElement("tr");
      if (!ch.reachable) tr.className = "unreachable";

      const charTd = document.createElement("td");
      charTd.className = "mono";
      charTd.textContent = ch.char;

      const spTd = document.createElement("td");
      const badge = document.createElement("span");
      badge.textContent = ch.speaker;
      badge.className = ch.tied ? "speaker-tied" : (ch.speaker === "甲" ? "speaker-a" : "speaker-b");
      spTd.appendChild(badge);
      if (ch.tied && ch.reachable) {
        const note = document.createElement("span");
        note.className = "hint";
        note.textContent = "（同分未定）";
        spTd.append(document.createTextNode(" "), note);
      }

      const posterior = (value) => {
        const td = document.createElement("td");
        td.className = "num mono";
        const bar = document.createElement("span");
        bar.className = "posterior-bar";
        const fill = document.createElement("i");
        fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
        bar.appendChild(fill);
        td.append(bar, document.createTextNode((value * 100).toFixed(1) + "%"));
        return td;
      };
      const aTd = posterior(ch.posterior_a);
      const bTd = posterior(ch.posterior_b);

      const spanTd = document.createElement("td");
      spanTd.className = "mono hint";
      spanTd.textContent = ch.reachable ? `${ch.first_frame}–${ch.last_frame}` : "无证据";

      tr.append(charTd, spTd, aTd, bTd, spanTd);
      els.attrResultBody.appendChild(tr);
    }

    renderEvidence(payload);
  }

  function renderEvidence(payload) {
    const cols = payload.characters.length;
    const headRow = document.createElement("tr");
    const head = (text) => {
      const th = document.createElement("th");
      th.textContent = text;
      return th;
    };
    headRow.appendChild(head("帧"));
    headRow.appendChild(head("blank"));
    for (const ch of payload.characters) {
      headRow.appendChild(head(`${ch.char}·甲`));
      headRow.appendChild(head(`${ch.char}·乙`));
    }
    els.attrEvidenceHead.innerHTML = "";
    els.attrEvidenceHead.appendChild(headRow);

    els.attrEvidenceBody.innerHTML = "";
    payload.frame_evidence.forEach((ev) => {
      const tr = document.createElement("tr");
      const frameTd = document.createElement("td");
      frameTd.className = "mono";
      frameTd.textContent = String(ev.frame);
      const blankTd = document.createElement("td");
      blankTd.className = "num mono";
      blankTd.textContent = ev.blank_prob.toFixed(3);
      tr.append(frameTd, blankTd);
      for (let j = 0; j < cols; j += 1) {
        const a = ev.label_a[j] || 0;
        const b = ev.label_b[j] || 0;
        const aTd = document.createElement("td");
        aTd.className = "num mono " + (a > 0 ? "cell-a" : "zero");
        aTd.textContent = a > 0 ? a.toFixed(3) : "·";
        const bTd = document.createElement("td");
        bTd.className = "num mono " + (b > 0 ? "cell-b" : "zero");
        bTd.textContent = b > 0 ? b.toFixed(3) : "·";
        tr.append(aTd, bTd);
      }
      els.attrEvidenceBody.appendChild(tr);
    });
  }

  async function runAttribution() {
    if (!state.sessionId || !state.snapshot) return;
    const { start, end, count } = attrCurrentRange();
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start) {
      showAttrError("帧区间非法。");
      return;
    }
    if (count < 1 || count > 64) {
      showAttrError("帧区间必须是连续 1～64 帧。");
      return;
    }
    if (end >= state.snapshot.next_frame) {
      showAttrError("结束帧超出已推进范围，请先推进到该帧之后再分析。");
      return;
    }
    const reference = els.attrReference.value.trim();
    if (!reference) {
      showAttrError("参考字幕不能为空。");
      return;
    }
    if (reference.length > 16) {
      showAttrError("参考字幕至多 16 个字。");
      return;
    }
    const speakerProbs = [];
    for (const row of attribution.rows) {
      const { a, b } = readRow(row);
      if (!Number.isFinite(a) || !Number.isFinite(b) || a < 0 || a > 1 || b < 0 || b > 1
          || Math.abs(a + b - 1) > 1e-6) {
        showAttrError("存在无效的说话人概率或行和不为 1，已保留上次结果。");
        return;
      }
      speakerProbs.push([a, b]);
    }
    if (speakerProbs.length !== count) {
      showAttrError("说话人概率行数与所选帧数不一致，请重建表格。");
      return;
    }

    const seq = ++attribution.requestSeq;
    const analysisId = `attr-${Date.now()}-${seq}`;
    const signature = attrSignature();
    attribution.inFlight = true;
    els.attrRun.disabled = true;
    showAttrError("");
    try {
      const payload = await api("/api/attribution", {
        method: "POST",
        body: JSON.stringify({
          session_id: state.sessionId,
          start_frame: start,
          end_frame: end,
          reference,
          speaker_probs: speakerProbs,
          analysis_id: analysisId,
        }),
      });
      // A newer request already superseded this one: never overwrite.
      if (seq !== attribution.requestSeq) return;
      renderAttributionResult(payload);
      // If the session advanced while the request was in flight, the freshly
      // rendered result is already stale relative to the new revision.
      if (attrSignature() !== signature || payload.revision !== state.snapshot.revision) {
        markAttributionStale();
      }
    } catch (error) {
      if (seq !== attribution.requestSeq) return;
      // Validation / unreachable errors keep the previous result on screen and
      // only surface a message; they must not overwrite any older conclusion.
      showAttrError(`归属分析未更新：${error.message}（已保留上次结果）`);
      markAttributionStale();
    } finally {
      attribution.inFlight = false;
      updateAttrRowHint();
    }
  }

  function syncAttributionBounds() {
    const snap = state.snapshot;
    const maxFrame = snap ? Math.max(0, snap.next_frame - 1) : 0;
    els.attrStart.max = String(maxFrame);
    els.attrEnd.max = String(maxFrame);
    if (Number(els.attrStart.value) > maxFrame) els.attrStart.value = String(maxFrame);
    if (Number(els.attrEnd.value) > maxFrame) els.attrEnd.value = String(maxFrame);
    els.attrStartOut.textContent = els.attrStart.value;
    els.attrEndOut.textContent = els.attrEnd.value;
  }

  function fillSampleSpeakers() {
    const sp = attribution.sampleSpeakers;
    if (!sp || !sp.length) return;
    const { start } = attrCurrentRange();
    for (const row of attribution.rows) {
      const source = sp[row.frame - start];
      if (!source) continue;
      row.aInput.value = String(source[0]);
      row.bInput.value = String(source[1]);
    }
    updateAttrRowHint();
  }

  // ------------------------------------------------------------------- wiring

  els.beamWidth.addEventListener("input", () => {
    els.beamOut.textContent = els.beamWidth.value;
  });
  els.chunkSize.addEventListener("input", () => {
    els.chunkOut.textContent = els.chunkSize.value;
  });

  els.loadBtn.addEventListener("click", () => {
    const vocabFile = els.vocabFile.files[0];
    const framesFile = els.framesFile.files[0];
    if (!vocabFile || !framesFile) {
      els.setupError.hidden = false;
      els.setupError.textContent = "请同时选择词表文件和逐帧概率文件";
      return;
    }
    const form = new FormData();
    form.append("vocab_file", vocabFile);
    form.append("frames_file", framesFile);
    form.append("beam_width", els.beamWidth.value);
    createSession(form);
  });

  els.resetBtn.addEventListener("click", async () => {
    if (state.sessionId) {
      try {
        await api(`/api/sessions/${state.sessionId}`, { method: "DELETE" });
      } catch { /* ignore */ }
    }
    state.sessionId = null;
    state.revision = -1;
    state.snapshot = null;
    els.workspace.hidden = true;
    state.loadedSample = null;
    setConnection("idle", "未加载");
    updateButtons();
  });

  els.confirmBest.addEventListener("click", () => {
    const snap = state.snapshot;
    if (!snap) return;
    const best = snap.best_text || "";
    let target = best;
    if (!best || best === snap.prefix) {
      showError("当前最佳候选与已确认前缀相同，无法继续确认");
      return;
    }
    // confirm the full best hypothesis; users can also type a shorter prefix
    confirmPrefix(target);
  });

  els.applyPrefix.addEventListener("click", () => {
    confirmPrefix(els.prefixInput.value.trim());
  });

  els.prefixInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      confirmPrefix(els.prefixInput.value.trim());
    }
  });

  els.undoBtn.addEventListener("click", undo);

  els.stepBtn.addEventListener("click", async () => {
    const snap = state.snapshot;
    if (!snap) return;
    const chunk = Math.max(1, Number(els.chunkSize.value) || 40);
    const target = Math.min(snap.frame_count, snap.next_frame + chunk);
    const ok = await advanceTo(target);
    if (ok) {
      els.peekFrame.value = String(state.snapshot.next_frame);
      els.peekFrameOut.textContent = `${els.peekFrame.value} / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
      renderFrameProbs(state.snapshot.frame_info);
    }
  });

  els.runBtn.addEventListener("click", runContinuously);
  els.stopBtn.addEventListener("click", () => {
    state.runAbort = true;
    state.running = false;
    updateButtons();
    updateStatusFromSnapshot();
  });

  els.restartBtn.addEventListener("click", async () => {
    // Reset by opening a fresh server session from the same sample or files.
    state.runAbort = true;
    if (state.loadedSample) {
      await loadSample(state.loadedSample);
      return;
    }
    if (!els.vocabFile.files[0] || !els.framesFile.files[0]) {
      showError("要回到起点请重新选择文件或重新载入样例");
      return;
    }
    const form = new FormData();
    form.append("vocab_file", els.vocabFile.files[0]);
    form.append("frames_file", els.framesFile.files[0]);
    form.append("beam_width", els.beamWidth.value);
    await createSession(form);
  });

  els.peekFrame.addEventListener("input", () => {
    const frame = Number(els.peekFrame.value);
    els.peekFrameOut.textContent = `${frame} / ${els.peekFrame.max}`;
    peek(frame);
  });

  // Speaker attribution wiring.
  els.attrRebuild.addEventListener("click", () => {
    rebuildSpeakerTable(true);
    showAttrError("");
  });
  els.attrRun.addEventListener("click", runAttribution);
  els.attrFillSample.addEventListener("click", fillSampleSpeakers);
  els.attrStart.addEventListener("input", () => {
    els.attrStartOut.textContent = els.attrStart.value;
    markAttributionStale();
  });
  els.attrEnd.addEventListener("input", () => {
    els.attrEndOut.textContent = els.attrEnd.value;
    markAttributionStale();
  });
  els.attrReference.addEventListener("input", () => {
    markAttributionStale();
  });
  els.attrReference.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      runAttribution();
    }
  });

  async function loadSamples() {
    try {
      const payload = await api("/api/samples");
      state.samples = payload.samples || [];
      els.sampleButtons.innerHTML = "";
      if (!state.samples.length) {
        const span = document.createElement("span");
        span.className = "hint";
        span.textContent = "未找到样例文件";
        els.sampleButtons.appendChild(span);
        return;
      }
      state.samples.forEach((sample) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sample ghost";
        btn.textContent = sample.name;
        btn.addEventListener("click", () => loadSample(sample.name));
        els.sampleButtons.appendChild(btn);
      });
    } catch { /* samples are optional */ }
  }

  async function loadSample(name) {
    els.setupError.hidden = true;
    setBusy(true, "载入样例…");
    try {
      const sample = await api(`/api/samples/${encodeURIComponent(name)}`);
      const payload = await api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          vocab: sample.vocab,
          beam_width: Number(els.beamWidth.value),
          frames: sample.frames,
        }),
      });
      startSession(payload, name);
      // Samples may ship a matching per-frame 甲/乙 table for the attribution
      // panel.  It stays on the client and only fills the editable rows.
      if (Array.isArray(sample.speakers) && sample.speakers.length === sample.frames.length) {
        attribution.sampleSpeakers = sample.speakers.map((row) => [Number(row[0]), Number(row[1])]);
        els.attrFillSample.hidden = false;
      } else {
        attribution.sampleSpeakers = null;
        els.attrFillSample.hidden = true;
      }
      showError("");
      hideNotice();
      showNotice(`已载入样例「${name}」，共 ${payload.frame_count} 帧。`);
    } catch (error) {
      els.setupError.hidden = false;
      els.setupError.textContent = error.message;
      setConnection("error", "加载失败");
    } finally {
      setBusy(false);
    }
  }

  loadSamples();
  updateButtons();
  updateStatusFromSnapshot();
})();
