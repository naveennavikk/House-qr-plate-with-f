const form = document.querySelector("#fetch-form");
const fromInput = document.querySelector("#from-day");
const toInput = document.querySelector("#to-day");
const ulbSelect = document.querySelector("#ulb");
const ULB_KEY = "qr-check-ulb";
const submitButton = document.querySelector("#submit");
const stopButton = document.querySelector("#stop");
const results = document.querySelector("#results");
const progress = document.querySelector("#progress");
const summary = document.querySelector("#summary");
const filters = document.querySelector("#filters");
const template = document.querySelector("#house-template");

const KIND = { Accepted: "approved", Rejected: "rejected", "Manual review": "manual", Error: "failed" };
const COUNTER = { Accepted: "#n-accepted", Rejected: "#n-rejected", "Manual review": "#n-manual", Error: "#n-error" };
const timer = document.querySelector("#timer");
const elapsedLabel = document.querySelector("#elapsed");
const batchTimes = document.querySelector("#batch-times");
let controller = null;
let activeFilter = "all";
let ticker = null;

function startTimer() {
  const started = performance.now();
  batchTimes.replaceChildren();
  timer.classList.remove("hidden");
  elapsedLabel.textContent = "0.0 s";
  ticker = setInterval(() => {
    elapsedLabel.textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`;
  }, 100);
}

function stopTimer(finalSeconds) {
  clearInterval(ticker);
  ticker = null;
  if (finalSeconds !== undefined) elapsedLabel.textContent = `${finalSeconds} s`;
}

function addBatchTime(message) {
  const item = document.createElement("span");
  item.textContent = `Batch ${message.batch} (${message.count} houses): ${message.seconds} s`;
  batchTimes.append(item);
}

fromInput.value = toInput.value = new Date().toISOString().slice(0, 10);

async function loadUlbs() {
  try {
    const response = await fetch("/api/ulbs");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    let saved = null;
    try { saved = localStorage.getItem(ULB_KEY); } catch {}
    const placeholder = new Option("Select a ULB…", "");
    placeholder.disabled = true;
    ulbSelect.replaceChildren(placeholder, ...payload.map((u) => new Option(`${u.name} (${u.appId})`, u.appId)));
    ulbSelect.value = payload.some((u) => String(u.appId) === saved) ? saved : "";
  } catch (error) {
    ulbSelect.replaceChildren(new Option("Could not load ULBs", ""));
    showError(`Could not load the ULB list: ${error.message}`);
  }
}

ulbSelect.addEventListener("change", () => {
  try { localStorage.setItem(ULB_KEY, ulbSelect.value); } catch {}
});
loadUlbs();

function bump(selector) {
  const el = document.querySelector(selector);
  el.textContent = Number(el.textContent) + 1;
}

function resetSummary(total) {
  for (const sel of ["#n-done", ...Object.values(COUNTER)]) document.querySelector(sel).textContent = "0";
  document.querySelector("#n-total").textContent = total;
  summary.classList.remove("hidden");
  filters.classList.remove("hidden");
}

function setImage(img, src) {
  if (src) img.src = src;
  else img.closest("figure").classList.add("empty");
}

function addHouse(house) {
  const card = template.content.firstElementChild.cloneNode(true);
  const status = house.status || "Error";
  card.classList.add(KIND[status] || "failed");
  card.dataset.status = status;
  card.style.order = house.index;
  setImage(card.querySelector(".qr-thumb"), house.qrThumb);
  setImage(card.querySelector(".house-thumb"), house.houseThumb);
  card.querySelector(".house-id").textContent = `House ID ${house.houseId ?? "—"}`;
  card.querySelector(".house-no").textContent = house.houseNo || "(no house number)";
  card.querySelector(".badge").textContent = status;
  const meta = [`User ${house.userId ?? "—"}`, `Batch ${house.batch ?? 1}`];
  if (house.propertyType) meta.push(house.propertyType);
  if (house.employee) meta.push(house.employee);
  if (house.seconds !== undefined) meta.push(`${house.seconds}s`);
  card.querySelector(".meta").textContent = meta.join(" · ");
  card.querySelector(".remark").textContent =
    house.Remark || (status === "Accepted" ? "Plate matched and inspection passed." : "");
  card.hidden = activeFilter !== "all" && activeFilter !== status;
  results.append(card);
  bump("#n-done");
  if (COUNTER[status]) bump(COUNTER[status]);
}

function showError(message) {
  const card = document.createElement("article");
  card.className = "house-card failed message";
  card.textContent = message;
  results.append(card);
}

filters.addEventListener("click", (event) => {
  const button = event.target.closest(".filter");
  if (!button) return;
  activeFilter = button.dataset.filter;
  filters.querySelectorAll(".filter").forEach((b) => b.classList.toggle("active", b === button));
  results.querySelectorAll(".house-card[data-status]").forEach((card) => {
    card.hidden = activeFilter !== "all" && activeFilter !== card.dataset.status;
  });
});

stopButton.addEventListener("click", () => controller?.abort());

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  controller = new AbortController();
  submitButton.disabled = true;
  submitButton.textContent = "Checking…";
  stopButton.classList.remove("hidden");
  results.replaceChildren();
  summary.classList.add("hidden");
  const ulbName = ulbSelect.selectedOptions[0]?.textContent || "";
  progress.textContent = `Fetching houses for ${ulbName} from the house details API…`;
  startTimer();
  try {
    const response = await fetch(`/api/houses/check?appId=${encodeURIComponent(ulbSelect.value)}&fromDate=${encodeURIComponent(fromInput.value)}&toDate=${encodeURIComponent(toInput.value)}`, {
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Request failed (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        const message = JSON.parse(line);
        if (message.type === "start") {
          resetSummary(message.count);
          progress.textContent = message.count
            ? `${ulbName}: checking ${message.count} house${message.count === 1 ? "" : "s"} in ${message.batches} batch${message.batches === 1 ? "" : "es"} of up to ${message.batchSize}…`
            : `${ulbName}: no houses found for these dates.`;
        } else if (message.type === "house") {
          addHouse(message);
        } else if (message.type === "batch") {
          addBatchTime(message);
        } else if (message.type === "done") {
          stopTimer(message.seconds);
          progress.textContent = "All houses checked.";
        }
      }
    }
  } catch (error) {
    if (error.name === "AbortError") progress.textContent = "Stopped.";
    else {
      progress.textContent = "";
      showError(`Could not check houses: ${error.message}`);
    }
  } finally {
    stopTimer();
    submitButton.disabled = false;
    submitButton.textContent = "Fetch & check";
    stopButton.classList.add("hidden");
    controller = null;
  }
});
