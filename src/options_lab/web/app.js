"use strict";
const byId = (id) => document.getElementById(id);
const colors = ["#55e0c2", "#b1a5fe", "#ffc485", "#f294b6"];
const seriesNames = ["Black–Scholes", "Price-only network", "Price + delta network", "Constrained mixture"];
const modelNames = seriesNames.slice(1);
const greekFields = {
  delta: ["Delta", "", "Change in premium per $1 increase in spot."],
  gamma: ["Gamma", "Δ / $", "Change in delta per $1 increase in spot."],
  vega: ["Vega", "$ / vol pt", "Change in premium per one percentage point of volatility."],
  theta: ["Theta", "$ / day", "Local change in premium per calendar day elapsed, holding spot and other market inputs fixed."],
  rho: ["Rho", "$ / rate pt", "Change in premium per one percentage point of interest rate."],
  vanna: ["Vanna", "Δ / vol pt", "Change in delta per one percentage point of volatility."],
  vomma: ["Vomma", "$ / vol pt²", "Change in vega per one percentage point of volatility."]
};
const greekLabel = key => {
  const [name, unit] = greekFields[key];
  return unit ? name + " (" + unit + ")" : name;
};
const number = (n, digits = 3) => new Intl.NumberFormat("en-US", {maximumFractionDigits: digits}).format(n);
const dollars = (n) => new Intl.NumberFormat("en-US", {style: "currency", currency: "USD"}).format(n);
const priceError = n => n > 0 && n < .01 ? "$" + number(n, 8) : dollars(n);
const formatError = value => value > 0 && value < .0001 ? value.toExponential(3) : number(value, 6);
const chartDrawers = new Map();

for (const button of document.querySelectorAll("[data-page]")) {
  button.addEventListener("click", () => {
    for (const item of document.querySelectorAll("[data-page]")) {
      if (item === button) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
      byId(item.dataset.page).hidden = item !== button;
    }
    for (const draw of chartDrawers.values()) draw();
  });
}

function toTable(container, fields, rows) {
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const field of fields) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = field[1];
    head.append(th);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const [key, , format] of fields) {
      const cell = tr.insertCell();
      const value = row[key];
      cell.textContent = value == null ? "—" : format ? format(value) : value;
      if (typeof value === "number") cell.dataset.number = "";
    }
  }
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.append(table);
  byId(container).replaceChildren(wrap);
}

function renderCards(id, cards) {
  byId(id).replaceChildren(...cards.map(([label, value, note]) => {
    const card = document.createElement("div");
    card.className = "card";
    for (const [type, text] of [["label", label], ["value", value], ["note", note]]) {
      const line = document.createElement("div");
      line.className = "card-" + type;
      line.textContent = text;
      card.append(line);
    }
    return card;
  }));
}

function chart(id, values, xLabel, yLabel, histogram = false) {
  const canvas = byId(id);
  let hover = null;
  function draw() {
    const bounds = canvas.getBoundingClientRect();
    const w = bounds.width, h = bounds.height;
    if (!w || !h) return;
    const scale = window.devicePixelRatio || 1;
    canvas.width = w * scale;
    canvas.height = h * scale;
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    const left = 60, right = w - 16, top = 13, bottom = h - 45;
    const xScale = values.logX ? Math.log2 : x => x;
    const xMin = xScale(values.x[0]), xMax = xScale(values.x.at(-1));
    const yValues = values.ys.flat().filter(Number.isFinite);
    const baseline = Math.min(0, ...yValues);
    const [yMin, yMax] = values.yRange ?? [baseline, Math.max(baseline + 1e-12, ...yValues)];
    const ySpan = (yMax - yMin) * 1.06;
    const px = x => xMin === xMax ? (left + right) / 2 :
      left + (xScale(x) - xMin) / (xMax - xMin) * (right - left);
    const py = y => bottom - (y - yMin) / ySpan * (bottom - top);
    ctx.font = "11px Consolas, monospace";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const frac = i / 4;
      const y = bottom - frac * (bottom - top);
      ctx.strokeStyle = "#273645";
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
      ctx.fillStyle = "#a0aebc"; ctx.textAlign = "right";
      const tick = yMin + frac * ySpan;
      ctx.fillText(tick !== 0 && Math.abs(tick) < .001 ? tick.toExponential(1) : number(tick, 3), left - 7, y + 4);
      ctx.textAlign = "center";
      if (!values.logX && !values.discreteX) {
        ctx.fillText(number(xMin + frac * (xMax - xMin), 1), left + frac * (right - left), bottom + 17);
      }
    }
    if (values.logX || values.discreteX) {
      const tickCount = Math.min(5, values.x.length);
      for (let i = 0; i < tickCount; i++) {
        const tick = values.x[Math.round(i * (values.x.length - 1) / (tickCount - 1 || 1))];
        const x = px(tick);
        ctx.fillText(number(tick), x, bottom + 17);
      }
    }
    ctx.fillText(xLabel, (left + right) / 2, h - 8);
    ctx.save();
    ctx.translate(12, (top + bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    values.ys.forEach((ys, series) => {
      ctx.strokeStyle = colors[series]; ctx.lineWidth = 2;
      if (histogram) {
        ctx.fillStyle = colors[series] + "25";
        for (let i = 0; i < ys.length; i++) {
          const x = px(values.x[i]), y = py(ys[i]), width = px(values.x[i + 1]) - x;
          ctx.fillRect(x + 1, y, width - 2, bottom - y);
          if (ys[i] > 0) ctx.strokeRect(x + 1, y, width - 2, bottom - y);
        }
      } else {
        if (series) ctx.setLineDash([[5, 3], [2, 3], [9, 3, 2, 3]][series - 1]);
        ctx.beginPath();
        let connected = false;
        ys.forEach((y, i) => {
          if (y === null) {
            connected = false;
          } else {
            if (connected) ctx.lineTo(px(values.x[i]), py(y));
            else ctx.moveTo(px(values.x[i]), py(y));
            connected = true;
          }
        });
        ctx.stroke(); ctx.setLineDash([]);
        if (values.logX || values.points) {
          ctx.fillStyle = colors[series];
          ys.forEach((y, i) => {
            if (y !== null) ctx.fillRect(px(values.x[i]) - 2, py(y) - 2, 4, 4);
          });
        }
      }
    });
    if (hover !== null && !histogram && hover >= left && hover <= right) {
      const i = values.x.reduce((closest, x, index) =>
        Math.abs(px(x) - hover) < Math.abs(px(values.x[closest]) - hover) ? index : closest, 0);
      const x = px(values.x[i]);
      ctx.strokeStyle = "#60717e"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
      const labels = [xLabel + " " + number(values.x[i], 2), ...values.ys.map((ys, j) =>
        (values.labels?.[j] || seriesNames[j]) + "  " + (ys[i] === null ? "—" : number(ys[i], 6)))];
      const width = Math.max(...labels.map(label => ctx.measureText(label).width)) + 18;
      const tipX = Math.max(0, Math.min(x + 10, w - width));
      ctx.fillStyle = "#0b131c"; ctx.fillRect(tipX, top, width, 14 + labels.length * 16);
      ctx.textAlign = "left";
      labels.forEach((label, j) => {
        ctx.fillStyle = j ? colors[j - 1] : "#edf3f8";
        ctx.fillText(label, tipX + 9, top + 16 + j * 16);
      });
    }
  }
  canvas.onpointermove = event => { hover = event.offsetX; draw(); };
  canvas.onpointerleave = () => { hover = null; draw(); };
  chartDrawers.set(id, draw);
  draw();
}
window.addEventListener("resize", () => { for (const draw of chartDrawers.values()) draw(); });

function downloadButton(id, filename, data, mime) {
  const button = byId(id);
  button.disabled = false;
  button.onclick = () => {
    const url = URL.createObjectURL(new Blob([data], {type: mime}));
    const link = document.createElement("a");
    link.href = url; link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
}

function formPayload(fields, textFields = ["kind"]) {
  const body = {};
  for (const [key, value] of fields) {
    if (typeof value === "string" && value !== "") body[key] = textFields.includes(key) ? value : Number(value);
  }
  return body;
}

function connect(page, endpoint, loadingText, render, readFields = formPayload) {
  const form = byId(page + "-form");
  const status = byId(page + "-status");
  const submit = form.querySelector("button[type=submit]");
  const download = byId(page + "-export");
  const changed = "Inputs changed. Results still reflect the previous run.";
  let inputRevision = 0;
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const revision = inputRevision;
    const fields = new FormData(form);
    submit.disabled = true;
    download.disabled = true;
    status.textContent = loadingText;
    status.dataset.error = "false";
    try {
      const body = await readFields(fields);
      const response = await fetch("/api/" + endpoint, {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Experiment failed.");
      const message = render(data);
      const stale = revision !== inputRevision;
      status.textContent = stale ? changed : message || "";
      if (stale) download.disabled = true;
    } catch (error) {
      status.textContent = error.message + " Any visible results are from the previous successful run.";
      status.dataset.error = "true";
    } finally { submit.disabled = false; }
  });
  form.addEventListener("input", () => {
    inputRevision++;
    if (download.disabled) return;
    status.textContent = changed;
    status.dataset.error = "false";
    download.disabled = true;
  });
}

connect("pricing", "price", "Calculating prices and Greeks. The first run fits all three models.", data => {
  const [reference, ...networks] = data.metrics;
  const cards = [[reference.method, dollars(reference.price), "Analytic price"],
    ...networks.map(model => [model.method, dollars(model.price),
      priceError(Math.abs(model.price - reference.price)) + " absolute error"])];
  if (data.implied_vol_percent !== null) {
    cards.push(["Implied volatility", number(data.implied_vol_percent, 2) + "%", "From observed premium"]);
  }
  renderCards("price-metrics", cards);
  const greekKeys = Object.keys(greekFields);
  const formatGreek = v => v !== 0 && Math.abs(v) < .00001 ? v.toExponential(3) : number(v, 6);
  toTable("greeks", [["method", "Model"], ...greekKeys.map(key => [key, greekLabel(key), formatGreek])], data.metrics);
  const x = data.curve.spot;
  const ys = seriesNames.map(label => data.curve[label]);
  chart("pricing-chart", {x, ys}, "Spot ($)", "Option price ($)");
  const greekControl = byId("greek-statistic");
  const renderGreek = () => {
    const key = greekControl.value;
    byId("greek-definition").textContent = greekFields[key][2];
    chart("greek-chart", {x, ys: seriesNames.map(label => data.greek_curves[label][key])}, "Spot ($)", greekLabel(key));
  };
  greekControl.onchange = renderGreek;
  renderGreek();

  const modelKeys = ["black_scholes", "price_only", "price_delta", "mixture"];
  const columns = modelKeys.map((model, i) => [model + "_price", ys[i]]);
  for (const greek of greekKeys) {
    seriesNames.forEach((label, i) => columns.push([modelKeys[i] + "_" + greek, data.greek_curves[label][greek]]));
  }
  downloadButton("pricing-export", "option-price-sweep.csv",
    ["spot", ...columns.map(([name]) => name)].join(",") + "\n" +
    x.map((spot, i) => [spot, ...columns.map(([, values]) => values[i])].join(",")).join("\n"), "text/csv");
  if (!data.in_domain) return "The selected contract is outside the neural training range. The neural result is an extrapolation.";
  if (!data.sweep_in_domain) return "Part of the spot sweep extends beyond the neural training range.";
});

connect("evaluation", "evaluate", "Training and evaluating on independent synthetic contracts…", data => {
  const [baseline, tuned, mixture] = modelNames.map(label => data.models[label]);
  const count = key => [tuned, mixture].map(model => number(model[key])).join(" → ");
  renderCards("eval-metrics", [["Held-out contracts", number(baseline.samples), "Common sample per option type"],
    ["Price-bound violations", count("price_bound_violations"), "Price + delta → Mixture"],
    ["Negative gamma", count("negative_gamma_rows"), "Price + delta → Mixture"],
    ["Negative vega", count("negative_vega_rows"), "Price + delta → Mixture"]]);
  const statControl = byId("error-statistic");
  const metricsControl = byId("evaluation-metrics");
  const renderErrors = () => {
    const stat = statControl.value;
    const group = metricsControl.value;
    const rows = baseline[group].map((row, i) => ({...row,
      metric: row.metric === "price" ? "Price ($)" : greekLabel(row.metric), price_only: row[stat],
      tuned: tuned[group][i][stat], mixture: mixture[group][i][stat]}));
    toTable("evaluation-table", [["option", "Option"], ["slice", "Slice"], ["metric", "Metric"], ["samples", "Count", number],
      ["price_only", "Price-only", formatError], ["tuned", "Price + delta", formatError],
      ["mixture", "Constrained mixture", formatError]], rows);
  };
  statControl.onchange = renderErrors;
  metricsControl.onchange = renderErrors;
  renderErrors();
  const countChecked = (row, key) => number(row[key + "_violations"]) + " / " + number(row[key + "_checks"]);
  const shapeRows = modelNames.map(method => {
    const checks = data.models[method].grid_checks;
    return {method, slope: countChecked(checks, "strike_slope"), convexity: countChecked(checks, "strike_convexity"),
      maturity: countChecked(checks, "forward_maturity")};
  });
  toTable("shape-checks", [["method", "Model"], ["slope", "Strike slope"],
    ["convexity", "Strike convexity"], ["maturity", "Forward maturity"]], shapeRows);
  toTable("timing", [["method", "Model"], ["batch", "Batch size", number],
    ["median_microseconds", "Median µs", v => number(v, 1)], ["repeats", "Repeats", number]], data.benchmark);
  const training = data.training[seriesNames[3]];
  chart("loss-chart", {x: training.loss.map((_, i) => i + 1), ys: [training.loss], labels: ["Joint loss"]},
    "Iteration", "Joint loss");
  downloadButton("evaluation-export", "options-model-evaluation.json", JSON.stringify(data, null, 2), "application/json");
});

connect("robustness", "robustness", "Fitting models and scoring each seed on the same contracts…", data => {
  byId("robustness-results").hidden = false;
  const regimeControl = byId("robustness-regime");
  const kindControl = byId("robustness-kind");
  const metricControl = byId("robustness-metric");
  const statControl = byId("robustness-stat");
  const previousRegime = regimeControl.value;
  regimeControl.replaceChildren(...data.regimes.map(regime => {
    const option = document.createElement("option");
    option.value = regime.key;
    option.textContent = regime.label;
    return option;
  }));
  regimeControl.value = data.regimes.some(row => row.key === previousRegime) ? previousRegime : data.regimes[0].key;
  const contractCount = data.regimes.reduce((total, row) => total + row.contracts, 0);
  const outOfDomain = data.regimes.reduce((total, row) => total + row.contracts - row.in_domain, 0);
  renderCards("robustness-metrics", [
    ["Training runs", number(data.runs.length), "Across all three fitted models"],
    ["Test contracts", number(contractCount), "Per option type in every run"],
    ["Outside training range", number(outOfDomain), "Contracts per option type"]
  ]);

  const counts = ["price_bound_violations", "negative_gamma_rows", "negative_vega_rows"];
  const render = () => {
    const regime = data.regimes.find(row => row.key === regimeControl.value);
    const kind = kindControl.value;
    const metric = metricControl.value;
    const stat = statControl.value;
    const trainingRange = number(regime.in_domain) + " of " + number(regime.contracts) + " contracts within the training range";
    byId("robustness-regime-detail").textContent = trainingRange + " · Held-out seed " + number(data.test_seed);
    const seeds = data.runs.map(run => run.seed);
    const byModel = modelNames.map(method => {
      const results = data.runs.map(run => run.models[method].regimes[regime.key]);
      const errors = results.map(row => row.errors[kind][metric][stat]);
      const worst = Math.max(...errors);
      const mean = errors.reduce((total, value) => total + value, 0) / errors.length;
      const violations = Object.fromEntries(counts.map(key => [
        key, number(Math.max(...results.map(row => row[key]))) + " / " + number(regime.contracts)
      ]));
      return {method, errors, mean, worst, worstSeed: seeds[errors.indexOf(worst)], ...violations};
    });
    const metricName = metric === "price" ? "Price ($)" : greekLabel(metric);
    const statName = {mae: "MAE", rmse: "RMSE", p95_absolute_error: "P95 error"}[stat];
    chart("robustness-chart", {x: seeds, ys: byModel.map(row => row.errors),
      labels: modelNames, points: true, discreteX: true}, "Training seed", statName + " · " + metricName);
    toTable("robustness-seeds", [["seed", "Seed", number],
      ...modelNames.map(method => [method, method, formatError])],
      seeds.map((seed, i) => ({seed, ...Object.fromEntries(byModel.map(row => [row.method, row.errors[i]]))})));
    toTable("robustness-summary", [["method", "Model"], ["mean", "Mean error", formatError],
      ["worst", "Worst error", formatError], ["worstSeed", "Worst seed", number],
      ["price_bound_violations", "Price bounds"], ["negative_gamma_rows", "Negative gamma"],
      ["negative_vega_rows", "Negative vega"]], byModel);
  };
  regimeControl.onchange = kindControl.onchange = metricControl.onchange = statControl.onchange = render;
  render();
  downloadButton("robustness-export", "options-robustness-report.json", JSON.stringify(data, null, 2), "application/json");
});

function pnlHistogram(id, values) {
  const flat = values.flat();
  const min = Math.min(...flat), max = Math.max(...flat);
  const span = max - min || 1;
  const bins = 35;
  const edges = Array.from({length: bins + 1}, (_, i) => min + i * span / bins);
  const ys = values.map(pnls => {
    const counts = new Array(bins).fill(0);
    for (const pnl of pnls) counts[Math.min(bins - 1, Math.floor((pnl - min) / span * bins))]++;
    return counts.map(count => count * 100 / pnls.length);
  });
  chart(id, {x: edges, ys}, "Terminal P&L ($)", "Paths (%)", true);
}

connect("hedging", "hedge", "Simulating paired price paths and rebalancing all four hedges…", data => {
  renderCards("hedge-metrics", [["Shared initial premium", dollars(data.initial_premium), "One option per path"],
    ["Hedge P&L RMSE", data.metrics.slice(2).map(row => dollars(row.rmse)).join(" → "), "Price + delta → Mixture"],
    ["Neural training range coverage", number(data.neural_domain_coverage * 100, 2) + "%", "Across all rebalance observations"]]);
  const values = seriesNames.map(name => data.pnl[name]);
  pnlHistogram("hedging-chart", values);
  toTable("hedge-table", [["method", "Model"], ["mean_pnl", "Mean P&L", dollars],
    ["rmse", "P&L RMSE", dollars], ["paired_difference_rmse", "Paired RMSE vs. B–S", priceError],
    ["p05_pnl", "5th pct. P&L", dollars], ["mean_cost", "Mean costs", dollars]], data.metrics);
  downloadButton("hedging-export", "paired-hedge-pnl.csv", "path,black_scholes_pnl,price_only_pnl,price_delta_pnl,mixture_pnl\n" +
    values[0].map((_, i) => [i + 1, ...values.map(pnls => pnls[i])].join(",")).join("\n"), "text/csv");
  if (data.neural_domain_coverage < 1) return "Some hedge rebalances used neural extrapolation. See training range coverage.";
});

connect("stress", "stress", "Running six matched hedge scenarios…", data => {
  byId("stress-results").hidden = false;
  const baseline = data.scenarios[0];
  const modelControl = byId("stress-model");
  const scenarioControl = byId("stress-scenario");
  const previousModel = modelControl.value;
  const previousScenario = scenarioControl.value;
  const stressScenarios = data.scenarios.slice(1);
  const optionFor = (value, label) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  };
  modelControl.replaceChildren(...data.model_names.map(name => optionFor(name, name)));
  scenarioControl.replaceChildren(...stressScenarios.map(scenario => optionFor(scenario.key, scenario.label)));
  modelControl.value = data.model_names.includes(previousModel) ? previousModel : data.model_names[0];
  scenarioControl.value = stressScenarios.some(item => item.key === previousScenario) ? previousScenario : "combined";

  const statsFor = (scenario, name) => scenario.metrics.find(row => row.method === name);
  const render = () => {
    const model = modelControl.value;
    const scenario = stressScenarios.find(item => item.key === scenarioControl.value);
    const baseStats = statsFor(baseline, model);
    const stressStats = statsFor(scenario, model);
    renderCards("stress-metrics", [
      ["Shared initial premium", dollars(data.initial_premium), number(data.settings.paths) + " paths per scenario"],
      ["Baseline tail mean", dollars(baseStats.worst_5pct_mean_pnl), "Worst " + number(data.tail_count) + " outcomes"],
      ["Stressed tail mean", dollars(stressStats.worst_5pct_mean_pnl), scenario.label],
      ["Mean P&L change", dollars(stressStats.mean_change_from_base), "Selected stress minus baseline"]
    ]);
    const terms = [percent(scenario.realized_vol * 100) + " path volatility",
      number(scenario.cost_bps) + " bps trading cost",
      number(scenario.hedge_dates) + " hedge dates before expiry"];
    if (scenario.jump_step !== null) {
      terms.push(percent(scenario.spot_jump * 100) + " jump at step " + number(scenario.jump_step));
    }
    byId("stress-scenario-detail").textContent = terms.join(" · ");
    byId("stress-scenario-legend").textContent = scenario.label;
    pnlHistogram("stress-chart", [baseline.pnl[model], scenario.pnl[model]]);
    toTable("stress-table", [
      ["label", "Scenario"], ["mean_pnl", "Mean P&L", dollars], ["p05_pnl", "5th pct.", dollars],
      ["worst_5pct_mean_pnl", "Tail mean", dollars], ["mean_change_from_base", "Change vs base", dollars],
      ["rmse", "P&L RMSE", dollars], ["mean_cost", "Mean cost", dollars], ["coverage", "Training range", percent]
    ], data.scenarios.map(item => ({label: item.label, coverage: item.neural_domain_coverage * 100, ...statsFor(item, model)})));
  };
  modelControl.onchange = scenarioControl.onchange = render;
  render();

  const csvHeader = "scenario,path,terminal_spot,black_scholes_pnl,price_only_pnl,price_delta_pnl,mixture_pnl," +
    "path_vol_pct,jump_pct,jump_step,cost_bps,rebalance_every,hedge_dates,spot,strike,days,kind,hedge_vol_pct,rate_pct,intervals,seed";
  const s = data.settings;
  const rows = data.scenarios.flatMap(item => item.terminal_spot.map((spot, index) => [
    item.key, index + 1, spot, ...data.model_names.map(name => item.pnl[name][index]),
    item.realized_vol * 100, item.spot_jump * 100, item.jump_step, item.cost_bps, item.rebalance_every,
    item.hedge_dates, s.spot, s.strike, s.years * 365, s.kind, s.vol * 100, s.rate * 100, s.steps, s.seed
  ].join(",")));
  downloadButton("stress-export", "hedge-stress-paths.csv", [csvHeader, ...rows].join("\r\n"), "text/csv");
  if (data.scenarios.some(item => item.neural_domain_coverage < 1)) {
    return "Some hedge dates used neural extrapolation. See training range by scenario.";
  }
});

connect("numerical", "numerical", "Pricing the contract at each resolution…", data => {
  const tree = data.tree.filter(row => row.price !== null);
  const finalTree = tree.at(-1);
  const finalMC = data.monte_carlo.at(-1);
  const {format: fmtPrice} = new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 6
  });
  const fmtError = value => value > 0 && value < .000001 ?
    "$" + value.toExponential(2) : fmtPrice(value);
  const fmtTime = value => number(value, 1) + " µs";
  renderCards("numerical-metrics", [
    ["Black–Scholes", fmtPrice(data.reference.price), fmtTime(data.reference.median_microseconds)],
    ["CRR tree", finalTree ? fmtPrice(finalTree.price) : "Unavailable",
      finalTree ? number(finalTree.steps) + " steps · " + fmtTime(finalTree.median_microseconds) : "Increase the step limit"],
    ["Plain Monte Carlo", fmtPrice(finalMC.plain.price), number(finalMC.paths) + " terminal payoffs"],
    ["Antithetic Monte Carlo", fmtPrice(finalMC.antithetic.price), number(finalMC.paths) + " terminal payoffs"]
  ]);

  byId("tree-plot").hidden = tree.length === 0;
  if (tree.length) {
    chart("tree-chart", {x: tree.map(row => row.steps), ys: [tree.map(row => row.absolute_error)],
      labels: ["CRR tree"], logX: true}, "Tree steps (log scale)", "Absolute error ($)");
  }
  toTable("tree-table", [["steps", "Steps", number], ["price", "Price", fmtPrice],
    ["absolute_error", "|Error|", fmtError], ["median_microseconds", "Median time", fmtTime],
    ["status", "Status"]], data.tree.map(row => ({...row,
      status: row.reason ? "Invalid risk-neutral probability" : "Priced"})));

  const samplers = [["plain", "Plain sampling"], ["antithetic", "Antithetic sampling"]];
  byId("mc-plot").hidden = false;
  chart("mc-chart", {x: data.monte_carlo.map(row => row.paths),
    ys: samplers.map(([key]) => data.monte_carlo.map(row => row[key].absolute_error)),
    labels: samplers.map(([, label]) => label), logX: true}, "Terminal payoffs (log scale)", "Absolute error ($)");
  const mcRows = data.monte_carlo.flatMap(row => samplers.map(([key, method]) => {
    const result = row[key];
    return {...result, method, paths: row.paths,
      interval: result.ci_lower === null ? "Insufficient sample" :
        fmtPrice(result.ci_lower) + " to " + fmtPrice(result.ci_upper)};
  }));
  toTable("mc-table", [["paths", "Payoffs", number], ["method", "Sampler"], ["price", "Estimate", fmtPrice],
    ["absolute_error", "|Error|", fmtError], ["standard_error", "Std. error", fmtError], ["interval", "Approx. 95% interval"],
    ["positive_observations", "Positive obs.", number], ["median_microseconds", "Median time", fmtTime]], mcRows);

  downloadButton("numerical-export", "options-numerical-benchmark.json", JSON.stringify(data, null, 2), "application/json");
  if (!tree.length) return "The tree could not price this contract at any selected step count. The Monte Carlo results are available.";
  if (tree.length < data.tree.length) return "Coarse tree grids with invalid risk-neutral probabilities were skipped.";
});

const percent = value => number(value, 2) + "%";
const surfacePremium = value => value > 0 && value < .0001 ? "$" + value.toExponential(3) :
  "$" + number(value, 6);

function colorHeatCell(cell, value, low, high) {
  const fraction = high - low < 1e-8 ? .5 : (value - low) / (high - low);
  const palette = [[24, 42, 69], [37, 104, 128], [113, 231, 195]];
  const segment = fraction < .5 ? 0 : 1;
  const mix = fraction * 2 - segment;
  const rgb = palette[segment].map((channel, i) => Math.round(channel + mix * (palette[segment + 1][i] - channel)));
  const linear = rgb.map(channel => ((channel / 255 + .055) / 1.055) ** 2.4);
  const luminance = .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
  cell.style.backgroundColor = "rgb(" + rgb.join(",") + ")";
  cell.style.color = luminance < .179 ? "#fff" : "#000";
}

function renderSurfaceHeatmap(data, low, high) {
  const table = document.createElement("table");
  table.className = "heatmap";
  table.setAttribute("aria-label", "Implied volatility by expiry and strike as a percentage of forward");
  const header = table.createTHead().insertRow();
  for (const label of ["Expiry", ...data.strike_pct_forward.map(percent)]) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    header.append(th);
  }
  const body = table.createTBody();
  const entries = data.expiries.map(expiry => {
    const row = body.insertRow();
    const heading = document.createElement("th");
    heading.scope = "row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "expiry-choice";
    button.textContent = expiry.days + " days";
    heading.append(button);
    row.append(heading);
    for (const quote of expiry.quotes) {
      const cell = row.insertCell();
      const value = quote.implied_vol_pct;
      cell.textContent = value === null ? "—" : percent(value);
      cell.title = "Strike " + dollars(quote.strike) + " · " + quote.kind + " · premium " +
        surfacePremium(quote.premium) + (value === null ? " · insufficient time value" : "");
      if (value !== null) colorHeatCell(cell, value, low, high);
    }
    return {row, button, expiry};
  });
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.tabIndex = 0;
  wrap.setAttribute("role", "region");
  wrap.setAttribute("aria-label", "Scrollable implied volatility heatmap");
  wrap.append(table);
  byId("surface-grid").replaceChildren(wrap);

  function select(entry) {
    for (const option of entries) {
      option.button.setAttribute("aria-pressed", String(option === entry));
      option.row.classList.toggle("selected", option === entry);
    }
    const {expiry} = entry;
    byId("surface-selection").textContent = expiry.days + " days to expiry · Forward " + dollars(expiry.forward);
    const pad = Math.max(.5, (high - low) * .06);
    chart("surface-chart", {x: data.strike_pct_forward,
      ys: [expiry.quotes.map(quote => quote.implied_vol_pct)], labels: ["Recovered IV"],
      yRange: [Math.max(0, low - pad), high + pad], points: true}, "Strike (% of forward)", "Implied volatility (%)");
    toTable("surface-quotes", [["strike_pct_forward", "Strike / forward", percent], ["strike", "Strike ($)", dollars],
      ["kind", "Option"], ["premium", "Premium", surfacePremium], ["input_iv_pct", "Input IV", percent],
      ["implied_vol_pct", "Recovered IV", percent]], expiry.quotes);
  }
  for (const entry of entries) entry.button.addEventListener("click", () => select(entry));
  select(entries.find(entry => entry.expiry.days === data.reference_days));
}

connect("surface", "surface", "Generating premiums and solving for implied volatility…", data => {
  const values = data.expiries.flatMap(expiry => expiry.quotes.map(quote => quote.implied_vol_pct)).filter(Number.isFinite);
  const low = Math.min(...values), high = Math.max(...values);
  const {recovered, contracts, max_absolute_error_vol_points: maxError} = data.summary;
  byId("surface-results").hidden = false;
  renderCards("surface-metrics", [
    ["Recovered quotes", recovered + " / " + contracts, data.expiries.length + " expiries"],
    ["Implied volatility range", percent(low) + " to " + percent(high), "Annualized"],
    ["Max recovery error", maxError.toExponential(2), "Volatility points"]
  ]);
  byId("surface-minimum").textContent = percent(low);
  byId("surface-maximum").textContent = percent(high);
  renderSurfaceHeatmap(data, low, high);

  const {spot, rate, dividend, atm_vol, skew, smile, term} = data.settings;
  const header = "days_to_expiry,spot,forward,strike_pct_forward,strike,option_type,premium,input_iv_pct,implied_vol_pct," +
    "absolute_error_vol_points,status,rate_pct,dividend_pct,atm_90d_vol_pct,skew_vol_points,smile_vol_points,term_vol_points_per_doubling";
  const rows = data.expiries.flatMap(expiry => expiry.quotes.map(quote =>
    [expiry.days, spot, expiry.forward, quote.strike_pct_forward, quote.strike, quote.kind, quote.premium, quote.input_iv_pct,
      quote.implied_vol_pct, quote.absolute_error_vol_points, quote.status, rate * 100, dividend * 100, atm_vol, skew, smile, term].join(",")));
  downloadButton("surface-export", "synthetic-volatility-surface.csv", [header, ...rows].join("\r\n"), "text/csv");
  if (recovered < contracts) return (contracts - recovered) + " quotes had too little time value for a reliable implied volatility.";
});

const marketSource = byId("market-source");
function updateMarketFields() {
  for (const mode of ["csv", "alpaca"]) {
    const fields = byId("market-" + mode + "-fields");
    fields.disabled = fields.hidden = marketSource.value !== mode;
  }
}
marketSource.addEventListener("change", updateMarketFields);
updateMarketFields();

const marketQuoteLabels = {
  two_sided: "Two-sided", missing_side: "Missing bid or ask", crossed: "Crossed market", zero_side: "Zero bid or ask"
};
const marketIVLabels = {
  recovered: "Recovered", unsupported_expiry: "Expiry outside model range", after_valuation_date: "Quote after valuation date",
  no_midpoint: "No usable midpoint", outside_european_bounds: "Outside European price bounds",
  no_time_value: "Insufficient time value", iv_unavailable: "IV unavailable"
};

async function readMarketFields(fields) {
  const body = formPayload(fields, ["source", "as_of", "symbol", "feed", "expiry"]);
  if (body.source === "csv") {
    const file = new Map(fields).get("quotes_file");
    if (!file?.size || file.size > 180000) throw new Error("Select a nonempty CSV up to 180,000 bytes.");
    body.quotes_csv = await file.text();
  }
  return body;
}

connect("market", "market", "Loading and analyzing the option chain…", data => {
  byId("market-results").hidden = false;
  const source = {example: "Synthetic example", csv: "Uploaded CSV", alpaca: "Alpaca · " + data.options_feed}[data.source];
  const recovered = data.rows.filter(row => row.iv_pct !== null);
  renderCards("market-metrics", [
    ["Underlying", dollars(data.spot), data.symbol || "Spot supplied with CSV"],
    ["Quotes", number(data.rows.length), source],
    ["Midpoint IVs", number(recovered.length), "Recovered with Black–Scholes"],
    ["Reference volatility", percent(data.reference_vol_pct), "Shared by all pricers"]
  ]);
  const spotTime = data.spot_time ? " · IEX spot trade " + data.spot_time : "";
  byId("market-provenance").textContent = source + " · Valuation date " + data.as_of + " UTC" + spotTime;
  const expiryControl = byId("market-expiry");
  const kindControl = byId("market-kind");
  const expiries = [...new Set(data.rows.map(row => row.expiry))];
  expiryControl.replaceChildren(...expiries.map(expiry => {
    const option = document.createElement("option");
    option.value = option.textContent = expiry;
    return option;
  }));
  expiryControl.value = recovered[0]?.expiry || expiries[0];
  kindControl.value = recovered[0]?.kind || data.rows[0].kind;

  const renderSlice = () => {
    const rows = data.rows.filter(row => row.expiry === expiryControl.value && row.kind === kindControl.value);
    const hasIV = rows.some(row => row.iv_pct !== null);
    byId("market-iv-plot").hidden = !hasIV;
    if (hasIV) {
      chart("market-iv-chart", {
        x: rows.map(row => row.strike),
        ys: [rows.map(row => row.iv_pct), rows.map(() => data.reference_vol_pct)],
        labels: ["Midpoint IV", "Reference volatility"], points: true
      }, "Strike ($)", "Volatility (%)");
    }
    toTable("market-quotes", [
      ["strike", "Strike", dollars], ["bid", "Bid", dollars], ["ask", "Ask", dollars], ["mid", "Mid", dollars],
      ["iv_pct", "Mid IV", percent], ...seriesNames.map((label, i) => [
        "model_" + i, ["Black–Scholes", "Price-only", "Price + delta", "Mixture"][i], dollars
      ]), ["quote_time", "Quote time (UTC)"], ["status", "Status"]
    ], rows.map(row => {
      let status = marketQuoteLabels[row.quote_status] + " · " + marketIVLabels[row.iv_status];
      if (row.quote_age_days > 0) status += " · Quote " + row.quote_age_days + " calendar day(s) old";
      if (row.in_domain === false) status += " · Outside training range";
      return {...row, status, ...Object.fromEntries(seriesNames.map((label, i) => ["model_" + i, row.model_prices[label]]))};
    }));
    byId("market-slice-status").textContent = !rows.length ? "This expiry has no quotes for the selected option type." :
      hasIV ? "" : "This selection has no recoverable midpoint implied volatilities.";
  };
  expiryControl.onchange = kindControl.onchange = renderSlice;
  renderSlice();

  const columns = "expiry,strike,kind,bid,ask,quote_time,contract_symbol,days,mid,spread,mid_iv_pct,quote_status,iv_status," +
    "quote_age_days,in_training_domain,black_scholes,price_only,price_delta,mixture,source,symbol,spot,spot_time," +
    "as_of,options_feed,stock_feed,reference_vol_pct,rate_pct,dividend_pct,truncated";
  const rows = data.rows.map(row => [
    row.expiry, row.strike, row.kind, row.bid, row.ask, row.quote_time, row.contract_symbol, row.days, row.mid, row.spread,
    row.iv_pct, row.quote_status, row.iv_status, row.quote_age_days, row.in_domain,
    ...seriesNames.map(label => row.model_prices[label]), data.source, data.symbol, data.spot, data.spot_time, data.as_of,
    data.options_feed, data.stock_feed, data.reference_vol_pct, data.rate_pct, data.dividend_pct, data.truncated
  ].join(","));
  downloadButton("market-export", "options-chain-analysis.csv", [columns, ...rows].join("\r\n"), "text/csv");
  if (data.truncated) return "The snapshot reached the quote limit. Select a single expiry to narrow the request.";
}, readMarketFields);

connect("report", "report", "Running the reference pricing, robustness, and hedge experiments…", data => {
  byId("report-results").hidden = false;
  const {pricing, robustness, hedge_stress: stress} = data;
  const testContracts = robustness.regimes.reduce((total, regime) => total + regime.contracts, 0);
  renderCards("report-metrics", [
    ["Training seeds", robustness.runs.map(run => number(run.seed)).join(", "), "Same test draw for every model"],
    ["Test contracts", number(testContracts), "Per option type and training seed"],
    ["Simulated paths", number(stress.settings.paths), "Per hedge scenario"]
  ]);
  const contract = pricing.contract;
  byId("report-contract").textContent = [
    "European call", number(contract.years * 365, 0) + " days", "Spot " + dollars(contract.spot),
    "Strike " + dollars(contract.strike), percent(contract.vol * 100) + " volatility",
    percent(contract.rate * 100) + " risk-free rate", percent(contract.dividend * 100) + " dividend yield"
  ].join(" · ");
  const formatGreek = value => number(value, 6);
  toTable("report-pricing", [["method", "Model"], ["price", "Premium", dollars],
    ...Object.keys(greekFields).map(key => [key, greekLabel(key), formatGreek])], pricing.metrics);

  const regimeRows = robustness.regimes.map(regime => ({
    label: regime.label,
    ...Object.fromEntries(modelNames.map(method => [method, Math.max(...robustness.runs.map(run =>
      run.models[method].regimes[regime.key].errors.call.price.mae))]))
  }));
  toTable("report-robustness", [["label", "Contract regime"],
    ...modelNames.map(method => [method, method, formatError])], regimeRows);

  const base = stress.scenarios.find(scenario => scenario.key === "base");
  const combined = stress.scenarios.find(scenario => scenario.key === "combined");
  const stressedByMethod = new Map(combined.metrics.map(row => [row.method, row]));
  const stressRows = base.metrics.map(row => {
    const stressed = stressedByMethod.get(row.method);
    return {method: row.method, baseMean: row.mean_pnl, stressMean: stressed.mean_pnl,
      baseTail: row.worst_5pct_mean_pnl, stressTail: stressed.worst_5pct_mean_pnl};
  });
  toTable("report-stress", [["method", "Model"], ["baseMean", "Baseline mean", dollars],
    ["stressMean", "Combined stress mean", dollars], ["baseTail", "Baseline tail", dollars],
    ["stressTail", "Combined stress tail", dollars]], stressRows);

  downloadButton("report-export", "options-pricing-lab-report.json", JSON.stringify(data, null, 2), "application/json");
});

byId("pricing-form").requestSubmit();
