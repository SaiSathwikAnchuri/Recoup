/* Recoup — one React app, two views: the results Overview and the live Console.
   No build step: React + htm are vendored under ./vendor. */
const { createElement, useState, useEffect, useRef, Fragment } = React;
const html = htm.bind(createElement);
const R = window.RECOUP_RESULTS;

const rs = n => (n < 0 ? "−" : "+") + "₹" + Math.abs(Math.round(n)).toLocaleString("en-IN");
const rsp = n => "₹" + Math.round(n).toLocaleString("en-IN");
const pct = n => (n * 100).toFixed(1) + "%";
const kk = n => n >= 1e5 ? "₹" + (n / 1e5).toFixed(2).replace(/\.?0+$/, "") + "L"
             : n >= 1000 ? "₹" + Math.round(n / 1000) + "k" : "₹" + Math.round(n);
const api = (u, o) => fetch(u, o).then(r => r.json());
const CAUSE = { insufficient_balance: "insufficient balance", bank_downtime: "bank downtime",
                limit_breach: "limit breach", mandate_dead: "dead mandate" };

/* ---------------------------------------------------------------- header */
function Header({ view, setView, live, mode, session }) {
  return html`
    <header><div class="bar">
      <div class="brand"><span class="dot"></span>Recoup
        <span class="tag">${mode ? mode.replace("_", " ") : "static"}</span></div>
      <div class="tabs">
        <button class=${view === "overview" ? "on" : ""} onClick=${() => setView("overview")}>Results</button>
        <button class=${view === "console" ? "on" : ""} disabled=${!live}
          title=${live ? "" : "run `python tasks.py serve` to enable"}
          onClick=${() => live && setView("console")}>Live console</button>
      </div>
      ${view === "console" && html`<div class="sessionstats">
        <div><div class="n">${session.seen}</div><div class="l">cases seen</div></div>
        <div><div class="n pos">${rsp(session.recovered)}</div><div class="l">recovered</div></div>
        <div><div class="n">${session.escalated}</div><div class="l">escalated</div></div>
        <div><div class="n pos">${rs(session.delta)}</div><div class="l">net vs fixed schedule</div></div>
      </div>`}
    </div></header>`;
}

/* -------------------------------------------------------------- overview */
function Bars({ rows, max, oracle }) {
  const ref = useRef();
  useEffect(() => {
    if (!ref.current) return;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      ref.current.querySelectorAll(".bfill").forEach(f => f.style.transform = "scaleX(1)");
    }));
  }, []);
  return html`<div class="bars" ref=${ref}>
    ${rows.map(b => {
      const w = (b.net / max * 100).toFixed(1);
      return html`<div class=${"brow" + (b.hi ? " hi" : "")} key=${b.name}>
        <span class="lab">${b.name}</span>
        <span class="btrack">
          <span class="bfill" style=${{ width: w + "%", transform: "scaleX(0)" }}></span>
          <span class="bval">${rs(b.net)}<span class="g">${b.gap != null ? b.gap + "% of gap" : ""}</span></span>
        </span></div>`;
    })}
    ${oracle != null && html`<div class="oline"
       style=${{ left: `calc(8rem + 0.85rem + ${oracle / max} * (100% - 8rem - 0.85rem))` }}>
       <span>oracle · recovery ceiling</span></div>`}
  </div>`;
}

function Dots({ rows, max }) {
  const x = n => (n / max * 100).toFixed(1) + "%";
  return html`<div>${rows.map(d => html`
    <div class="drow" key=${d.name}>
      <span class="lab">${d.name}</span>
      <span class="dplot">
        <span class="daxis"></span><span class="dzero"></span>
        <span class="dci" style=${{ left: x(d.lo), width: x(d.hi - d.lo) }}></span>
        <span class="dpt" style=${{ left: x(d.v) }}></span>
        <span class="dv">${rs(d.v)}</span>
      </span></div>`)}</div>`;
}

function ExampleDecision({ live }) {
  const [d, setD] = useState(null);
  useEffect(() => {
    if (live) api("/demo/random", { method: "POST" }).then(r => setD(r)).catch(() => {});
  }, [live]);
  const c = d && d.case, dec = d && d.decision;
  return html`<div class="lane" style=${{ marginTop: "1.4rem" }}>
    <div class="lh"><span>${c ? c.case_id : "case c0003"} · ${c ? rsp(c.mandate.amount) : "₹4,622"} ·
      ${c ? c.failure.token.replace(/_/g, " ") : "insufficient balance"}</span>
      ${live && html`<span class="muted" style=${{ fontSize: ".72rem" }}>live · fresh from the engine</span>`}</div>
    <div class="lb" style=${{ fontFamily: '"Inter Tight",sans-serif', color: "var(--ink-body)", fontSize: ".9rem", lineHeight: 1.6 }}>
      ${dec ? dec.narration
        : "Diagnosis: most likely insufficient balance (72% confidence). Predicted funding window 26–33 days out; billing date at day 25. Decision: retry on days 26 / 33 / 40 — bracket the window, hedge late, no message. Best EV ≈ ₹7,173."}
    </div>
  </div>`;
}

function Overview({ d, live }) {
  const sb = d.scoreboard, base = sb.find(r => r.name === "fixed schedule");
  return html`<${Fragment}>
    <section class="hero wrap">
      <span class="eyebrow">Razorpay Buildathon · Track 03</span>
      <h1>Fewer, better-timed actions <span class="accent">recover more money</span> than retrying harder.</h1>
      <p class="lede">Recoup replaces the fixed retry schedule for a failed UPI AutoPay debit with one
        priced decision: infer why it failed, predict when the customer will have funds, weigh every
        permitted action against its cost — and stop when nothing is worth doing.</p>
      <div class="headline-stat">
        <span class="big">${rs(d.headline.per_case)} / case</span>
        <span class="cap">${`net value created versus the industry fixed schedule, across ${d.headline.n} simulated mandate failures · paired 95% CI [${rs(d.headline.ci[0])}, ${rs(d.headline.ci[1])}]`}</span>
      </div>
    </section>

    <section class="tint"><div class="wrap">
      <p class="kicker">The problem</p>
      <h2>A blind schedule leaves money on the table and churns good customers.</h2>
      <p>UPI AutoPay debits fail <span class="mono">8–15%</span> of the time — most often because an
        account is briefly short of funds — and roughly <span class="mono">20 million</span> mandates
        are revoked every month. The standard response is a calendar picked once in 2023: retry on
        day 1, 3, 7, send an SMS, then lapse. It is blind to the cause, to the customer's pay cycle,
        and to what each attempt costs.</p>
    </div></section>

    <section class="wrap">
      <p class="kicker">The decision</p>
      <h2>Every candidate action is priced. The best one wins, or none does.</h2>
      <div class="formula">${"EV(a) = "}<b>P(success | cause, day)</b>${" · recovery_value\n        − action_cost\n        − "}<b>P(revocation)</b>${" · mandate_LTV\n        − missed_cycle_penalty · P(recovered late)"}</div>
      <p style=${{ marginTop: "1rem" }}>
        Cause from a <strong>calibrated classifier</strong>${" "}
        ${`(accuracy ${d.classifier.acc.toFixed(2)} vs a 0.60 prior, ECE ${d.classifier.ece}).`}${" "}
        Timing from a <strong>quantile funding-day model</strong>${" "}
        ${`(median error ${d.liquidity.mae} days vs ${d.liquidity.naive} for a heuristic).`}${" "}
        Every plan is then filtered through the hard NPCI / TRAI limits — a structural filter, not a prompt.</p>
      <${ExampleDecision} live=${live} />
    </section>

    <section class="tint"><div class="wide">
      <p class="kicker">The scoreboard</p>
      <h2>Each policy adds one idea, so the harness can price it alone.</h2>
      <p class="muted" style=${{ maxWidth: "44rem", fontSize: ".92rem" }}>Net value = recovered debit,
        minus action spend, minus the missed-cycle penalty on late recoveries, plus the LTV of every
        mandate still alive. Compare by the paired delta.</p>
      <div class="panel"><div class="scroll"><table>
        <thead><tr><th>Policy</th><th>Recovered</th><th>Rate</th><th>Attempts</th><th>Msg/case</th>
          <th>On-time</th><th>Preserved</th><th>Net Δ/case</th></tr></thead>
        <tbody>${sb.map(r => {
          const delta = r.name === "fixed schedule" ? 0 : Math.round(r.net - base.net) / d.headline.n;
          const isRec = r.name === "Recoup";
          return html`<tr class=${isRec ? "hi" : (["never act", "fixed schedule", "always nudge"].includes(r.name) ? "dim" : "")} key=${r.name}>
            <td>${r.name}${r.name === "fixed schedule" ? html`<span class="tag">baseline</span>` : ""}</td>
            <td>${kk(r.recovered)}</td><td>${pct(r.rate)}</td><td>${r.attempts}</td>
            <td>${r.msg.toFixed(2)}</td><td>${pct(r.on_time)}</td><td>${pct(r.preserved)}</td>
            <td class=${delta > 0 ? "pos" : ""}>${r.name === "fixed schedule" ? "0" : rs(delta)}</td>
          </tr>`;
        })}</tbody></table></div></div>
    </div></section>

    <section class="wide">
      <p class="kicker">What each part is worth</p>
      <h2>Take a capability away, watch the value fall.</h2>
      <p class="muted" style=${{ maxWidth: "44rem", fontSize: ".92rem" }}>Net value gained per case vs
        the fixed schedule. Dashed line: the perfect-timing oracle — the ceiling for raw recovery,
        not net value, which is why Recoup passes it (by ${kk(Math.abs(d.recoup_minus_oracle))}).</p>
      <figure><${Bars} rows=${d.ablation.filter(a => a.name !== "oracle").map(a => ({
        name: a.name.replace("no cause", "classifier only").replace("no timing", "timing only")
                     .replace("liquidity aware", "timing, no cost"),
        net: a.net, gap: a.gap, hi: a.name === "recoup"
      }))} max=${3400} oracle=${(d.ablation.find(a => a.name === "oracle") || {}).net || 2443} />
      <figcaption>Removing the classifier costs ~20 points of the recovery gap and collapses the
        on-time rate to 15%. The timing model adds ~6 more.</figcaption></figure>
    </section>

    <section class="tint"><div class="wide">
      <p class="kicker">Does it hold up</p>
      <h2>The models stay frozen; the world moves under them.</h2>
      <p class="muted" style=${{ maxWidth: "44rem", fontSize: ".92rem" }}>Each row re-draws a fresh
        400-case world under a different assumption set. Recoup's net gain over the fixed schedule,
        95% interval:</p>
      <div class="panel"><${Dots} rows=${d.sensitivity} max=${4900} /></div>
      <p class="muted" style=${{ marginTop: "1rem", fontSize: ".88rem" }}>Every interval clears zero —
        the ordering survives a churn hazard nearly doubled, an LTV halved, outages tripled.</p>
    </div></section>

    <section class="wide">
      <p class="kicker">Fair to whom</p>
      <h2>Every income group gains — but not equally, and we say so.</h2>
      <div class="panel"><div class="scroll"><table>
        <thead><tr><th>Group</th><th>n</th><th>Recovery</th><th>vs fixed</th><th>On-time</th>
          <th>Escalated</th><th>Net Δ/case</th></tr></thead>
        <tbody>${d.fairness.map(f => html`<tr key=${f.group}>
          <td>${f.group}</td><td>${f.n}</td><td>${pct(f.recovery)}</td>
          <td class="pos">+${(f.vs_fixed * 100).toFixed(0)} pts</td><td>${pct(f.on_time)}</td>
          <td>${pct(f.escalated)}</td><td class="pos">${rs(f.net)}</td></tr>`)}</tbody>
      </table></div></div>
      <p class="muted" style=${{ marginTop: "1rem", fontSize: ".88rem" }}>Salaried customers recover
        ~10 points more — a regular payday makes the funding-window model most accurate for them.
        The gap is model accuracy by income regularity, not the decision logic.</p>
    </section>

    <section class="tint"><div class="wrap">
      <p class="kicker">Built to a scope</p>
      <h2>What this is, and what it deliberately is not.</h2>
      <div class="facts">
        <div class="fact"><div class="n">1</div><div class="l">decision chain — not a multi-agent orchestra</div></div>
        <div class="fact"><div class="n">4</div><div class="l">action types: retry, nudge, SMS, re-auth</div></div>
        <div class="fact"><div class="n">117</div><div class="l">tests; one command reproduces every number</div></div>
        <div class="fact"><div class="n">0</div><div class="l">real customer records — synthetic under DPDP</div></div>
      </div>
      <ul class="plain">
        <li>No dashboard as the headline — the console operates the engine, it doesn't stand in for the result.</li>
        <li>The LLM writes audit prose only; it never makes or influences a decision.</li>
        <li>Every prior is sourced or flagged; a sensitivity sweep shows the result doesn't hinge on the exact numbers.</li>
      </ul>
    </div></section>

    <section class="wrap" style=${{ paddingTop: "2.4rem", paddingBottom: "3.5rem" }}>
      <p class="kicker" style=${{ marginBottom: ".7rem" }}>Reproduce every figure</p>
      <div class="codeblock">pip install -r requirements.txt <span class="p">&&</span> python tasks.py reproduce</div>
      <p class="muted" style=${{ marginTop: "1rem", fontSize: ".87rem" }}>Simulator → harness → cost model →
        classifier → funding-window model → constraint filter → EV policy → audit trail → evidence → service.
        Each stage has its own tests; the policy is stage seven, after everything it needs to be honest about.</p>
    </section>
  <//>`;
}

/* --------------------------------------------------------------- console */
function CausePosterior({ p }) {
  const sorted = Object.entries(p).sort((a, b) => b[1] - a[1]);
  return html`<div>${sorted.map(([k, v]) => html`
    <div class=${"cbar" + (k === "mandate_dead" ? " dead" : "")} key=${k}>
      <span>${CAUSE[k] || k}</span>
      <span class="t"><span class="f" style=${{ width: (v * 100).toFixed(0) + "%" }}></span></span>
      <span class="v">${(v * 100).toFixed(0)}%</span></div>`)}</div>`;
}

function CustomerStateStrip({ s }) {
  if (!s) return null;
  const risk = s.churn_risk >= 0.35 ? "warn" : "";
  return html`<div style=${{ display: "flex", gap: ".4rem", flexWrap: "wrap", margin: ".6rem 0" }}>
    <span class="chip">stage · ${s.recovery_stage}</span>
    <span class=${"chip " + risk}>churn risk · ${(s.churn_risk * 100).toFixed(0)}%</span>
    <span class="chip">prior recovery attempts · ${s.recovery_attempts}</span>
    <span class="chip">messages sent · ${s.message_count}</span>
    <span class="chip">est. LTV · ${kk(s.estimated_ltv)}</span>
  </div>`;
}

function Decision({ d, c }) {
  const w = d.funding_window, plan = d.decision;
  const ros = d.ros_candidates;
  return html`<div class="card">
    <div class="row" style=${{ justifyContent: "space-between", alignItems: "baseline" }}>
      <h2 style=${{ margin: 0 }}>Decision · <span class="mono">${c.case_id}</span>
        ${d.policy === "recoup_v2" && html`<span class="chip">adaptive</span>`}</h2>
      <span class="mono muted">${rsp(c.mandate.amount)} · ${c.mandate.category} · ${c.failure.token.replace(/_/g, " ")}</span>
    </div>
    <${CustomerStateStrip} s=${d.customer_state} />
    <div style=${{ display: "grid", gap: "1rem", gridTemplateColumns: "1fr 1fr", marginTop: ".8rem" }}>
      <div><h3>Likely cause</h3><${CausePosterior} p=${d.cause_posterior} /></div>
      <div><h3>Funding window</h3><dl class="kv">
        <dt>predicted p50 / p85</dt><dd>${w.p50_days.toFixed(0)} / ${w.p85_days.toFixed(0)} d</dd>
        <dt>next billing date</dt><dd>day ${d.cycle_close_day}</dd>
        <dt>mandate age</dt><dd>${c.mandate.age_months} mo</dd>
        <dt>recent funding days</dt><dd>${(c.history.success_days_of_month || []).slice(-6).join(", ") || "—"}</dd>
      </dl></div>
    </div>
    ${ros
      ? html`<h3 style=${{ marginTop: "1rem" }}>Recovery Opportunity Score — every candidate</h3>
        <div class="scroll"><table>
          <thead><tr><th>Action</th><th>When</th><th>P(clear)</th><th>Retention</th>
            <th>ROS</th><th>EV</th></tr></thead>
          <tbody>${ros.map((x, i) => html`<tr key=${i}>
            <td>${x.action}</td><td>${x.scheduled_day == null ? "—" : "day " + x.scheduled_day}</td>
            <td>${x.p_success ? (x.p_success * 100).toFixed(0) + "%" : "—"}</td>
            <td>${(x.retention * 100).toFixed(0)}%</td>
            <td class=${x.score > 0 ? "pos" : "muted"}>${rs(x.score)}</td>
            <td class="muted">${rs(x.ev)}</td></tr>`)}</tbody>
        </table></div>`
      : html`<h3 style=${{ marginTop: "1rem" }}>Every option, priced</h3>
        <div class="scroll"><table>
          <thead><tr><th>Action</th><th>When</th><th>Expected value</th></tr></thead>
          <tbody>${(d.candidates_top || []).map((x, i) => html`<tr key=${i}>
            <td>${x.action}</td><td>${x.day == null ? "—" : "day " + x.day}</td>
            <td class=${x.ev > 0 ? "" : "muted"}>${rs(x.ev)}</td></tr>`)}</tbody>
        </table></div>`}
    <h3 style=${{ marginTop: "1rem" }}>Recoup does</h3>
    <div>${plan.actions.length
      ? plan.actions.map((a, i) => html`<span class="chip" key=${i}>${a.kind} · day ${a.day}</span>`)
      : html`<span class="chip warn">nothing — hand to a human</span>`}
      ${plan.terminal === "escalate" && html`<span class="chip warn">then escalate</span>`}</div>
    <div class="narr">${d.narration}</div>
  </div>`;
}

function Lane({ o, label, amount }) {
  const evs = o.timeline.filter(x => ["retry", "reauth", "sms", "nudge", "revoked"].includes(x.action));
  const end = o.recovered
    ? html`<span class="pill rec">recovered ${rsp(o.amount_recovered)} ${o.recovered_on_time ? "on time" : "late (" + Math.round(o.days_to_recovery) + "d)"}</span>`
    : o.revoked ? html`<span class="pill rev">mandate revoked — ${rsp(amount)} lost</span>`
    : o.escalated ? html`<span class="pill esc">escalated to a human</span>`
    : html`<span class="pill open">unresolved</span>`;
  return html`<div class="lane">
    <div class="lh"><span>${label}</span>${end}</div>
    <div class="lb">
      ${evs.length ? evs.map((e, i) => e.action === "revoked"
        ? html`<div class="ev bad" key=${i}>day ${e.day} · customer revoked the mandate</div>`
        : html`<div class=${"ev" + (e.result === "success" ? " hit" : "")} key=${i}>day ${e.day} · ${e.action}${e.p_success != null ? " · P " + e.p_success.toFixed(2) : ""}${e.result ? " · " + e.result : ""}</div>`)
        : html`<div class="ev">no action taken</div>`}
      <div class="ev" style=${{ borderTop: "1px solid var(--rule)", paddingTop: ".3rem", marginTop: ".2rem" }}>
        <span>${o.attempts_used} retries · ${o.messages_sent} messages</span><span>net ${rs(o.net_value)}</span></div>
    </div></div>`;
}

function ModelHealth() {
  const [h, setH] = useState(null);
  useEffect(() => { api("/api/models/health").then(setH).catch(() => {}); }, []);
  if (!h) return null;
  const dot = s => ({ good: "var(--green)", warn: "var(--amber)", alert: "var(--red)" }[s] || "var(--ink-soft)");
  const cell = (label, val, status) => html`<div>
    <div class="l" style=${{ fontSize: ".7rem", color: "var(--ink-soft)" }}>${label}</div>
    <div style=${{ fontFamily: '"IBM Plex Mono",monospace', fontSize: ".95rem" }}>
      ${status && html`<span style=${{ color: dot(status) }}>● </span>`}${val}</div></div>`;
  return html`<div class="card">
    <h2>Model health</h2>
    <p class="muted" style=${{ fontSize: ".78rem", margin: ".2rem 0 .8rem" }}>offline batch, seed 42 · ${h.generated_from}</p>
    <div style=${{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: ".8rem" }}>
      ${cell("classifier accuracy", (h.classifier.accuracy * 100).toFixed(1) + "%")}
      ${cell("calibration ECE", h.classifier.ece_calibrated, h.classifier.status)}
      ${cell("escalate precision", (h.classifier.escalate_precision * 100).toFixed(0) + "%")}
      ${cell("funding-day MAE", h.liquidity.mae_days + " d")}
      ${cell("p85 coverage", (h.liquidity.p85_coverage * 100).toFixed(0) + "%", h.liquidity.status)}
      ${cell("policy recovery", (h.policy.recovery_rate * 100).toFixed(1) + "%")}
    </div>
  </div>`;
}

function Timeline({ caseId }) {
  const [ev, setEv] = useState(null);
  useEffect(() => {
    if (caseId) api("/api/cases/" + encodeURIComponent(caseId) + "/timeline")
      .then(r => setEv(r.events || [])).catch(() => setEv([]));
  }, [caseId]);
  if (!ev) return null;
  return html`<div class="card">
    <h2>Customer timeline</h2>
    ${ev.length === 0
      ? html`<p class="empty">no events yet</p>`
      : html`<div class="lane"><div class="lb">${ev.map((e, i) => {
          const p = e.payload || {};
          const extra = p.result ? " · " + p.result : p.channel ? " · " + p.channel
            : p.via ? " · via " + p.via : p.day != null ? " · day " + p.day : "";
          return html`<div key=${i} style=${{ display: "flex", justifyContent: "space-between" }}>
            <span>${e.label}${extra}</span></div>`;
        })}</div></div>`}
  </div>`;
}

function Console({ live, mode, session, setSession }) {
  const [cause, setCause] = useState("");
  const [busy, setBusy] = useState(false);
  const [cur, setCur] = useState(null);       // {case, decision, simulation?}
  const [log, setLog] = useState([]);
  const [wh, setWh] = useState("");

  const refreshLog = () => {
    api("/cases?limit=40").then(r => setLog(r.cases || []));
    // Both all-time (DB-backed) so a page reload shows the same numbers it did before —
    // "net vs fixed schedule" used to be a client-only counter that reset to ₹0 on refresh.
    api("/stats").then(s => setSession(x => ({ ...x, seen: s.n, recovered: s.recovered_rs, escalated: s.escalated })));
    api("/api/metrics").then(m => setSession(x => ({ ...x, delta: m.demo_net_delta_total || 0 })));
  };
  useEffect(refreshLog, []);

  const gen = async () => {
    setBusy(true);
    try {
      const r = await api("/demo/random" + (cause ? "?cause=" + cause : ""), { method: "POST" });
      if (r.detail) { alert(r.detail); return; }
      setCur(r);
      refreshLog();
    } finally { setBusy(false); }
  };
  const sendWebhook = async () => {
    let body; try { body = JSON.parse(wh); } catch { alert("not valid JSON"); return; }
    const r = await fetch("/webhook", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then(x => x.json());
    if (r.detail) { alert(r.detail); return; }
    setCur({ case: r.case, decision: r.decision });
    refreshLog();
  };
  const openCase = async id => {
    const r = await api("/cases/" + encodeURIComponent(id));
    setCur({ case: r.case, decision: r.decision });
  };

  const sim = cur && cur.simulation;
  const curId = cur && cur.case && cur.case.case_id;
  return html`<div class="console-grid">
    <div class="stack">
      <${ModelHealth} />
      <div class="card">
        <h2>Ingest a failed mandate</h2>
        <p class="muted" style=${{ fontSize: ".85rem", margin: ".2rem 0 .8rem" }}>
          Synthesise one (with its hidden outcome, so it can be scored), or POST a real
          Razorpay <code>payment.failed</code> body.</p>
        <div class="row" style=${{ marginBottom: ".7rem" }}>
          <select value=${cause} onChange=${e => setCause(e.target.value)}>
            <option value="">any cause</option>
            <option value="insufficient_balance">insufficient balance</option>
            <option value="bank_downtime">bank downtime</option>
            <option value="limit_breach">limit breach</option>
            <option value="mandate_dead">dead mandate</option>
          </select>
          <button class="primary" onClick=${gen} disabled=${busy}>
            ${busy ? html`<span class="spin"></span> deciding` : "Simulate failure"}</button>
        </div>
        <details>
          <summary class="muted" style=${{ fontSize: ".82rem", cursor: "pointer" }}>paste a webhook</summary>
          <textarea value=${wh} onChange=${e => setWh(e.target.value)}
            placeholder=${'{"event":"payment.failed","payload":{"payment":{"entity":{"amount":149900,"error_code":"U30","error_description":"insufficient funds"}}}}'}></textarea>
          <button style=${{ marginTop: ".5rem" }} onClick=${sendWebhook}>Send to /webhook</button>
        </details>
      </div>
      <div class="card">
        <h2>Case log</h2>
        ${log.length === 0
          ? html`<p class="empty">nothing yet</p>`
          : html`<div class="scroll"><table><tbody>${log.map(c => html`<tr key=${c.case_id}>
              <td><a href="#" onClick=${e => { e.preventDefault(); openCase(c.case_id); }}>${c.case_id}</a><br/>
                <span class="muted" style=${{ fontSize: ".72rem" }}>${(c.top_cause || "").replace(/_/g, " ")}</span></td>
              <td>${rsp(c.amount)}<br/>
                <span class=${"pill " + ({ recovered: "rec", revoked: "rev", escalated: "esc" }[c.status] || "open")}>${c.status}</span></td>
            </tr>`)}</tbody></table></div>`}
      </div>
    </div>

    <div class="stack">
      ${cur
        ? html`<${Decision} d=${cur.decision} c=${cur.case} />`
        : html`<div class="card"><p class="empty">Ingest a mandate to see the decision.</p></div>`}
      ${curId && html`<${Timeline} caseId=${curId} />`}
      ${sim && html`<div class="card">
        <div class="row" style=${{ justifyContent: "space-between", alignItems: "baseline" }}>
          <h2 style=${{ margin: 0 }}>45-day outcome</h2>
          <span class="mono muted">true cause: ${sim.true_cause.replace(/_/g, " ")}${sim.true_best_retry_day != null ? " · funds actually arrive day " + sim.true_best_retry_day : ""}</span>
        </div>
        <${Lane} o=${sim.recoup} label="Recoup" amount=${cur.case.mandate.amount} />
        <${Lane} o=${sim.fixed_schedule} label="Fixed schedule (retry d+1/d+3/d+7 + SMS)" amount=${cur.case.mandate.amount} />
        <div class="row" style=${{ justifyContent: "space-between", alignItems: "center", marginTop: "1rem", paddingTop: ".9rem", borderTop: "1px solid var(--rule)" }}>
          <span class="muted">net value, Recoup − fixed schedule, this case</span>
          <span class="bigdelta">${rs(sim.net_delta)}</span>
        </div>
      </div>`}
    </div>
  </div>`;
}

/* ------------------------------------------------------------------ app */
function App() {
  const [view, setView] = useState("overview");
  const [health, setHealth] = useState(null);
  const [results, setResults] = useState(R);
  const [session, setSession] = useState({ seen: 0, recovered: 0, escalated: 0, delta: 0 });

  useEffect(() => {
    api("/healthz").then(setHealth).catch(() => setHealth({ ok: false }));
    api("/api/results").then(d => { if (d && d.scoreboard) setResults(d); }).catch(() => {});
  }, []);

  const live = !!(health && health.ok && health.models_ready);
  const mode = health && (health.models_ready ? health.execute_mode : null);
  return html`<${Fragment}>
    <${Header} view=${view} setView=${setView} live=${live} mode=${mode} session=${session} />
    ${view === "console" && live
      ? html`<${Console} live=${live} mode=${mode} session=${session} setSession=${setSession} />`
      : html`<${Overview} d=${results} live=${live} />`}
  <//>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
