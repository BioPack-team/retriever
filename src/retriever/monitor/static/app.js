// Retriever dashboard.
// Single Alpine component holds the whole page state; per-panel polling
// timers keep it fresh. No server-side session state — every refresh is
// an HTTP GET against the existing /status/* JSON API.

const TERMINAL_FAILURE = new Set([
  "Failed",
  "QueryNotTraversable",
  "UnsupportedConstraint",
]);

const REALTIME_INTERVAL_MS = 10_000;
const AGGREGATE_INTERVAL_MS = 60_000;
const TIMELINE_LOOKBACK_HOURS = 24;
const RECENT_FAILURES_LIMIT = 25;
const ACTIVITY_PAGE_SIZE = 25;
const ACTIVITY_MODES = ["Active", "Stuck", "Completed", "Failed"];
const ACTIVITY_LIVE_MODES = new Set(["Active", "Stuck"]);
const LOOKBACK_OPTIONS = ["Last 24h", "Last 3 days", "Last week", "All time"];
const LOOKBACK_HOURS = {
  "Last 24h": 24,
  "Last 3 days": 72,
  "Last week": 168,
  "All time": null,
};
const VIEWS = ["home", "activity", "performance", "heatmaps"];

// Sorted alphabetically so the row dimension in the failure-breakdown
// heatmaps is deterministic regardless of which statuses showed up.
const FAILURE_STATUSES = ["Failed", "QueryNotTraversable", "UnsupportedConstraint"];

// Performance view filters. Lookback is shared with the activity LOOKBACK_*
// vocabulary (so URL state survives a tab switch); the status filter is
// either "Complete" or "any-failure" (matches `/status/durations`).
const PERFORMANCE_STATUS_OPTIONS = [
  { value: "Complete", label: "Completed" },
  { value: "failed", label: "Failed" },
];
const PERFORMANCE_STATUS_VALUES = new Set(
  PERFORMANCE_STATUS_OPTIONS.map((o) => o.value),
);

// Sort fields for the activity table. For terminal modes these map
// directly to the /status/completed and /status/failed `sort` query
// param (server-side, cross-page). For live modes the same labels drive
// a client-side reorder of the in-memory list.
const SORT_FIELDS = [
  { value: "created", label: "Started" },
  { value: "completed", label: "Completed" },
  { value: "duration", label: "Duration" },
  { value: "submitter", label: "Submitter" },
  { value: "status", label: "Status" },
  { value: "results", label: "Results" },
];
const SORT_FIELD_VALUES = new Set(SORT_FIELDS.map((o) => o.value));

// Backend only ever defines tiers 0, 1, 2. Centralized so URL parsing,
// the filter dropdown, and the heatmap matrix all stay in lockstep.
const TIERS = [0, 1, 2];
const TIER_VALUES = new Set(TIERS);
// Cap how many log lines we hand the modal — large dumps (5k+ lines)
// stall the page during render. Tail-keeping the most recent N matches
// how a developer typically reads them.
const MODAL_LOG_MAX_LINES = 500;

// ============== Helpers ==============

async function fetchJson(path, { signal } = {}) {
  const resp = await fetch(path, { signal, headers: { accept: "application/json" } });
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}`);
  }
  return resp.json();
}

async function fetchText(path, { signal } = {}) {
  const resp = await fetch(path, { signal, headers: { accept: "text/plain" } });
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}`);
  }
  return resp.text();
}

/** Build `path?key=value&...`, skipping entries whose value is null/undefined.
 * Lets callers pass `{ lookback: null }` for "All time" and have the param
 * drop out cleanly without string-replace gymnastics. */
function buildUrl(path, params) {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v != null) sp.set(k, String(v));
  }
  const qs = sp.toString();
  return qs ? `${path}?${qs}` : path;
}

function relTime(iso) {
  if (!iso) return "";
  const when = new Date(iso);
  if (Number.isNaN(when.valueOf())) return iso;
  const delta = (Date.now() - when.getTime()) / 1000;
  if (delta < 0) return "just now";
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  const m = delta / 60;
  if (m < 60) return `${Math.floor(m)}m ago`;
  const h = m / 60;
  if (h < 24) return `${Math.floor(h)}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function formatUptime(seconds) {
  const s = Math.floor(seconds || 0);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${m % 60}m`;
  const d = Math.floor(h / 24);
  return `${d}d ${h % 24}h`;
}

function formatMb(mb) {
  if (mb == null) return "—";
  return `${mb.toFixed(1)}`;
}

function bucketLabel(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.valueOf())) return iso;
  return `${String(d.getHours()).padStart(2, "0")}:00`;
}

function formatLocal(iso) {
  if (!iso) return "—";
  const when = new Date(iso);
  if (Number.isNaN(when.valueOf())) return iso;
  // Browser-local timezone, medium date + time.
  return when.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatDurationSeconds(seconds) {
  if (seconds == null) return "—";
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(2)}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  if (m < 60) return `${m}m ${s.toFixed(0)}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

/**
 * Linear interpolation across two HSL endpoints, normalized within a
 * heatmap (vmin/vmax come from the surrounding helper, not a global
 * scale). Returns `transparent` for nulls so empty cells fall back to
 * the `.heatmap-cell` default background.
 *
 * Scales:
 *   `fail`     — green (good) → red (bad), hue 120 → 0.
 *   `duration` — blue (fast)  → orange (slow), hue 210 → 30.
 *   `count`    — same hue, lightness 80% → 35% (pale to saturated).
 */
const HEATMAP_SCALES = {
  fail: { h0: 120, h1: 0, s: 60, l: 50 },
  duration: { h0: 210, h1: 30, s: 65, l: 50 },
  count: { h0: 210, h1: 210, s: 70, l0: 80, l1: 35 },
};
function heatmapColor(value, vmin, vmax, scale) {
  if (value == null) return "transparent";
  if (vmax === vmin) {
    // Single non-null value with no range — pick the midpoint colour
    // so it's still visually distinct from the empty-cell background.
    const c = HEATMAP_SCALES[scale];
    if (c.l0 != null) return `hsl(${c.h0}, ${c.s}%, ${(c.l0 + c.l1) / 2}%)`;
    return `hsl(${(c.h0 + c.h1) / 2}, ${c.s}%, ${c.l}%)`;
  }
  const t = Math.max(0, Math.min(1, (value - vmin) / (vmax - vmin)));
  const c = HEATMAP_SCALES[scale];
  if (c.l0 != null) {
    const l = c.l0 + (c.l1 - c.l0) * t;
    return `hsl(${c.h0}, ${c.s}%, ${l}%)`;
  }
  const h = c.h0 + (c.h1 - c.h0) * t;
  return `hsl(${h}, ${c.s}%, ${c.l}%)`;
}

function truncateLogLines(text) {
  if (!text) return "(no logs)";
  const lines = text.split("\n");
  if (lines.length <= MODAL_LOG_MAX_LINES) return text;
  const kept = lines.slice(lines.length - MODAL_LOG_MAX_LINES);
  const dropped = lines.length - MODAL_LOG_MAX_LINES;
  return (
    `(showing last ${MODAL_LOG_MAX_LINES} of ${lines.length} lines; ` +
    `${dropped} earlier lines hidden — use the "Open full logs" link above)\n\n` +
    kept.join("\n")
  );
}

// ============== uPlot helpers ==============

/** Run `attempt` until it returns truthy, polling every 80ms.
 *
 * Used by every ensure*Plot — both Alpine `x-cloak` (which holds the
 * body at display:none until init() finishes) and the lazy `defer`
 * load of uPlot mean that a plot built during init() can be built
 * into a zero-width container, and uPlot doesn't recover cleanly
 * from that. Polling lets each ensure*Plot wait out both conditions
 * (uPlot loaded AND container measured) before constructing. */
function retryUntilBuilt(attempt) {
  const tick = () => {
    if (attempt()) return;
    window.setTimeout(tick, 80);
  };
  tick();
}

let throughputPlot = null;
let performancePlot = null;

/** Force a redraw of every live timeline plot. Called from the theme
 * toggle so axis/grid/series colors are re-read from the CSS palette. */
function redrawAllPlots() {
  for (const plot of [throughputPlot, performancePlot]) {
    if (plot) plot.redraw(false, true);
  }
  for (const plot of tierPlots.values()) {
    plot.redraw(false, true);
  }
}
// Per-tier perf chart plots, keyed by tier number. Containers are
// rendered by an `x-for` so they appear/disappear as the tier list
// changes; we lazy-init in response to each container's x-init hook.
const tierPlots = new Map();

/**
 * Compute the shared x-axis range for a timeline chart family. The
 * returned object is also threaded through `applyTimelineData` to pin
 * the y-axis: `yMax` is taken from the supplied buckets so a per-tier
 * chart fed this same range will use the overall chart's y scale, not
 * its own (otherwise each chart auto-scales independently and the
 * visual heights become incomparable).
 *
 * - `hours` finite → x-window is `[now - hours*3600, now]`.
 * - `hours` null ("All time") → derive x-window from the supplied
 *   buckets so tier charts still align with the main chart.
 */
function computeTimelineRange(hours, buckets = []) {
  const now = Math.floor(Date.now() / 1000);
  const yMax = buckets.length
    ? Math.max(1, ...buckets.map((b) => b.count))
    : 1;
  if (hours != null) {
    return { xMin: now - hours * 3600, xMax: now, yMax };
  }
  if (!buckets.length) {
    return { xMin: now - 24 * 3600, xMax: now, yMax };
  }
  const xs = buckets.map((b) =>
    Math.floor(new Date(b.bucket_start).getTime() / 1000),
  );
  return { xMin: Math.min(...xs), xMax: Math.max(now, ...xs), yMax };
}

function emptyTimelineRange(hours = 24) {
  // Two-point dataset spanning the lookback window with zero counts.
  // uPlot needs at least two distinct x values to compute a sensible
  // time scale — [[0], [0]] makes it draw an axis around the epoch.
  const now = Math.floor(Date.now() / 1000);
  return [
    [now - hours * 3600, now],
    [0, 0],
  ];
}

function makeTimelinePlot(container, { hours = 24 } = {}) {
  // Pane may still be `display:none` when init runs — fall back to a
  // sensible width and let the ResizeObserver below correct it once the
  // browser has laid the container out.
  const initWidth = container.clientWidth || 600;
  const initHeight = container.clientHeight || 180;
  // Each color is read from the CSS palette on every redraw rather
  // than once at construction. That lets a theme toggle re-tint the
  // axes/grid without rebuilding the plot.
  const cssVar = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const opts = {
    width: initWidth,
    height: initHeight,
    cursor: { drag: { setScale: false }, points: { show: false } },
    legend: { show: false },
    axes: [
      {
        // `stroke` colors the axis tick LABELS (the text); `ticks` is
        // the short perpendicular tick-mark segments; `grid` is the
        // background grid lines.
        stroke: () => cssVar("--axis-label"),
        ticks: { stroke: () => cssVar("--axis-tick") },
        grid: { stroke: () => cssVar("--border") },
      },
      {
        stroke: () => cssVar("--axis-label"),
        ticks: { stroke: () => cssVar("--axis-tick") },
        grid: { stroke: () => cssVar("--border") },
      },
    ],
    series: [
      {},
      {
        label: "jobs",
        stroke: () => cssVar("--accent"),
        width: 1,
        fill: "rgba(59, 130, 246, 0.55)",
        // Render each bucket as a discrete bar instead of a connected
        // line — for hourly aggregates a line was misleading (implied
        // continuous data) and a single populated bucket disappeared
        // into a hairline spike at 1/24th of the pinned x-axis.
        paths: uPlot.paths.bars({ size: [0.85, 18] }),
        points: { show: false },
      },
    ],
    scales: { x: { time: true } },
  };
  const plot = new uPlot(opts, emptyTimelineRange(hours), container);
  // Track container size so the chart resizes when the pane is unhidden
  // (x-show flip), when the window resizes, or when the column widens.
  // Covers the "rendered at half width" race where init ran before
  // layout had a real width.
  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = Math.floor(entry.contentRect.width);
        if (width <= 0) continue;
        const height = container.clientHeight || initHeight;
        if (plot.width === width && plot.height === height) continue;
        plot.setSize({ width, height });
      }
    });
    ro.observe(container);
  }
  return plot;
}

function applyTimelineData(plot, buckets, range) {
  if (!plot) return;
  const xs = buckets.map((b) =>
    Math.floor(new Date(b.bucket_start).getTime() / 1000),
  );
  const ys = buckets.map((b) => b.count);
  // uPlot needs at least 2 points; pad with current time if only one.
  if (xs.length === 0) {
    if (range && range.xMin != null && range.xMax != null) {
      plot.setData([[range.xMin, range.xMax], [0, 0]]);
      return;
    }
    plot.setData(emptyTimelineRange());
    return;
  }
  if (xs.length === 1) {
    xs.push(xs[0] + 3600);
    ys.push(0);
  }
  plot.setData([xs, ys]);
  // Pinning the x scale here is what keeps every chart on the perf
  // page aligned to the same window, even when individual tiers cover
  // sparser ranges than the overall timeline.
  if (range && range.xMin != null && range.xMax != null) {
    plot.setScale("x", { min: range.xMin, max: range.xMax });
  }
  // Pin the y-axis too so per-tier charts share the overall chart's
  // y scale. Without this each chart auto-scales independently and a
  // tier with count=8 ends up the same visual height as one with 80.
  if (range && range.yMax != null) {
    plot.setScale("y", { min: 0, max: range.yMax });
  }
}

// ============== Alpine root component ==============

function dashboard() {
  return {
    view: "home",
    theme: localStorage.getItem("retriever-theme") || "dark",
    data: {
      status: null,
      active: [],
      stuck: [],
      counts: null,
      timeline: [],
      failures: [],
    },
    activity: {
      mode: "Completed",
      lookback: "Last 24h",
      submitter: null,
      tier: null,
      sortField: "completed",
      sortDir: "desc",
      items: [],
      cursor: null,
      nextCursor: null,
      history: [],
      submitterOptions: [],
    },
    performance: {
      lookback: "Last 24h",
      status: "Complete",
      durations: null,
      tiers: [],
      timeline: [],
      // tier number -> list of bucket objects {bucket_start, count}.
      // Populated by fetchTierTimelines after the tier list arrives.
      tierTimelines: {},
      // Shared x-axis range applied to the main + every per-tier chart.
      // Computed once per fetchPerformance() and reused when individual
      // tier plots spin up lazily.
      xRange: null,
    },
    heatmaps: {
      lookback: "Last week",
      submitterTable: [], // /status/submitters
      tierTable: [], // /status/tiers
      submitterTier: [], // /status/submitter_tier_stats
      failuresBySubmitter: [], // /status/failure_breakdown?by=submitter
      failuresByTier: [], // /status/failure_breakdown?by=tier
    },
    modalStack: [],
    pollers: {},

    // Constants exposed for template iteration.
    ACTIVITY_MODES,
    LOOKBACK_OPTIONS,
    SORT_FIELDS,
    PERFORMANCE_STATUS_OPTIONS,
    FAILURE_STATUSES,
    TIERS,

    init() {
      // Apply persisted theme.
      document.documentElement.dataset.theme = this.theme;

      // URL hash → initial view + activity filter state.
      this.applyHashState();
      window.addEventListener("popstate", () => {
        this.applyHashState();
        this.lazyInitForView();
      });

      // Fire the same per-view lazy init that setView() would on click.
      // Without this, a hard refresh to #performance or #submitters
      // leaves the view empty until the user tab-switches and back.
      // For home this constructs the throughput plot lazily so it
      // sees a real container width (a hidden pane has clientWidth=0
      // and uPlot doesn't recover from being built into one).
      this.lazyInitForView();

      // First fetches (immediate) + recurring timers.
      this.startPolling();

      // Pause polls when the tab is hidden.
      document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
          this.stopPolling();
        } else {
          this.startPolling();
        }
      });
    },

    /** Idempotent: tears down any existing timers, fires immediate
     * refreshes, then sets up the recurring intervals. Safe to call
     * whenever we want "fresh data + active polling" — `init()` startup
     * and visibility-resume both use it. */
    startPolling() {
      this.stopPolling();
      this.refreshRealtime();
      this.refreshAggregates();
      this.pollers.realtime = window.setInterval(
        () => this.refreshRealtime(),
        REALTIME_INTERVAL_MS,
      );
      this.pollers.aggregate = window.setInterval(
        () => this.refreshAggregates(),
        AGGREGATE_INTERVAL_MS,
      );
    },

    stopPolling() {
      for (const k of Object.keys(this.pollers)) {
        window.clearInterval(this.pollers[k]);
        delete this.pollers[k];
      }
    },

    setView(name) {
      this.view = name;
      this.writeHashState();
      this.lazyInitForView();
    },

    /** Trigger the per-view first-fetch + chart init for the currently
     * active view. Idempotent — fetchers run on every call (the user
     * gets fresh data on tab switch), but plot init is gated by
     * `ensurePerformancePlot`'s null check. Called from both setView()
     * (tab click) and init() (hard refresh) so refresh-to-perf works. */
    lazyInitForView() {
      if (this.view === "home") this.ensureThroughputPlot();
      if (this.view === "activity") this.fetchActivity();
      if (this.view === "performance") {
        this.ensurePerformancePlot();
        this.fetchPerformance();
      }
      if (this.view === "heatmaps") this.fetchHeatmaps();
    },

    /** Lazy-construct the Home throughput chart once both uPlot is
     * loaded and its container has real layout dimensions. uPlot
     * built into a zero-width container renders a degenerate canvas
     * that doesn't redraw cleanly on later resize — and the
     * [x-cloak] CSS rule means EVERY container has clientWidth=0
     * during init(), so we have to wait it out. */
    ensureThroughputPlot() {
      if (throughputPlot) {
        // Already built — push current data through so a tab-back also
        // refreshes the chart with whatever the latest poll fetched.
        this.applyHomeTimeline();
        return;
      }
      retryUntilBuilt(() => {
        if (throughputPlot) return true;
        if (typeof uPlot === "undefined") return false;
        const container = document.getElementById("throughput-chart");
        if (!container || container.clientWidth === 0) return false;
        throughputPlot = makeTimelinePlot(container, {
          hours: TIMELINE_LOOKBACK_HOURS,
        });
        this.applyHomeTimeline();
        return true;
      });
    },

    applyHomeTimeline() {
      if (!throughputPlot) return;
      const buckets = this.data.timeline || [];
      applyTimelineData(
        throughputPlot,
        buckets,
        computeTimelineRange(TIMELINE_LOOKBACK_HOURS, buckets),
      );
    },

    toggleTheme() {
      this.theme = this.theme === "dark" ? "light" : "dark";
      localStorage.setItem("retriever-theme", this.theme);
      document.documentElement.dataset.theme = this.theme;
      // Axis/grid/series colors are read from CSS vars on every uPlot
      // redraw, so a forced redraw is enough to pick up the new theme.
      redrawAllPlots();
    },

    jumpToActivity({ mode, lookback, submitter, tier } = {}) {
      if (mode && ACTIVITY_MODES.includes(mode)) this.activity.mode = mode;
      if (lookback && LOOKBACK_HOURS[lookback] !== undefined)
        this.activity.lookback = lookback;
      this.activity.submitter = submitter || null;
      this.activity.tier = tier === 0 || tier === 1 ? tier : null;
      this.activity.cursor = null;
      this.activity.history = [];
      this.view = "activity";
      this.writeHashState();
      this.fetchActivity();
    },

    // ============== URL hash <-> state ==============

    applyHashState() {
      const hash = (location.hash || "#home").replace(/^#/, "");
      const [viewPart, qs = ""] = hash.split("?");
      this.view = VIEWS.includes(viewPart) ? viewPart : "home";
      const params = new URLSearchParams(qs);

      if (this.view === "activity") {
        const mode = params.get("mode");
        if (mode && ACTIVITY_MODES.includes(mode)) this.activity.mode = mode;
        const lb = params.get("window");
        if (lb && LOOKBACK_HOURS[lb] !== undefined)
          this.activity.lookback = lb;
        this.activity.submitter = params.get("submitter") || null;
        const tierRaw = params.get("tier");
        const tier = tierRaw == null || tierRaw === "" ? NaN : Number(tierRaw);
        this.activity.tier = TIER_VALUES.has(tier) ? tier : null;
        const sortField = params.get("sort");
        if (sortField && SORT_FIELD_VALUES.has(sortField))
          this.activity.sortField = sortField;
        const dir = params.get("dir");
        if (dir === "asc" || dir === "desc") this.activity.sortDir = dir;
        this.activity.cursor = null;
        this.activity.history = [];
      } else if (this.view === "performance") {
        const lb = params.get("window");
        if (lb && LOOKBACK_HOURS[lb] !== undefined)
          this.performance.lookback = lb;
        const status = params.get("status");
        if (status && PERFORMANCE_STATUS_VALUES.has(status))
          this.performance.status = status;
      } else if (this.view === "heatmaps") {
        const lb = params.get("window");
        if (lb && LOOKBACK_HOURS[lb] !== undefined)
          this.heatmaps.lookback = lb;
      }
    },

    writeHashState() {
      let hash = `#${this.view}`;
      const p = new URLSearchParams();
      if (this.view === "activity") {
        if (this.activity.mode !== "Completed") p.set("mode", this.activity.mode);
        if (this.activity.lookback !== "Last 24h")
          p.set("window", this.activity.lookback);
        if (this.activity.submitter) p.set("submitter", this.activity.submitter);
        if (this.activity.tier !== null) p.set("tier", String(this.activity.tier));
        if (this.activity.sortField !== "completed")
          p.set("sort", this.activity.sortField);
        if (this.activity.sortDir !== "desc")
          p.set("dir", this.activity.sortDir);
      } else if (this.view === "performance") {
        if (this.performance.lookback !== "Last 24h")
          p.set("window", this.performance.lookback);
        if (this.performance.status !== "Complete")
          p.set("status", this.performance.status);
      } else if (this.view === "heatmaps") {
        if (this.heatmaps.lookback !== "Last week")
          p.set("window", this.heatmaps.lookback);
      }
      const qs = p.toString();
      if (qs) hash += `?${qs}`;
      const target = `${location.pathname}${location.search}${hash}`;
      if (location.hash !== hash) history.replaceState(null, "", target);
    },

    // ============== Activity view ==============

    isLiveMode() {
      return ACTIVITY_LIVE_MODES.has(this.activity.mode);
    },

    setActivityMode(mode) {
      if (this.activity.mode === mode) return;
      this.activity.mode = mode;
      this.activity.cursor = null;
      this.activity.history = [];
      this.writeHashState();
      this.fetchActivity();
    },

    setActivityLookback(value) {
      this.activity.lookback = value;
      this.activity.cursor = null;
      this.activity.history = [];
      this.writeHashState();
      this.fetchActivity();
    },

    setActivitySubmitter(value) {
      this.activity.submitter = value || null;
      this.activity.cursor = null;
      this.activity.history = [];
      this.writeHashState();
      this.fetchActivity();
    },

    setActivityTier(value) {
      this.activity.tier = value === "" ? null : Number(value);
      this.activity.cursor = null;
      this.activity.history = [];
      this.writeHashState();
      this.fetchActivity();
    },

    setActivitySortField(value) {
      if (!SORT_FIELD_VALUES.has(value)) return;
      this.activity.sortField = value;
      this.activity.cursor = null;
      this.activity.history = [];
      this.writeHashState();
      // Live modes sort client-side over the in-memory list; sortedItems()
      // re-reads sortField on every render, so no refetch needed.
      if (!this.isLiveMode()) this.fetchActivity();
    },

    toggleActivitySortDir() {
      this.activity.sortDir =
        this.activity.sortDir === "asc" ? "desc" : "asc";
      this.activity.cursor = null;
      this.activity.history = [];
      this.writeHashState();
      if (!this.isLiveMode()) this.fetchActivity();
    },

    /**
     * Click a column header to sort by that field. Same field → flip
     * direction; new field → switch + default to desc.
     */
    headerSortClick(field) {
      if (!SORT_FIELD_VALUES.has(field)) return;
      if (this.activity.sortField === field) {
        this.toggleActivitySortDir();
        return;
      }
      this.activity.sortField = field;
      this.activity.sortDir = "desc";
      this.activity.cursor = null;
      this.activity.history = [];
      this.writeHashState();
      if (!this.isLiveMode()) this.fetchActivity();
    },

    /** Sort-arrow indicator for a column header ("", "↑", or "↓"). */
    headerArrow(field) {
      if (this.activity.sortField !== field) return "";
      return this.activity.sortDir === "asc" ? "↑" : "↓";
    },

    /**
     * Which sort field a given column-header click should map to. The
     * "When" column is the contextual one — `created` for live modes,
     * `completed` for terminal ones.
     */
    whenSortField() {
      return this.isLiveMode() ? "created" : "completed";
    },

    /**
     * Outcome column maps to `status` in Failed mode, `results` elsewhere.
     */
    outcomeSortField() {
      return this.activity.mode === "Failed" ? "status" : "results";
    },

    async activityNext() {
      if (!this.activity.nextCursor) return;
      this.activity.history.push(this.activity.cursor);
      this.activity.cursor = this.activity.nextCursor;
      await this.fetchActivity();
    },

    async activityPrev() {
      if (this.activity.history.length === 0) return;
      this.activity.cursor = this.activity.history.pop();
      await this.fetchActivity();
    },

    async fetchActivity() {
      const mode = this.activity.mode;
      try {
        if (ACTIVITY_LIVE_MODES.has(mode)) {
          // Active/Stuck — flat list, no cursor, no server-side filters.
          // Sort is applied client-side over the in-memory list.
          const path = mode === "Active" ? "/status/active" : "/status/stuck";
          const rows = await fetchJson(path);
          this.activity.items = rows;
          this.activity.nextCursor = null;
          return;
        }
        const endpoint =
          mode === "Failed" ? "/status/failed" : "/status/completed";
        const p = new URLSearchParams({
          limit: String(ACTIVITY_PAGE_SIZE),
          sort: this.activity.sortField,
          direction: this.activity.sortDir,
        });
        if (this.activity.cursor) p.set("cursor", this.activity.cursor);
        const lookback = LOOKBACK_HOURS[this.activity.lookback];
        if (lookback !== null) p.set("lookback", String(lookback));
        if (this.activity.submitter) p.set("submitter", this.activity.submitter);
        if (this.activity.tier !== null)
          p.set("data_tier", String(this.activity.tier));
        const page = await fetchJson(`${endpoint}?${p}`);
        this.activity.items = page.items || [];
        this.activity.nextCursor = page.next_cursor || null;
      } catch (err) {
        console.warn("activity fetch failed:", err);
        this.activity.items = [];
        this.activity.nextCursor = null;
      }
    },

    async populateSubmitterOptions() {
      try {
        const rows = await fetchJson("/status/submitters?top=200");
        this.activity.submitterOptions = rows
          .map((r) => r.submitter)
          .filter((s) => typeof s === "string" && s.length > 0);
      } catch (err) {
        console.warn("submitter list fetch failed:", err);
      }
    },

    sortedItems() {
      // Terminal modes are pre-sorted server-side; just hand back the
      // current page. Live modes (Active/Stuck) need a client-side sort
      // over the in-memory list.
      if (!this.isLiveMode()) return this.activity.items;
      const items = [...this.activity.items];
      const field = this.activity.sortField;
      const sign = this.activity.sortDir === "asc" ? 1 : -1;
      items.sort((a, b) => {
        const av = this.sortKey(a, field);
        const bv = this.sortKey(b, field);
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        if (av < bv) return -sign;
        if (av > bv) return sign;
        return 0;
      });
      return items;
    },

    sortKey(row, field) {
      if (field === "created" || field === "completed") {
        const iso = row[field];
        return iso ? new Date(iso).getTime() : null;
      }
      if (field === "duration") {
        return this.isLiveMode()
          ? row.age_seconds ?? null
          : row.duration_seconds ?? null;
      }
      if (field === "submitter") return (row.submitter || "").toLowerCase();
      if (field === "status") return (row.status || "").toLowerCase();
      if (field === "results") return row.results ?? null;
      return null;
    },

    activityWhen(row) {
      if (ACTIVITY_LIVE_MODES.has(this.activity.mode)) {
        return relTime(row.created);
      }
      return relTime(row.completed);
    },

    activityDuration(row) {
      if (ACTIVITY_LIVE_MODES.has(this.activity.mode)) {
        return formatDurationSeconds(row.age_seconds ?? null);
      }
      return formatDurationSeconds(row.duration_seconds ?? null);
    },

    activityOutcome(row) {
      const m = this.activity.mode;
      if (m === "Active") return "running";
      if (m === "Stuck") return "stuck";
      if (m === "Failed") return row.status || "";
      const r = row.results;
      return r != null ? `${r} results` : "—";
    },

    activityStatusText() {
      const n = this.activity.items.length;
      const noun = n === 1 ? "job" : "jobs";
      if (this.isLiveMode()) return `Showing ${n} ${noun}`;
      if (this.activity.nextCursor)
        return `Showing ${n} ${noun} · more available`;
      return `Showing ${n} ${noun} · end of results`;
    },

    async refreshRealtime() {
      try {
        const [status, active, stuck] = await Promise.all([
          fetchJson("/status"),
          fetchJson("/status/active"),
          fetchJson("/status/stuck"),
        ]);
        this.data.status = status;
        this.data.active = active;
        this.data.stuck = stuck;
      } catch (err) {
        console.warn("realtime refresh failed:", err);
      }
    },

    async refreshAggregates() {
      try {
        const [counts, timeline, failures] = await Promise.all([
          fetchJson("/status/counts"),
          fetchJson(
            `/status/timeline?field=completed&granularity=hour&lookback=${TIMELINE_LOOKBACK_HOURS}`,
          ),
          fetchJson(`/status/failed?limit=${RECENT_FAILURES_LIMIT}`),
        ]);
        this.data.counts = counts;
        this.data.timeline = timeline;
        this.data.failures = failures.items || [];
        applyTimelineData(
          throughputPlot,
          timeline,
          computeTimelineRange(TIMELINE_LOOKBACK_HOURS, timeline),
        );
      } catch (err) {
        console.warn("aggregate refresh failed:", err);
      }
      // Refresh the submitter dropdown on the same cadence so a
      // long-lived session picks up new submitters without a reload.
      // Failure here doesn't drag down the whole aggregate refresh.
      this.populateSubmitterOptions();
    },

    // ============== Performance view ==============

    /** Build the perf throughput uPlot lazily — its container only
     * exists after the user has switched to the perf tab at least once
     * (Alpine x-show keeps the element mounted after that). Safe to
     * call repeatedly; init is a no-op once `performancePlot` is set. */
    ensurePerformancePlot() {
      if (performancePlot) return;
      retryUntilBuilt(() => {
        if (performancePlot) return true;
        if (typeof uPlot === "undefined") return false;
        const container = document.getElementById("performance-chart");
        if (!container || container.clientWidth === 0) return false;
        performancePlot = makeTimelinePlot(container, {
          hours: this.performanceLookbackHours() ?? 24,
        });
        // fetchPerformance() may have resolved before the plot was
        // built; feed the cached timeline in immediately.
        if (this.performance.timeline.length > 0) {
          applyTimelineData(
            performancePlot,
            this.performance.timeline,
            this.performance.xRange,
          );
        }
        return true;
      });
    },

    performanceLookbackHours() {
      return LOOKBACK_HOURS[this.performance.lookback];
    },

    async fetchPerformance() {
      const lookback = this.performanceLookbackHours();
      const status = this.performance.status;
      try {
        const [durations, tiers, timeline] = await Promise.all([
          fetchJson(buildUrl("/status/durations", { status, lookback })),
          fetchJson(buildUrl("/status/tiers", { lookback })),
          fetchJson(
            buildUrl("/status/timeline", {
              field: "completed",
              granularity: "hour",
              lookback,
            }),
          ),
        ]);
        this.performance.durations = durations;
        this.performance.tiers = tiers;
        this.performance.timeline = timeline;
        // Compute the shared x-range for every chart on the page.
        // Lookback gives an exact window; "All time" derives it from
        // the union timeline so tier charts still line up.
        const range = computeTimelineRange(lookback, timeline);
        this.performance.xRange = range;
        applyTimelineData(performancePlot, timeline, range);
        // Per-tier timelines fan out in parallel after we know which
        // tiers showed activity.
        await this.fetchTierTimelines(tiers, range);
      } catch (err) {
        console.warn("performance fetch failed:", err);
      }
    },

    async fetchTierTimelines(tiers, range) {
      const lookback = this.performanceLookbackHours();
      const fetches = tiers.map(async (row) => {
        try {
          const buckets = await fetchJson(
            buildUrl("/status/timeline", {
              field: "completed",
              granularity: "hour",
              data_tier: row.tier,
              lookback,
            }),
          );
          this.performance.tierTimelines[row.tier] = buckets;
          // Plot may not be initialized yet if the container only just
          // entered the DOM; ensureTierPlot will pull data from
          // tierTimelines once it spins up.
          const plot = tierPlots.get(row.tier);
          if (plot) applyTimelineData(plot, buckets, range);
        } catch (err) {
          console.warn(`tier ${row.tier} timeline fetch failed:`, err);
        }
      });
      await Promise.all(fetches);
    },

    /** Lazy-init the chart for a tier pane. Called via x-init on each
     * tier pane's chart container, which fires both on first render and
     * any time Alpine re-renders the pane. */
    ensureTierPlot(tier) {
      if (tierPlots.has(tier)) return;
      retryUntilBuilt(() => {
        if (tierPlots.has(tier)) return true;
        if (typeof uPlot === "undefined") return false;
        const container = document.getElementById(
          `performance-chart-tier-${tier}`,
        );
        if (!container || container.clientWidth === 0) return false;
        const hours = this.performanceLookbackHours() ?? 24;
        const plot = makeTimelinePlot(container, { hours });
        tierPlots.set(tier, plot);
        // Buckets may already be in state if the fetch finished
        // before x-init fired — feed them in immediately, with the
        // same shared range the main + sibling tier plots use.
        const buckets = this.performance.tierTimelines[tier];
        if (buckets) applyTimelineData(plot, buckets, this.performance.xRange);
        return true;
      });
    },

    setPerformanceLookback(value) {
      if (!LOOKBACK_OPTIONS.includes(value)) return;
      this.performance.lookback = value;
      this.writeHashState();
      this.fetchPerformance();
    },

    setPerformanceStatus(value) {
      if (!PERFORMANCE_STATUS_VALUES.has(value)) return;
      this.performance.status = value;
      this.writeHashState();
      this.fetchPerformance();
    },

    // ============== Heatmaps view ==============

    async fetchHeatmaps() {
      const lookback = LOOKBACK_HOURS[this.heatmaps.lookback];
      try {
        const [submitters, tiers, st, failsSub, failsTier] = await Promise.all([
          fetchJson(buildUrl("/status/submitters", { top: 100, lookback })),
          fetchJson(buildUrl("/status/tiers", { lookback })),
          fetchJson(buildUrl("/status/submitter_tier_stats", { lookback })),
          fetchJson(
            buildUrl("/status/failure_breakdown", { by: "submitter", lookback }),
          ),
          fetchJson(
            buildUrl("/status/failure_breakdown", { by: "tier", lookback }),
          ),
        ]);
        this.heatmaps.submitterTable = submitters;
        this.heatmaps.tierTable = tiers;
        this.heatmaps.submitterTier = st;
        this.heatmaps.failuresBySubmitter = failsSub.rows || [];
        this.heatmaps.failuresByTier = failsTier.rows || [];
      } catch (err) {
        console.warn("heatmaps fetch failed:", err);
      }
    },

    setHeatmapsLookback(value) {
      if (!LOOKBACK_OPTIONS.includes(value)) return;
      this.heatmaps.lookback = value;
      this.writeHashState();
      this.fetchHeatmaps();
    },

    /**
     * Build the (submitter × tier) heatmap matrix for one of two
     * metrics: `failed_pct` (failed/count) or `p95_seconds`. Returns a
     * single flat array of grid items so the HTML template can render
     * it with one x-for and the CSS grid auto-flows the layout.
     */
    submitterTierMatrix(metric) {
      const subRows = this.heatmaps.submitterTable.map((r) => r.submitter);
      const cols = TIERS;
      const byCell = new Map();
      for (const r of this.heatmaps.submitterTier) {
        byCell.set(`${r.submitter}|${r.tier}`, r);
      }
      const values = [];
      const cellGrid = subRows.map((sub) =>
        cols.map((tier) => {
          const r = byCell.get(`${sub}|${tier}`);
          if (!r) return { value: null, display: "—" };
          if (metric === "failed_pct") {
            if (r.count === 0) return { value: null, display: "—" };
            const v = (r.failed / r.count) * 100;
            values.push(v);
            return { value: v, display: `${v.toFixed(0)}%`, count: r.count };
          }
          if (r.p95_seconds == null) return { value: null, display: "—" };
          values.push(r.p95_seconds);
          return {
            value: r.p95_seconds,
            display: formatDurationSeconds(r.p95_seconds),
            count: r.count,
          };
        }),
      );
      const vmin = values.length ? Math.min(...values) : 0;
      const vmax = values.length ? Math.max(...values) : 0;
      const scale = metric === "failed_pct" ? "fail" : "duration";
      const flat = [];
      // Top-left corner.
      flat.push({ kind: "corner" });
      // Column headers.
      for (const t of cols) {
        flat.push({ kind: "colHeader", text: `tier ${t}` });
      }
      // Each data row: row label then its cells.
      subRows.forEach((sub, i) => {
        flat.push({ kind: "rowHeader", text: sub });
        cellGrid[i].forEach((c) => {
          flat.push({
            kind: "cell",
            value: c.value,
            display: c.display,
            color: heatmapColor(c.value, vmin, vmax, scale),
            title:
              c.value == null
                ? `${sub}: no data`
                : `${sub}: ${c.display}${c.count != null ? ` (n=${c.count})` : ""}`,
          });
        });
      });
      return { columnCount: cols.length, items: flat };
    },

    /**
     * Build the (failure-status × {submitter|tier}) count matrix. Same
     * flat-array shape as `submitterTierMatrix`.
     */
    failuresMatrix(by) {
      const raw =
        by === "submitter"
          ? this.heatmaps.failuresBySubmitter
          : this.heatmaps.failuresByTier;
      const colSet = new Set();
      const byCell = new Map();
      for (const r of raw) {
        colSet.add(r.key);
        byCell.set(`${r.status}|${r.key}`, r.count);
      }
      const cols = [...colSet].sort();
      const values = raw.map((r) => r.count);
      const vmin = values.length ? Math.min(...values) : 0;
      const vmax = values.length ? Math.max(...values) : 0;
      const flat = [];
      flat.push({ kind: "corner" });
      for (const col of cols) {
        flat.push({
          kind: "colHeader",
          text: by === "tier" ? `tier ${col}` : col,
        });
      }
      for (const status of FAILURE_STATUSES) {
        flat.push({ kind: "rowHeader", text: status });
        for (const col of cols) {
          const v = byCell.get(`${status}|${col}`) ?? 0;
          flat.push({
            kind: "cell",
            value: v === 0 ? null : v,
            display: v === 0 ? "—" : String(v),
            color: heatmapColor(v === 0 ? null : v, vmin, vmax, "count"),
            title:
              v === 0
                ? `${status} × ${by === "tier" ? `tier ${col}` : col}: 0`
                : `${status} × ${by === "tier" ? `tier ${col}` : col}: ${v}`,
          });
        }
      }
      return { columnCount: cols.length, items: flat };
    },

    // ============== Derived / formatting accessors ==============

    versionLabel() {
      const v = this.data.status?.version;
      if (!v) return "";
      return v.git_branch ? `${v.git_commit}@${v.git_branch}` : v.git_commit;
    },

    last24Completed() {
      const c = this.data.counts?.windows?.last_24h?.counts || {};
      return c.Complete ?? 0;
    },

    last24Failed() {
      const c = this.data.counts?.windows?.last_24h?.counts || {};
      let total = 0;
      for (const s of TERMINAL_FAILURE) total += c[s] || 0;
      return total;
    },

    relTime(iso) {
      return relTime(iso);
    },

    freshnessCount(rec) {
      if (!rec) return "—";
      return `${rec.count} entries`;
    },

    freshnessSub(rec) {
      if (!rec) return "not yet published";
      return `refreshed ${relTime(rec.refreshed_at)}`;
    },

    mongoDetail() {
      const m = this.data.status?.mongo;
      if (!m) return "—";
      const bits = [];
      if (m.storage_mb != null) bits.push(`${formatMb(m.storage_mb)} MB storage`);
      if (m.queue_depth != null) bits.push(`queue ${m.queue_depth}`);
      return bits.length ? bits.join(" · ") : "connected";
    },

    redisDetail() {
      const r = this.data.status?.redis;
      if (!r) return "—";
      if (r.used_memory_mb != null) return `${formatMb(r.used_memory_mb)} MB memory`;
      return "connected";
    },

    tierRow(n) {
      const tiers = this.data.status?.tiers || [];
      return tiers.find((t) => t.tier === n);
    },

    processesEmptyHint() {
      if (this.data.status === null) return "Loading…";
      if (!this.data.status.processes)
        return "No `processes` field in /status response.";
      const p = this.data.status.processes;
      const empty =
        !p.main && !p.background && (p.workers ?? []).length === 0;
      return empty
        ? "Process registry empty — check Redis HGETALL {Retriever}:workers."
        : "";
    },

    processRows() {
      const p = this.data.status?.processes;
      if (!p) return [];
      const out = [];
      if (p.main) {
        out.push({
          role: "main",
          pid: p.main.pid,
          uptime: formatUptime(p.main.uptime_seconds),
          rss: formatMb(p.main.rss_mb),
        });
      }
      if (p.background) {
        out.push({
          role: "background",
          pid: p.background.pid,
          uptime: formatUptime(p.background.uptime_seconds),
          rss: formatMb(p.background.rss_mb),
        });
      }
      (p.workers || []).forEach((w, i) => {
        out.push({
          role: `worker [${i}]`,
          pid: w.pid,
          uptime: formatUptime(w.uptime_seconds),
          rss: formatMb(w.rss_mb),
        });
      });
      return out;
    },

    // ============== Modal stack ==============

    openModal(m) {
      this.modalStack.push(m);
      if (m.kind === "job") {
        // Lazy-load the per-job metadata and log preview in parallel.
        // No `lookback` on the logs query — job logs almost always span a
        // short window so the time filter would just risk dropping
        // useful lines.
        const idx = this.modalStack.length - 1;
        const target = this.modalStack[idx];
        fetchJson(`/status/jobs/${encodeURIComponent(m.jobId)}`)
          .then((detail) => {
            target.detail = detail;
          })
          .catch((err) => {
            console.warn("job detail fetch failed:", err);
          });
        fetchText(`/logs?job_id=${encodeURIComponent(m.jobId)}&fmt=flat`)
          .then((text) => {
            target.logs = truncateLogLines(text);
          })
          .catch((err) => {
            target.logs = `Failed to load logs: ${err.message || err}`;
          });
      }
    },

    popModal() {
      this.modalStack.pop();
    },

    closeModal() {
      this.modalStack = [];
    },

    modalTitle(m) {
      // Job modals build their own title inline (with a click-to-copy code
      // chip); this is the fallback used for any other modal kind.
      if (m.kind === "job") return `Job ${m.jobId}`;
      return "Details";
    },

    async copyJobId(jobId) {
      try {
        await navigator.clipboard.writeText(jobId);
      } catch (err) {
        console.warn("copy failed:", err);
      }
    },

    localTime(iso) {
      return formatLocal(iso);
    },

    fmtDuration(seconds) {
      return formatDurationSeconds(seconds);
    },

    fmtBool(v) {
      if (v == null) return "—";
      return v ? "yes" : "no";
    },

    fmtNum(v) {
      return v == null ? "—" : v;
    },

    fmtTiers(arr) {
      return !arr || arr.length === 0 ? "—" : arr.join(", ");
    },

    /** Failure percentage over terminal jobs (completed + failed).
     * Returns "—" when no terminal jobs have happened in the window so
     * the cell doesn't read as "0% (false negative)". */
    failPct(failed, completed) {
      const terminal = (failed || 0) + (completed || 0);
      if (terminal === 0) return "—";
      return `${((failed / terminal) * 100).toFixed(0)}%`;
    },
  };
}

// Expose so the inline `x-data="dashboard()"` finds it.
window.dashboard = dashboard;
