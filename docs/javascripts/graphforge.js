(function () {
  "use strict";

  const frameSelector = "iframe[data-gf-report-frame]";

  function currentTheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate"
      ? "dark"
      : "light";
  }

  function reportFrames() {
    return Array.from(document.querySelectorAll(frameSelector));
  }

  function syncReportTheme(frame) {
    if (frame.contentWindow) {
      frame.contentWindow.postMessage(
        { source: "graphforge-docs", type: "theme", value: currentTheme() },
        window.location.origin
      );
    }
  }

  function initializePage() {
    reportFrames().forEach(function (frame) {
      if (frame.dataset.gfBound !== "true") {
        frame.dataset.gfBound = "true";
        frame.addEventListener("load", function () {
          syncReportTheme(frame);
        });
      }
      syncReportTheme(frame);
    });
  }

  window.addEventListener("message", function (event) {
    if (event.origin !== window.location.origin || !event.data) return;
    if (event.data.source !== "graphforge-report") return;
    const frame = reportFrames().find(function (candidate) {
      return candidate.contentWindow === event.source;
    });
    if (!frame) return;
    if (event.data.type === "height") {
      const height = Math.max(520, Math.min(900, Number(event.data.value) || 0));
      frame.style.height = height + "px";
    }
    if (event.data.type === "fallback") {
      frame.closest("[data-gf-report]")?.classList.add("gf-report-embed--fallback");
    }
  });

  function watchTheme() {
    if (!document.body || document.body.dataset.gfThemeObserver === "true") return;
    document.body.dataset.gfThemeObserver = "true";
    new MutationObserver(function (mutations) {
      if (mutations.some(function (entry) {
        return entry.attributeName === "data-md-color-scheme";
      })) {
        reportFrames().forEach(syncReportTheme);
      }
    }).observe(document.body, { attributes: true });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
      initializePage();
      watchTheme();
    });
  } else {
    document.addEventListener("DOMContentLoaded", function () {
      initializePage();
      watchTheme();
    });
  }
})();
