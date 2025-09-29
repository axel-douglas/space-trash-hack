(function () {
  const GLOBAL_KEY = "RexAIInteractions";
  const FLAG_ATTR = "data-rexai-interactions";
  const REVEAL_TOKEN = "reveal";
  const TARGET_SELECTOR = ".reveal";

  const existing = window[GLOBAL_KEY] || {};
  if (existing.__revealBootstrapLoaded) {
    return;
  }

  existing.__revealBootstrapLoaded = true;
  window[GLOBAL_KEY] = existing;

  function tokenized(value) {
    return (value || "")
      .split(/\s+/)
      .map((token) => token.trim())
      .filter(Boolean);
  }

  function hasRevealFlag() {
    const nodes = document.querySelectorAll(`[${FLAG_ATTR}]`);
    for (const node of nodes) {
      if (tokenized(node.getAttribute(FLAG_ATTR)).includes(REVEAL_TOKEN)) {
        return true;
      }
    }
    return false;
  }

  function observeRevealTargets(observer, root) {
    const scope = root || document;
    scope.querySelectorAll(TARGET_SELECTOR).forEach((element) => {
      observer.observe(element);
    });
  }

  function createRevealObserver() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.2 }
    );
    observeRevealTargets(observer, document);
    return observer;
  }

  function watchNewRevealTargets(observer) {
    if (!document.body) {
      return;
    }
    const watcher = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (!(node instanceof HTMLElement)) {
            return;
          }
          if (node.matches(TARGET_SELECTOR)) {
            observer.observe(node);
          }
          node
            .querySelectorAll?.(TARGET_SELECTOR)
            .forEach((child) => observer.observe(child));
        });
      });
    });
    watcher.observe(document.body, { childList: true, subtree: true });
  }

  function enableReveal() {
    if (existing.revealInitialized) {
      return;
    }

    existing.revealInitialized = true;

    const start = () => {
      const observer = createRevealObserver();
      watchNewRevealTargets(observer);
    };

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", start, { once: true });
    } else {
      start();
    }
  }

  existing.enableReveal = enableReveal;

  function checkAndEnable() {
    if (hasRevealFlag()) {
      enableReveal();
    }
  }

  const flagObserver = new MutationObserver(checkAndEnable);
  flagObserver.observe(document.documentElement, {
    subtree: true,
    attributes: true,
    attributeFilter: [FLAG_ATTR],
    childList: true,
  });

  document.addEventListener("DOMContentLoaded", checkAndEnable);
  checkAndEnable();
})();
