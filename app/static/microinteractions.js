(function () {
  if (window.RexAIMicro) {
    return;
  }

  const AudioContextRef = window.AudioContext || window.webkitAudioContext;
  const supportsVibration = typeof navigator !== "undefined" && typeof navigator.vibrate === "function";

  function createAudioEngine() {
    let ctx;
    return {
      play(type) {
        if (!AudioContextRef) {
          return;
        }
        try {
          if (!ctx) {
            ctx = new AudioContextRef();
          }
          const now = ctx.currentTime;
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          const baseFreq = type === "success" ? 620 : type === "hover" ? 480 : 540;
          osc.frequency.value = baseFreq + Math.random() * 40;
          osc.type = "sine";
          gain.gain.value = 0.0001;
          gain.gain.linearRampToValueAtTime(0.09, now + 0.02);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.25);
          osc.connect(gain).connect(ctx.destination);
          osc.start(now);
          osc.stop(now + 0.3);
        } catch (err) {
          // Ignore audio errors (autoplay policies, etc.)
        }
      },
    };
  }

  const audioEngine = createAudioEngine();

  function vibrate(pattern) {
    if (!supportsVibration || !pattern || !pattern.length) {
      return;
    }
    try {
      navigator.vibrate(pattern);
    } catch (err) {
      // noop
    }
  }

  function spawnParticles(layer, palette, opts = {}) {
    if (!layer) {
      return;
    }
    const intensity = opts.burst ? 12 : 6;
    for (let i = 0; i < intensity; i += 1) {
      const particle = document.createElement("span");
      particle.className = "rexai-particle";
      const angle = Math.random() * Math.PI * 2;
      const distance = (opts.burst ? 80 : 46) + Math.random() * (opts.burst ? 50 : 26);
      const size = 3 + Math.random() * 6;
      const color = palette[Math.floor(Math.random() * palette.length)] || "rgba(96,165,250,0.9)";
      particle.style.setProperty("--dx", `${Math.cos(angle) * distance}px`);
      particle.style.setProperty("--dy", `${Math.sin(angle) * distance}px`);
      particle.style.setProperty("--size", `${size}px`);
      particle.style.background = color;
      particle.style.opacity = "0";
      if (opts.origin) {
        particle.style.left = `${opts.origin[0]}px`;
        particle.style.top = `${opts.origin[1]}px`;
      }
      layer.appendChild(particle);
      requestAnimationFrame(() => {
        particle.style.opacity = "1";
        particle.style.transform = "translate(var(--dx), var(--dy)) scale(1)";
      });
      setTimeout(() => {
        if (particle.parentNode) {
          particle.parentNode.removeChild(particle);
        }
      }, opts.burst ? 720 : 520);
    }
  }

  function mount(root, config) {
    const button = root.querySelector("button");
    const label = root.querySelector(".rexai-fx-label");
    const status = root.querySelector(".rexai-fx-status");
    const particles = root.querySelector(".rexai-fx-particles");
    const stateMessages = config.stateMessages || {};
    const palette = config.particleColors || [
      "rgba(96,165,250,0.9)",
      "rgba(147,197,253,0.9)",
      "rgba(244,114,182,0.8)",
      "rgba(129,140,248,0.8)",
    ];

    if (!button) {
      return null;
    }

    let currentState = config.state || "idle";

    function applyState(nextState) {
      currentState = nextState;
      root.setAttribute("data-state", nextState);
      const nextLabel = stateMessages[nextState] || stateMessages.idle || button.dataset.baseLabel || button.textContent;
      if (label) {
        label.textContent = nextLabel;
      } else {
        button.textContent = nextLabel;
      }
      if (status) {
        status.textContent = config.statusHints && config.statusHints[nextState] ? config.statusHints[nextState] : "";
        status.setAttribute("data-active", status.textContent ? "true" : "false");
      }
      if (nextState === "loading") {
        button.setAttribute("aria-busy", "true");
      } else {
        button.removeAttribute("aria-busy");
      }
    }

    button.dataset.baseLabel = button.textContent;
    applyState(currentState);

    button.addEventListener("pointerenter", () => {
      if (config.sound !== false) {
        audioEngine.play("hover");
      }
      spawnParticles(particles, palette, { burst: false });
    });

    button.addEventListener("pointerdown", (event) => {
      const rect = button.getBoundingClientRect();
      const origin = [event.clientX - rect.left, event.clientY - rect.top];
      spawnParticles(particles, palette, { burst: true, origin });
      if (config.vibration) {
        vibrate(config.vibrationPattern || [8, 12]);
      }
      if (config.sound !== false) {
        audioEngine.play("press");
      }
    });

    function handleSuccessTone() {
      if (config.sound !== false) {
        audioEngine.play("success");
      }
      if (config.vibration && currentState === "success") {
        vibrate(config.vibrationPattern || [12, 40]);
      }
    }

    return {
      applyState,
      handleSuccessTone,
    };
  }

  window.RexAIMicro = {
    mount(rootOrId, config) {
      const root = typeof rootOrId === "string" ? document.getElementById(rootOrId) : rootOrId;
      if (!root) {
        return null;
      }
      return mount(root, config);
    },
  };
})();
