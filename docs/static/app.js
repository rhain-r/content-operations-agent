// Video Checker dashboard. Plain JavaScript, no build step. All wording about results comes
// from the server; this file only handles uploading, polling, and moving the video around.
(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function clock(totalSeconds) {
    const seconds = Math.max(0, Math.floor(totalSeconds));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }

  function megabytes(bytes) {
    const mb = bytes / (1024 * 1024);
    return mb >= 10 ? `${Math.round(mb)} MB` : `${mb.toFixed(1)} MB`;
  }

  // --- Upload (home page) ------------------------------------------------------------------
  function initUpload() {
    const form = $("#upload-form");
    if (!form) return;
    const input = $("#video-file", form);
    const dropzone = $("[data-dropzone]", form);
    const chooseButton = $("[data-choose]", form);
    const fileName = $("[data-file-name]", form);
    const errorBox = $("[data-upload-error]", form);
    const progress = $("[data-upload-progress]", form);
    const bar = $("progress", progress);
    const status = $("[data-upload-status]", progress);
    const submit = $("[data-submit]", form);
    const maxBytes = Number(form.dataset.maxBytes);
    let chosen = null;

    const showError = (message) => {
      errorBox.textContent = message;
      errorBox.hidden = false;
    };
    const clearError = () => {
      errorBox.hidden = true;
      errorBox.textContent = "";
    };
    const tooBig = (file) =>
      `This file is ${megabytes(file.size)}, and the largest file we can take is ` +
      `${form.dataset.maxSize}. Please export a smaller file.`;

    function choose(file) {
      clearError();
      chosen = file || null;
      dropzone.classList.toggle("has-file", Boolean(chosen));
      if (!chosen) {
        fileName.hidden = true;
        return;
      }
      fileName.textContent = `Ready to check: ${chosen.name} (${megabytes(chosen.size)})`;
      fileName.hidden = false;
      if (chosen.size > maxBytes) showError(tooBig(chosen));
    }

    function setBusy(busy) {
      submit.disabled = busy;
      chooseButton.disabled = busy;
      progress.hidden = !busy;
      if (busy) {
        bar.value = 0;
        status.textContent = "Uploading… 0%";
      }
    }

    chooseButton.addEventListener("click", () => input.click());
    input.addEventListener("change", () => choose(input.files[0]));
    for (const type of ["dragenter", "dragover"]) {
      dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        if (!form.querySelector("fieldset").disabled) dropzone.classList.add("is-dragging");
      });
    }
    for (const type of ["dragleave", "drop"]) {
      dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
      });
    }
    dropzone.addEventListener("drop", (event) => {
      if (form.querySelector("fieldset").disabled) return;
      const file = event.dataTransfer && event.dataTransfer.files[0];
      if (file) choose(file);
    });

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      clearError();
      if (!chosen) {
        showError("Choose a video file first.");
        chooseButton.focus();
        return;
      }
      if (chosen.size > maxBytes) {
        showError(tooBig(chosen));
        return;
      }
      const slack = $("#editor-slack", form);
      if (slack && !slack.value.trim()) {
        showError("Add the editor's Slack member ID so they can be told about the result.");
        slack.focus();
        return;
      }

      const data = new FormData();
      data.append("file", chosen);
      data.append("editor_name", $("#editor-name", form).value);
      if (slack) data.append("editor_slack_id", slack.value.trim());

      const request = new XMLHttpRequest();
      request.open("POST", form.dataset.action);
      request.responseType = "json";
      request.upload.addEventListener("progress", (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.round((event.loaded / event.total) * 100);
        bar.value = percent;
        status.textContent =
          percent < 100 ? `Uploading… ${percent}%` : "Uploaded. Making sure it's a video…";
      });
      request.addEventListener("load", () => {
        const body = request.response || {};
        if (request.status === 202 && body.job_id) {
          status.textContent = "Uploaded. Opening its page…";
          window.location.assign(`/videos/${body.job_id}`);
          return;
        }
        setBusy(false);
        showError(body.message || "The upload didn't work. Please try again.");
      });
      request.addEventListener("error", () => {
        setBusy(false);
        showError("The upload was interrupted. Check the app is still running, then try again.");
      });
      setBusy(true);
      request.send(data);
    });
  }

  // --- "Checked videos" list (home page) ----------------------------------------------------
  function initVideoList() {
    const holder = $("[data-video-list]");
    // The static showcase has no server to ask, so it never polls.
    if (!holder || !holder.dataset.refreshUrl) return;
    const isActive = () => {
      const list = $(".video-list", holder);
      return Boolean(list && list.dataset.active === "true");
    };
    async function refresh() {
      if (!isActive()) return; // nothing being checked: stop polling
      try {
        const response = await fetch(holder.dataset.refreshUrl, { headers: { Accept: "text/html" } });
        // Don't yank the list out from under someone using the keyboard in it.
        if (response.ok && !holder.contains(document.activeElement)) {
          holder.innerHTML = await response.text();
        }
      } catch (error) {
        // The app may be restarting; try again on the next tick.
      }
      window.setTimeout(refresh, 2000);
    }
    window.setTimeout(refresh, 2000);
  }

  // --- Progress, then the report (video page) -----------------------------------------------
  function initProgress(page) {
    const started = new Date(page.dataset.created).getTime();
    const elapsed = $("[data-elapsed]", page);
    const stepper = $("[data-stepper]", page);
    const queued = $("[data-queued]", page);
    const steps = $$("[data-stage]", stepper);
    const order = steps.map((step) => step.dataset.stage);
    const stateWords = { done: "finished", current: "in progress", upcoming: "not started" };

    const tick = () => {
      elapsed.textContent = clock((Date.now() - started) / 1000);
    };
    tick();
    const timer = window.setInterval(tick, 1000);

    function showStage(stage) {
      // Same rule as the server: "received" means reading is next.
      const active = Math.max(order.indexOf(stage === "received" ? "reading" : stage), 1);
      steps.forEach((step, index) => {
        const state = stage === "done" || index < active ? "done" : index === active ? "current" : "upcoming";
        step.className = `step step-${state}`;
        if (state === "current") step.setAttribute("aria-current", "step");
        else step.removeAttribute("aria-current");
        $("[data-step-state]", step).textContent = stateWords[state];
      });
    }

    async function showReport() {
      window.clearInterval(timer);
      try {
        const response = await fetch(window.location.href, { headers: { Accept: "text/html" } });
        const html = await response.text();
        const fresh = $("[data-video-page]", new DOMParser().parseFromString(html, "text/html"));
        if (!response.ok || !fresh) throw new Error("no report");
        page.replaceWith(fresh);
        initVideoPage(fresh);
        const heading = $("#verdict-heading", fresh);
        if (heading) {
          heading.setAttribute("tabindex", "-1");
          heading.focus();
        }
      } catch (error) {
        window.location.reload();
      }
    }

    async function poll() {
      try {
        const response = await fetch(page.dataset.statusUrl, { headers: { Accept: "application/json" } });
        if (response.ok) {
          const job = await response.json();
          showStage(job.stage);
          if (queued) queued.hidden = job.status !== "queued";
          if (job.status === "completed" || job.status === "error") {
            await showReport();
            return;
          }
        }
      } catch (error) {
        // Keep polling; the app may be busy.
      }
      window.setTimeout(poll, 1000);
    }
    window.setTimeout(poll, 1000);
  }

  function seekAndPlay(video, seconds) {
    const go = () => {
      video.currentTime = seconds;
      const playing = video.play();
      if (playing) playing.catch(() => {});
    };
    if (video.readyState >= 1) go();
    else video.addEventListener("loadedmetadata", go, { once: true });
  }

  function initReport(page) {
    const video = $("[data-video]", page);
    if (video) {
      for (const button of $$("[data-seek]", page)) {
        button.addEventListener("click", () => {
          seekAndPlay(video, Number(button.dataset.seek));
          // On phones the player is pinned and already visible; elsewhere bring it into view.
          const box = video.getBoundingClientRect();
          if (box.bottom < 0 || box.top > window.innerHeight) {
            video.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        });
      }
      const timeline = $("[data-timeline]", page);
      const playhead = $("[data-playhead]", page);
      if (timeline && playhead) {
        const total = Number(timeline.dataset.duration);
        let frame = 0;
        const place = () => {
          const length = total || video.duration;
          if (!length) return;
          playhead.style.left = `${Math.min(video.currentTime / length, 1) * 100}%`;
        };
        const follow = () => {
          place();
          if (!video.paused && !video.ended) frame = window.requestAnimationFrame(follow);
        };
        video.addEventListener("play", () => {
          window.cancelAnimationFrame(frame);
          follow();
        });
        for (const type of ["timeupdate", "seeked", "loadedmetadata"]) {
          video.addEventListener(type, place);
        }
        place();
      }
    }

    for (const button of $$("[data-copy]", page)) {
      button.addEventListener("click", async () => {
        const source = $(button.dataset.copy, page);
        const status = $("[data-copy-status]", page);
        try {
          await navigator.clipboard.writeText(source.value);
          status.textContent = "Copied. Paste it into a message to the editor.";
        } catch (error) {
          source.focus();
          source.select();
          status.textContent = "Selected. Press Ctrl+C to copy it.";
        }
      });
    }
  }

  function initVideoPage(page) {
    if (page.dataset.state === "checking") initProgress(page);
    else initReport(page);
  }

  document.addEventListener("DOMContentLoaded", () => {
    initUpload();
    initVideoList();
    const page = $("[data-video-page]");
    if (page) initVideoPage(page);
  });
})();
