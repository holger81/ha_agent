class HaAgentPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._narrow = false;
    this._entryId = null;
    this._conversationId = this._newConversationId();
    this._tab = "chat";
    this._messages = [];
    this._skills = [];
    this._activity = [];
    this._threads = [];
    this._config = null;
    this._status = {};
    this._pendingDraft = null;
    this._streaming = false;
    this._msgId = 1;
    this._toolId = 1;
    this._unsubEvents = null;
    this._eventsReady = null;
    this._bootstrapError = null;
    this._turnTimeout = null;
    this._stickToBottom = true;
    this._chatRenderPending = false;
    this._threadSearch = "";
    this._threadSearchTimer = null;
    this._messagesScrollEl = null;
    this._messagesInteractEl = null;
    this._historyLoadSeq = 0;
    this._skillSaveNotice = null;
    this._editingSkill = null;
    this._viewingSkill = null;
    this._skillNotice = null;
    this._playbooks = [];
    this._editingPlaybook = null;
    this._playbookNotice = null;
    this._routeKeywords = [];
    this._editingRoute = null;
    this._routeNotice = null;
    this._recoveryHints = [];
    this._editingHint = null;
    this._hintNotice = null;
    this._evalStatus = null;
    this._evalCapabilities = null;
    this._evalNotice = null;
    this._evalCases = [];
    this._evalPollTimer = null;
    this._evalStatusFingerprint = "";
    this._activityNotice = null;
    this._discoverRequireDownloadApproval = true;
    this._discoverRequireTrialApproval = true;
    this._discoverModelsDir = "";
    this._discoverWebhookUrl = "";
    this._discoverSelectedModels = new Set();
    this._reloading = false;
    this._headerNotice = null;
    this._headerNoticeTimer = null;
    this._hacsStatus = null;
    this._hacsBusy = false;
  }

  _newConversationId() {
    return `console-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  }

  set hass(hass) {
    const first = !this._hass;
    this._hass = hass;
    if (first) {
      void this._bootstrap();
    }
  }

  set narrow(narrow) {
    this._narrow = narrow;
    this._render();
  }

  connectedCallback() {
    if (this._hass) {
      void this._ensureEventSubscription().then(() => {
        if (this._streaming || this._findOpenStreamMessage()) {
          this._scheduleChatRender();
        }
      });
    }
  }

  disconnectedCallback() {
    this._clearTurnTimeout();
    this._clearThreadSearchTimer();
    this._clearEvalPoll();
    if (this._messagesScrollEl) {
      this._messagesScrollEl.removeEventListener("scroll", this._onMessagesScroll);
      this._messagesScrollEl = null;
    }
    if (this._unsubEvents) {
      void this._unsubEvents();
      this._unsubEvents = null;
    }
    this._eventsReady = null;
  }

  _clearTurnTimeout() {
    if (this._turnTimeout) {
      clearTimeout(this._turnTimeout);
      this._turnTimeout = null;
    }
  }

  _turnTimeoutMs() {
    const llmTimeout = Number(this._config?.llm_timeout) || 120;
    const maxIterations = Number(this._config?.max_iterations) || 8;
    return (llmTimeout * maxIterations + 180) * 1000;
  }

  _armTurnTimeout() {
    this._clearTurnTimeout();
    this._turnTimeout = setTimeout(
      () => void this._recoverStuckTurn(),
      this._turnTimeoutMs()
    );
  }

  _newStreamMessage() {
    return {
      id: this._msgId++,
      role: "assistant",
      content: "",
      thinking: "",
      tools: [],
      thinkingCollapsed: false,
      thinkingUserToggled: false,
      activeSkill: null,
      turnMeta: null,
      _streamOpen: true,
    };
  }

  _findToolById(toolId) {
    const id = String(toolId);
    for (const msg of this._messages) {
      for (const tool of msg.tools || []) {
        if (String(tool.id) === id) {
          return tool;
        }
      }
    }
    return null;
  }

  _toolPreview(tool) {
    if (tool.detail) {
      return this._thinkingPreview(tool.detail);
    }
    if (tool.arguments && Object.keys(tool.arguments).length) {
      return this._thinkingPreview(JSON.stringify(tool.arguments));
    }
    return "";
  }

  _findMessageById(messageId) {
    const id = Number(messageId);
    if (!Number.isFinite(id)) return null;
    return this._messages.find((msg) => msg.id === id) || null;
  }

  _thinkingPreview(text) {
    const line = String(text || "")
      .split("\n")
      .find((part) => part.trim());
    const trimmed = (line || text || "").trim();
    if (trimmed.length <= 72) return trimmed;
    return `${trimmed.slice(0, 69)}…`;
  }

  _renderThinkingPanel(msg) {
    const thinking = String(msg.thinking || "").trim();
    if (!thinking) return "";
    const collapsed = Boolean(msg.thinkingCollapsed);
    const chevron = collapsed ? "▸" : "▾";
    const live =
      msg._streamOpen && !String(msg.content || "").trim() ? " (live)" : "";
    const preview = collapsed ? this._thinkingPreview(thinking) : "";
    return `
      <div class="thinking-panel ${collapsed ? "collapsed" : "expanded"}">
        <button
          type="button"
          class="thinking-toggle"
          data-thinking-toggle
          data-msg-id="${msg.id}"
          aria-expanded="${collapsed ? "false" : "true"}"
        >
          <span class="thinking-toggle-icon">${chevron}</span>
          <span class="thinking-toggle-label">Reasoning${live}</span>
          ${
            preview
              ? `<span class="thinking-preview">${this._escape(preview)}</span>`
              : ""
          }
        </button>
        <div class="thinking-body">${this._escape(thinking)}</div>
      </div>`;
  }

  _findOpenStreamMessage() {
    for (let index = this._messages.length - 1; index >= 0; index -= 1) {
      const msg = this._messages[index];
      if (msg.role === "assistant" && msg._streamOpen) {
        return msg;
      }
    }
    return null;
  }

  _closeOpenStreamMessages() {
    for (const msg of this._messages) {
      if (msg.role === "assistant" && msg._streamOpen) {
        msg._streamOpen = false;
        if (String(msg.thinking || "").trim() && !msg.thinkingUserToggled) {
          msg.thinkingCollapsed = true;
        }
        for (const tool of msg.tools || []) {
          if (!tool.userToggled && tool.phase !== "start") {
            tool.collapsed = true;
          }
        }
      }
    }
  }

  async _recoverStuckTurn() {
    if (!this._streaming) return;
    this._clearTurnTimeout();
    try {
      await this._loadHistory();
    } catch (_err) {
      /* history poll is best-effort */
    }
    const hasAssistant = this._messages.some(
      (m) => m.role === "assistant" && (m.content || m.thinking)
    );
    if (!hasAssistant) {
      this._messages.push({
        role: "assistant",
        content:
          "Error: No response received. Check HA Agent logs and LLM/MCP connectivity in Settings.",
        thinking: "",
      });
    }
    this._closeOpenStreamMessages();
    this._streaming = false;
    this._render();
    await this._loadPendingDraft();
    await this._refreshStatus();
  }

  async _ensureEventSubscription() {
    if (this._eventsReady) {
      return this._eventsReady;
    }
    this._eventsReady = (async () => {
      const onDelta = (ev) => {
        this._handleDelta(ev.data || {});
      };
      const onDone = (ev) => {
        void this._handleChatDone(ev.data || {});
      };
      const conn = this._hass.connection;
      const unsubDelta = await conn.subscribeEvents(
        onDelta,
        "ha_agent_chat_delta"
      );
      const unsubDone = await conn.subscribeEvents(
        onDone,
        "ha_agent_chat_done"
      );
      this._unsubEvents = async () => {
        await unsubDelta();
        await unsubDone();
      };
    })();
    return this._eventsReady;
  }

  async _call(type, payload = {}) {
    return this._hass.callWS({
      type,
      ...payload,
    });
  }

  async _bootstrap() {
    try {
      await this._ensureEventSubscription();
      const data = await this._call("ha_agent/subscribe", {});
      this._entryId = data.entry_id;
      this._config = data.config;
      this._status = data.status || {};
      await Promise.all([
        this._loadThreads(),
        this._loadSkills(),
        this._loadPendingDraft(),
      ]);
      if (this._threads.length > 0) {
        this._conversationId = this._threads[0].conversation_id;
      }
      await this._loadHistory();
      await this._loadHacsStatus();
    } catch (err) {
      this._bootstrapError = err?.message || String(err);
    }
    this._render();
  }

  async _refreshStatus() {
    if (!this._entryId) return;
    this._status = await this._call("ha_agent/status", {
      entry_id: this._entryId,
    });
    this._render();
  }

  _setHeaderNotice(message) {
    if (this._headerNoticeTimer) {
      clearTimeout(this._headerNoticeTimer);
      this._headerNoticeTimer = null;
    }
    this._headerNotice = message;
    if (
      message &&
      !message.startsWith("Reloading") &&
      !message.startsWith("Checking") &&
      !message.startsWith("Redownloading") &&
      !message.startsWith("Installed")
    ) {
      this._headerNoticeTimer = setTimeout(() => {
        this._headerNotice = null;
        this._headerNoticeTimer = null;
        this._render();
      }, 5000);
    }
  }

  async _loadHacsStatus() {
    if (!this._entryId) return;
    try {
      this._hacsStatus = await this._call("ha_agent/hacs/status", {
        entry_id: this._entryId,
      });
    } catch (_err) {
      this._hacsStatus = { hacs_available: false, repository_found: false };
    }
  }

  _hacsControlsEnabled() {
    const s = this._hacsStatus;
    return Boolean(s?.hacs_available && s?.repository_found);
  }

  _hacsUpdateLabel() {
    const s = this._hacsStatus;
    if (s?.update_available && s.latest_version) {
      const from = s.installed_version || "?";
      return `Update (${from} → ${s.latest_version})`;
    }
    return "Update";
  }

  async _checkHacsUpdate() {
    if (!this._entryId || this._hacsBusy) return;
    this._hacsBusy = true;
    this._setHeaderNotice("Checking for updates…");
    this._render();
    try {
      this._hacsStatus = await this._call("ha_agent/hacs/refresh", {
        entry_id: this._entryId,
      });
      const s = this._hacsStatus;
      if (s.update_available) {
        this._setHeaderNotice(
          `Update available: ${s.installed_version || "?"} → ${s.latest_version}`,
        );
      } else {
        this._setHeaderNotice(`Up to date (${s.installed_version || "unknown"}).`);
      }
    } catch (err) {
      this._setHeaderNotice(`Check failed: ${err?.message || err}`);
    } finally {
      this._hacsBusy = false;
      this._render();
    }
  }

  async _hacsUpdate(forceReinstall = false) {
    if (!this._entryId || this._hacsBusy) return;
    this._hacsBusy = true;
    this._setHeaderNotice(
      forceReinstall ? "Redownloading from HACS…" : "Checking for updates…",
    );
    this._render();
    try {
      const data = await this._call("ha_agent/hacs/update", {
        entry_id: this._entryId,
        force_refresh: true,
        force_reinstall: forceReinstall,
      });
      this._hacsStatus = data;
      if (data.installed) {
        this._setHeaderNotice(
          `Installed ${data.installed_version || data.latest_version || "update"}. Reloading…`,
        );
        this._render();
        await this._reloadIntegration(true);
        this._setHeaderNotice(
          `Updated to ${data.installed_version || data.latest_version || "latest"}. Reloaded.`,
        );
      } else {
        this._setHeaderNotice(`Already on latest (${data.installed_version || "unknown"}).`);
      }
    } catch (err) {
      this._setHeaderNotice(`Update failed: ${err?.message || err}`);
    } finally {
      this._hacsBusy = false;
      this._render();
    }
  }

  async _reloadIntegration(quiet = false) {
    if (!this._entryId || this._reloading) return;
    this._reloading = true;
    if (!quiet) {
      this._setHeaderNotice("Reloading integration…");
    }
    this._render();
    try {
      const data = await this._call("ha_agent/reload", {
        entry_id: this._entryId,
      });
      this._config = data.config || this._config;
      this._bootstrapError = null;
      await this._refreshStatus();
      await this._loadHacsStatus();
      if (this._tab === "eval") await this._loadEvalStatus();
      if (this._tab === "activity") await this._loadActivity();
      if (this._tab === "playbooks") await this._loadPlaybooks();
      if (this._tab === "routes") await this._loadRouteKeywords();
      if (this._tab === "recovery") await this._loadRecoveryHints();
      if (!quiet) {
        this._setHeaderNotice("Integration reloaded.");
      }
    } catch (err) {
      if (!quiet) {
        this._setHeaderNotice(`Reload failed: ${err?.message || err}`);
      } else {
        throw err;
      }
    } finally {
      this._reloading = false;
      this._render();
    }
  }

  async _loadSkills() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/skills/list", {
      entry_id: this._entryId,
      limit: 100,
    });
    this._skills = data.skills || [];
  }

  async _loadActivity() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/activity/list", {
      entry_id: this._entryId,
      limit: 50,
    });
    this._activity = data.turns || [];
  }

  async _loadPlaybooks() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/playbooks/list", {
      entry_id: this._entryId,
    });
    this._playbooks = data.playbooks || [];
  }

  async _loadRouteKeywords() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/route_keywords/list", {
      entry_id: this._entryId,
    });
    this._routeKeywords = data.routes || [];
  }

  async _loadRecoveryHints() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/recovery_hints/list", {
      entry_id: this._entryId,
    });
    this._recoveryHints = data.hints || [];
  }

  async _loadEvalStatus() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/eval/status", {
      entry_id: this._entryId,
    });
    this._evalStatus = data;
    this._evalStatusFingerprint = this._evalStatusFingerprintFor(data);
    const discover = data.discover || {};
    const discoverMsg = discover.progress?.message;
    if (discoverMsg && (data.pipeline === "discover" || discover.status === "awaiting_approval")) {
      this._evalNotice = discoverMsg;
    } else if (data.pipeline === "eval" && data.run?.progress?.message) {
      this._evalNotice = data.run.progress.message;
    }
    if (data.running) {
      this._startEvalPoll();
    } else {
      this._clearEvalPoll();
    }
    await this._loadEvalCases();
  }

  async _loadEvalCases() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/eval/cases/list", {
      entry_id: this._entryId,
    });
    this._evalCases = data.cases || [];
  }

  async _promoteActivityTurn(timestamp) {
    if (!this._entryId || timestamp == null) return;
    this._activityNotice = "Promoting turn to eval case…";
    this._render();
    try {
      const data = await this._call("ha_agent/eval/cases/promote", {
        entry_id: this._entryId,
        timestamp,
      });
      const caseId = data.case?.id || "case";
      this._activityNotice = `Promoted ${caseId}. It will run on the next eval suite.`;
      await this._loadEvalCases();
    } catch (err) {
      this._activityNotice = `Promote failed: ${err?.message || err}`;
    }
    this._render();
  }

  async _deleteEvalCase(caseId) {
    if (!this._entryId || !caseId) return;
    if (!confirm(`Delete promoted eval case ${caseId}?`)) return;
    await this._call("ha_agent/eval/cases/delete", {
      entry_id: this._entryId,
      case_id: caseId,
    });
    this._evalNotice = `Deleted ${caseId}.`;
    await this._loadEvalCases();
    this._render();
  }

  _evalStatusFingerprintFor(data) {
    if (!data) return "";
    const discover = data.discover || {};
    const run = data.run || {};
    const progress = discover.progress || {};
    const runProgress = run.progress || {};
    return JSON.stringify({
      running: data.running,
      pipeline: data.pipeline,
      discoverStatus: discover.status,
      discoverPhase: progress.phase,
      discoverMessage: progress.message,
      discoverProgressPct: progress.progress_percent ?? null,
      discoverPending: discover.pending_approval,
      proposalsLen: (discover.proposals || []).length,
      trialLen: (discover.trial_results || []).length,
      runStatus: run.status,
      runPhase: runProgress.phase,
      runMessage: runProgress.message,
      runModel: runProgress.model,
      runCase: runProgress.case_id,
      runTask: runProgress.task,
      runCaseIndex: runProgress.case_index,
      runCompletedCases: runProgress.completed_cases,
      runProgressPct: runProgress.progress_percent ?? null,
      taskScoresLen: (run.task_scores || []).length,
    });
  }

  _clearEvalPoll() {
    if (this._evalPollTimer) {
      clearInterval(this._evalPollTimer);
      this._evalPollTimer = null;
    }
  }

  _startEvalPoll() {
    this._clearEvalPoll();
    this._evalPollTimer = setInterval(async () => {
      try {
        const previous = this._evalStatusFingerprint;
        await this._loadEvalStatus();
        const fingerprint = this._evalStatusFingerprintFor(this._evalStatus);
        if (fingerprint !== previous) {
          this._render();
        }
      } catch (_err) {
        this._clearEvalPoll();
      }
    }, 3000);
  }

  async _probeEvalServer() {
    if (!this._entryId) return;
    this._evalNotice = "Probing llama.cpp server…";
    this._render();
    const data = await this._call("ha_agent/eval/probe", {
      entry_id: this._entryId,
    });
    this._evalCapabilities = data.capabilities || null;
    this._evalNotice = "Server probe complete.";
    this._render();
  }

  async _startEvalRun(preloadModels = false) {
    if (!this._entryId) return;
    this._evalNotice = preloadModels
      ? "Starting eval suite (preloading models first)…"
      : "Starting eval suite…";
    this._render();
    await this._call("ha_agent/eval/start", {
      entry_id: this._entryId,
      include_settings: true,
      preload_models: preloadModels,
    });
    await this._loadEvalStatus();
    this._evalNotice = "Eval running in background.";
    this._render();
  }

  async _evalCatalogModel(modelId, needsPreload = false) {
    if (!this._entryId || !modelId) return;
    const label = this._shortModelId(modelId);
    this._evalNotice = needsPreload
      ? `Starting eval for ${label} (preload first)…`
      : `Starting eval for ${label}…`;
    this._render();
    try {
      await this._call("ha_agent/eval/start", {
        entry_id: this._entryId,
        models: [modelId],
        include_settings: false,
        preload_models: needsPreload,
      });
      await this._loadEvalStatus();
      this._evalNotice = `Eval running for ${label}.`;
    } catch (err) {
      this._evalNotice = err?.message || `Could not start eval for ${label}.`;
    }
    this._render();
  }

  async _preloadEvalModels() {
    if (!this._entryId) return;
    const assignments = this._evalStatus?.run?.settings_recommendation?.model_assignments || {};
    const models = [
      ...new Set(
        Object.values(assignments)
          .map((item) => item?.model)
          .filter(Boolean),
      ),
    ];
    if (!models.length) {
      this._evalNotice = "No recommended models to preload — run eval first.";
      this._render();
      return;
    }
    this._evalNotice = `Preloading ${models.length} model(s)…`;
    this._render();
    const data = await this._call("ha_agent/eval/preload_models", {
      entry_id: this._entryId,
      models,
    });
    this._evalCapabilities = data.capabilities || this._evalCapabilities;
    this._evalNotice = `Preload complete: ${data.loaded_count || 0} loaded, ${data.failed_count || 0} failed.`;
    this._render();
  }

  async _unloadEvalModel(modelId) {
    if (!this._entryId || !modelId) return;
    if (!confirm(`Unload model ${modelId} from llama.cpp?`)) return;
    const data = await this._call("ha_agent/eval/unload_model", {
      entry_id: this._entryId,
      model_id: modelId,
    });
    this._evalCapabilities = data.capabilities || null;
    const ok = data.result?.ok;
    this._evalNotice = ok
      ? `Unloaded ${modelId}.`
      : `Failed to unload ${modelId}: ${data.result?.error || "unknown error"}`;
    this._render();
  }

  async _deleteEvalModel(modelId) {
    if (!this._entryId || !modelId) return;
    if (
      !confirm(
        `Delete ${modelId} from llama.cpp cache? This removes the downloaded GGUF (preset models cannot be deleted).`,
      )
    ) {
      return;
    }
    const data = await this._call("ha_agent/eval/delete_model", {
      entry_id: this._entryId,
      model_id: modelId,
    });
    this._evalCapabilities = data.capabilities || null;
    const ok = data.delete?.ok;
    this._evalNotice = ok
      ? `Deleted ${modelId} from router cache.`
      : `Could not delete ${modelId}: ${data.delete?.error || data.delete?.reason || "unknown error"}`;
    this._render();
  }

  async _applyEvalRecommendations() {
    if (!this._entryId) return;
    if (!confirm("Apply recommended chat, action, email, news, and classifier models from the latest eval?")) {
      return;
    }
    const data = await this._call("ha_agent/eval/apply", {
      entry_id: this._entryId,
    });
    this._config = data.config || this._config;
    this._evalNotice = "Applied eval model recommendations.";
    this._render();
  }

  async _applyEvalServerSettings() {
    if (!this._entryId) return;
    const applyMode =
      this._evalStatus?.run?.settings_recommendation?.apply_mode || "preset";
    const confirmMsg =
      applyMode === "preset"
        ? "Router mode: settings cannot be applied live. Copy the preset and restart the llama Docker container?"
        : "Apply recommended llama.cpp server settings via POST /props and re-probe?";
    if (!confirm(confirmMsg)) {
      return;
    }
    const data = await this._call("ha_agent/eval/apply_settings", {
      entry_id: this._entryId,
    });
    if (data.mode === "preset") {
      const preset = data.preset_ini || "";
      if (preset.trim()) {
        try {
          await navigator.clipboard.writeText(preset);
          this._evalNotice = `${data.message || "Preset copied."} Restart the llama container after updating the preset file.`;
        } catch (_err) {
          this._evalNotice = `${data.message || "Use the preset below."} ${data.docker_hint || ""}`;
        }
      } else {
        this._evalNotice = data.message || "Router mode requires a preset edit.";
      }
    } else {
      const verified = data.verification?.verified_count ?? 0;
      const total = (data.verification?.checks || []).length;
      const applied = (data.applied || []).length;
      const failed = (data.failed || []).length;
      this._evalNotice = `${data.message || "Settings applied."} (${applied} ok, ${failed} failed, ${verified}/${total} verified)`;
      if (data.after) {
        this._evalCapabilities = {
          ...(this._evalCapabilities || {}),
          summary: data.after,
        };
      }
    }
    this._render();
  }

  async _cancelPipeline() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/eval/cancel", { entry_id: this._entryId });
    this._evalNotice = data.cancelled
      ? "Cancel requested — stopping pipeline…"
      : "No active pipeline to cancel.";
    await this._loadEvalStatus();
    this._render();
  }

  async _startDiscoverPipeline() {
    if (!this._entryId) return;
    this._discoverSelectedModels = new Set();
    this._evalNotice = "Starting model discover pipeline…";
    this._render();
    const payload = {
      require_download_approval: this._discoverRequireDownloadApproval,
      require_trial_approval: this._discoverRequireTrialApproval,
    };
    const dir = (this._discoverModelsDir || this._config?.eval_models_dir || "").trim();
    const webhook = (
      this._discoverWebhookUrl || this._config?.eval_download_webhook_url || ""
    ).trim();
    if (dir) payload.models_dir = dir;
    if (webhook) payload.download_webhook_url = webhook;
    await this._call("ha_agent/eval/discover/start", {
      entry_id: this._entryId,
      ...payload,
    });
    await this._loadEvalStatus();
    this._evalNotice =
      this._evalStatus?.discover?.progress?.message || "Discover pipeline running.";
    this._render();
  }

  _selectedDiscoverModelIds() {
    return [
      ...this.shadowRoot.querySelectorAll('[data-action="discover-select-model"]:checked'),
    ]
      .map((input) => input.getAttribute("data-model-id"))
      .filter(Boolean);
  }

  async _approveDiscoverDownloads(modelIds = null) {
    if (!this._entryId) return;
    const ids =
      modelIds ??
      (() => {
        const selected = this._selectedDiscoverModelIds();
        return selected.length ? selected : [...this._discoverSelectedModels];
      })();
    if (!ids.length && modelIds === null) {
      this._evalNotice = "Select at least one model to download.";
      this._render();
      return;
    }
    await this._call("ha_agent/eval/discover/approve_download", {
      entry_id: this._entryId,
      model_ids: ids,
    });
    this._discoverSelectedModels = new Set();
    await this._loadEvalStatus();
    this._evalNotice =
      this._evalStatus?.discover?.progress?.message || "Download approved.";
    this._render();
  }

  async _approveDiscoverTrial(modelId, approved) {
    if (!this._entryId || !modelId) return;
    await this._call("ha_agent/eval/discover/approve_trial", {
      entry_id: this._entryId,
      model_id: modelId,
      approved,
    });
    await this._loadEvalStatus();
    this._evalNotice =
      this._evalStatus?.discover?.progress?.message ||
      (approved ? `Trial approved for ${modelId}.` : `Skipped trial for ${modelId}.`);
    this._render();
  }

  _evalPhaseLabel(phase) {
    const labels = {
      starting: "Starting",
      probe: "Probing server",
      preload: "Loading model",
      benchmark: "Benchmarking",
      recommend: "Recommendations",
      completed: "Completed",
      cancelled: "Cancelled",
      failed: "Failed",
    };
    return labels[phase] || phase || "Idle";
  }

  _renderEvalProgressPanel(progress, isRunning) {
    if (!isRunning && !progress?.message) return "";
    const phase = progress?.phase || "idle";
    const message = progress?.message || "";
    const caseIndex = progress?.case_index;
    const caseTotal = progress?.case_total;
    const completed = progress?.completed_cases;
    const pct =
      progress?.progress_percent != null
        ? Number(progress.progress_percent)
        : caseIndex != null && caseTotal
          ? Math.round((caseIndex / caseTotal) * 100)
          : null;
    const bar =
      pct != null && !Number.isNaN(pct)
        ? `<div class="discover-progress" aria-label="Eval progress">
            <div class="discover-progress-bar" style="width:${Math.max(0, Math.min(100, pct))}%"></div>
            <span class="discover-progress-label">${Math.round(pct)}%</span>
          </div>`
        : "";
    const meta = [];
    if (progress?.model) {
      meta.push(`Model: ${this._shortModelId(progress.model)}`);
    }
    if (caseIndex != null && caseTotal) {
      meta.push(`Case ${caseIndex}/${caseTotal}`);
    }
    if (progress?.model_index != null && progress?.model_total) {
      meta.push(`Model ${progress.model_index}/${progress.model_total}`);
    }
    if (completed != null) {
      meta.push(`${completed} completed`);
    }
    if (progress?.task && progress?.case_id) {
      meta.push(`${progress.task}/${progress.case_id}`);
    }
    return `
      <div class="eval-progress-panel">
        <p class="eval-progress-phase"><strong>${this._escape(this._evalPhaseLabel(phase))}</strong></p>
        ${message ? `<p class="eval-progress-message">${this._escape(message)}</p>` : ""}
        ${bar}
        ${meta.length ? `<p class="activity-hint">${this._escape(meta.join(" · "))}</p>` : ""}
      </div>`;
  }

  _shortModelId(modelId) {
    if (!modelId || modelId.length <= 64) return modelId || "";
    return `${modelId.slice(0, 44)}…${modelId.slice(-16)}`;
  }

  _discoverProgressBar(progress) {
    const pctValue = progress?.progress_percent;
    const pct =
      pctValue != null
        ? Number(pctValue)
        : this._discoverProgressValue(progress);
    if (pct == null || Number.isNaN(pct)) return "";
    const clamped = Math.max(0, Math.min(100, Math.round(pct)));
    return `<div class="discover-progress" aria-label="Download progress">
      <div class="discover-progress-bar" style="width:${clamped}%"></div>
      <span class="discover-progress-label">${clamped}%</span>
    </div>`;
  }

  _discoverProgressValue(progress) {
    if (!progress || typeof progress !== "object") return null;
    if (progress.progress_percent != null) return Number(progress.progress_percent);
    const nested = progress.progress || progress.payload || progress;
    const text = this._formatCatalogProgress(nested);
    if (text === "—" || !text.endsWith("%")) return null;
    return Number(text.replace("%", ""));
  }

  _renderDiscoverSection(discover) {
    const state = discover || {
      status: "idle",
      progress: {},
      proposals: [],
      trial_results: [],
    };
    const progress = state.progress || {};
    const proposals = state.proposals || [];
    const trialResults = state.trial_results || [];
    const pending = state.pending_approval;
    const proposalChecks = proposals
      .map((item) => {
        const id = item.model_id || "";
        const checked = this._discoverSelectedModels.has(id) ? "checked" : "";
        return `<li>
          <label>
            <input type="checkbox" data-action="discover-select-model" data-model-id="${this._escape(id)}" ${checked} />
            <strong>${this._escape(id)}</strong>
            ${item.skip_download ? " (skipped — tried before)" : ""}
          </label>
          <div class="activity-hint">${this._escape(item.reason || "")}</div>
        </li>`;
      })
      .join("");
    const trialRows = trialResults
      .map(
        (item) => `<tr>
          <td class="model-id-short">${this._escape(this._shortModelId(item.model_id || ""))}</td>
          <td>${item.mean_score != null ? Number(item.mean_score).toFixed(2) : "—"}</td>
          <td>${item.incumbent_score != null ? Number(item.incumbent_score).toFixed(2) : "—"}</td>
          <td>${item.accepted ? "accepted" : item.skipped ? "skipped" : "rejected"}</td>
          <td>${this._escape(item.reason || "")}</td>
          <td>${
            !item.accepted
              ? `<button data-action="discover-retry" data-model-id="${this._escape(item.model_id || "")}">Retry</button>`
              : ""
          }</td>
        </tr>`,
      )
      .join("");
    const trialModel =
      state.pending_trial_model_id || progress.model_id || null;
    const manual = progress.manual_download || {};
    const statusMessage =
      progress.message ||
      "Search Hugging Face, download candidates, and trial them against your incumbent model.";
    const shortStatusMessage = progress.model_id
      ? statusMessage.replace(progress.model_id, this._shortModelId(progress.model_id))
      : statusMessage;
    const showApproval = pending === "download" && proposals.length > 0;
    const showProposalSummary =
      proposals.length > 0 && !showApproval && state.status !== "idle";
    const progressBar =
      progress.phase === "downloading" || progress.phase === "loading"
        ? this._discoverProgressBar(progress)
        : "";
    return `
      <section class="eval-section discover-section form-grid">
      <h3>Discover models</h3>
      <p class="discover-status eval-status-line"><strong>${this._escape(state.status || "idle")}</strong> — ${this._escape(shortStatusMessage)}</p>
      ${progressBar}
      ${manual.hf_url ? `<p class="activity-hint">Download URL: <a href="${this._escape(manual.hf_url)}" target="_blank" rel="noopener">${this._escape(manual.hf_url)}</a></p>` : ""}
      ${manual.docker_hint ? `<p class="activity-hint">${this._escape(manual.docker_hint)}</p>` : ""}
      ${manual.llama_cli_hint ? `<p class="activity-hint">${this._escape(manual.llama_cli_hint)}</p>` : ""}
      ${state.error ? `<p class="banner">${this._escape(state.error)}</p>` : ""}
      <div class="discover-options">
        <label class="checkbox-inline"><input type="checkbox" data-action="discover-require-download" ${this._discoverRequireDownloadApproval ? "checked" : ""} /> Wait for my approval before downloading</label>
        <label class="checkbox-inline"><input type="checkbox" data-action="discover-require-trial" ${this._discoverRequireTrialApproval ? "checked" : ""} /> Wait for my approval before trial eval</label>
        <p class="activity-hint">When download approval is off, every proposed model downloads automatically. When on, check the models you want and click Download selected.</p>
      </div>
      <label class="stacked-field">Download webhook <span class="field-hint">optional — runs on llama host</span>
        <input data-field="discover-webhook-url" value="${this._escape(this._discoverWebhookUrl || this._config?.eval_download_webhook_url || "")}" placeholder="http://192.168.10.31:9999/ha-agent/download" />
      </label>
      <label class="stacked-field">Shared models dir <span class="field-hint">optional — only if HA can write the llama volume</span>
        <input data-field="discover-models-dir" value="${this._escape(this._discoverModelsDir || this._config?.eval_models_dir || "")}" placeholder="/path/to/llama/models" />
      </label>
      <div class="eval-action-row">
        <button data-action="discover-start">Start discover pipeline</button>
        <button data-action="eval-cancel">Cancel pipeline</button>
      </div>
      ${showApproval ? `
        <h4>Approve downloads</h4>
        <ul class="discover-proposals">${proposalChecks}</ul>
        <div class="eval-action-row">
          <button data-action="discover-approve-download">Download selected</button>
          <button data-action="discover-approve-download-none">Skip all downloads</button>
        </div>` : ""}
      ${pending === "trial" && trialModel ? `
        <h4>Approve trial</h4>
        <p>Benchmark <strong class="model-id-short">${this._escape(this._shortModelId(trialModel))}</strong> against your incumbent model?</p>
        <div class="eval-action-row">
          <button data-action="discover-approve-trial" data-model-id="${this._escape(trialModel)}" data-approved="true">Run trial eval</button>
          <button data-action="discover-approve-trial" data-model-id="${this._escape(trialModel)}" data-approved="false">Skip this model</button>
        </div>` : ""}
      ${showProposalSummary ? `<h4>Proposals</h4><ul class="discover-proposals">${proposalChecks.replace(/<input[^>]+>/g, "")}</ul>` : ""}
      ${trialRows ? `<h4>Trial results</h4>
        <div class="eval-table-wrap">
        <table>
          <thead><tr><th>Model</th><th>Score</th><th>Incumbent</th><th>Outcome</th><th>Notes</th><th></th></tr></thead>
          <tbody>${trialRows}</tbody>
        </table>
        </div>` : ""}
      <p class="activity-hint">Router llama.cpp servers download via POST /models (HA tracks /models/sse progress). Fallbacks: optional webhook, shared models dir, or manual host download with polling.</p>
      </section>`;
  }

  async _retryDiscoverModel(modelId) {
    if (!this._entryId || !modelId) return;
    if (
      !confirm(
        `Retry download and trial eval for ${modelId}? This unloads any partial copy and downloads again.`,
      )
    ) {
      return;
    }
    this._evalNotice = `Retrying ${modelId}…`;
    this._render();
    await this._call("ha_agent/eval/discover/retry", {
      entry_id: this._entryId,
      model_id: modelId,
    });
    await this._loadEvalStatus();
    this._evalNotice =
      this._evalStatus?.discover?.progress?.message || `Retry started for ${modelId}.`;
    this._render();
  }

  async _copyEvalPreset() {
    const preset = this._evalStatus?.run?.settings_recommendation?.preset_ini || "";
    if (!preset.trim()) {
      this._evalNotice = "No preset available yet — run eval first.";
      this._render();
      return;
    }
    try {
      await navigator.clipboard.writeText(preset);
      this._evalNotice = "Copied llama.cpp preset to clipboard.";
    } catch (_err) {
      this._evalNotice = "Could not copy preset — select the preset text manually.";
    }
    this._render();
  }

  _clearThreadSearchTimer() {
    if (this._threadSearchTimer) {
      clearTimeout(this._threadSearchTimer);
      this._threadSearchTimer = null;
    }
  }

  _onThreadSearchInput(value) {
    this._threadSearch = value;
    this._clearThreadSearchTimer();
    this._threadSearchTimer = setTimeout(async () => {
      await this._loadThreads(this._threadSearch);
      this._render();
    }, 250);
  }

  async _loadThreads(query = this._threadSearch) {
    if (!this._entryId) return;
    const payload = { entry_id: this._entryId };
    const trimmed = String(query || "").trim();
    if (trimmed) {
      payload.query = trimmed;
    }
    const data = await this._call("ha_agent/threads/list", payload);
    this._threads = data.threads || [];
  }

  async _deleteThread(conversationId) {
    if (!this._entryId || !conversationId || this._streaming) return;
    if (!confirm("Delete this chat and its history?")) return;

    await this._call("ha_agent/threads/delete", {
      entry_id: this._entryId,
      conversation_id: conversationId,
    });

    if (this._conversationId === conversationId) {
      this._conversationId = this._newConversationId();
      this._messages = [];
      this._pendingDraft = null;
    }

    await this._loadThreads(this._threadSearch);
    if (this._conversationId !== conversationId) {
      await this._loadHistory();
    }
    this._render();
  }

  _hasRenderableMessage(message) {
    if (message.tools && message.tools.length) return true;
    if (String(message.thinking || "").trim()) return true;
    if (String(message.content || "").trim()) return true;
    if (message.turnMeta && Object.keys(message.turnMeta).length) return true;
    return false;
  }

  _mergeTurnMeta(existing, patch) {
    if (!patch || typeof patch !== "object") return existing || null;
    const merged = { ...(existing || {}) };
    for (const [key, value] of Object.entries(patch)) {
      if (value && typeof value === "object" && !Array.isArray(value)) {
        merged[key] = { ...(merged[key] || {}), ...value };
      } else if (value !== undefined && value !== null) {
        merged[key] = value;
      }
    }
    return merged;
  }

  _renderTurnMeta(msg) {
    const m = msg.turnMeta;
    if (!m || !Object.keys(m).length) return "";
    const chips = [];
    const add = (label, value) => {
      if (value !== undefined && value !== null && String(value).trim()) {
        chips.push({ label, value: String(value) });
      }
    };
    add("Route", m.route);
    add("Classification", m.classification);
    add("Playbook", m.playbook);
    if (m.playbook_detail) {
      add("Playbook picker", m.playbook_detail);
    }
    if (m.skill) {
      add("Skill", m.skill);
    } else if (msg.activeSkill?.title) {
      add("Skill", msg.activeSkill.title);
    }
    if (m.model_role && m.model) {
      const host = m.host ? ` @ ${m.host}` : "";
      add(m.model_role, `${m.model}${host}`);
    }
    if (m.classifier?.model) {
      const host = m.classifier.host ? ` @ ${m.classifier.host}` : "";
      add("Classifier", `${m.classifier.model}${host}`);
    }
    add("LLM step", m.iteration);
    add("MCP tools", m.mcp_tools);
    add("History msgs", m.history_messages);
    if (!chips.length) return "";
    const items = chips
      .map(
        (chip) =>
          `<span class="turn-meta-chip" title="${this._escape(chip.value)}">` +
          `<span class="turn-meta-label">${this._escape(chip.label)}</span> ` +
          `<span class="turn-meta-value">${this._escape(chip.value)}</span>` +
          `</span>`
      )
      .join("");
    return `<div class="turn-meta"><span class="turn-meta-heading">Turn info</span>${items}</div>`;
  }

  _applyHistory(history) {
    if (this._streaming) return;
    const priorMeta = new Map();
    for (const msg of this._messages) {
      if (msg.role === "assistant" && msg.content && msg.turnMeta) {
        priorMeta.set(String(msg.content).trim(), msg.turnMeta);
      }
    }
    this._messages = (history || []).map((item) => {
      const thinking = String(item.thinking || "");
      const content = item.content || "";
      return {
        id: this._msgId++,
        role: item.role,
        content,
        thinking,
        tools: item.tools || [],
        turnMeta:
          item.turn_meta ||
          priorMeta.get(String(content).trim()) ||
          null,
        thinkingCollapsed: Boolean(thinking.trim()),
        thinkingUserToggled: false,
      };
    });
  }

  async _loadHistory() {
    if (!this._entryId) return;
    const seq = ++this._historyLoadSeq;
    const data = await this._call("ha_agent/chat/history/list", {
      entry_id: this._entryId,
      conversation_id: this._conversationId,
    });
    if (seq !== this._historyLoadSeq || this._streaming) return;
    this._applyHistory(data.history);
  }

  async _loadPendingDraft() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/skills/pending_get", {
      entry_id: this._entryId,
      conversation_id: this._conversationId,
    });
    this._pendingDraft = data.draft;
    this._render();
  }

  _appendStreamText(buffer, piece) {
    if (!piece) return buffer || "";
    const current = buffer || "";
    if (!current) return piece;
    if (piece.startsWith(current)) return piece;
    if (current.endsWith(piece)) return current;
    return current + piece;
  }

  _handleDelta(data) {
    if (data.entry_id && data.entry_id !== this._entryId) return;
    if (data.conversation_id !== this._conversationId) return;
    if (
      !data.thinking &&
      !data.content &&
      !data.tool &&
      !data.thinking_clear &&
      !data.skill &&
      !data.meta
    ) {
      return;
    }

    if (!this._streaming) {
      this._streaming = true;
      this._armTurnTimeout();
    }

    let msg = this._findOpenStreamMessage();
    if (!msg) {
      msg = this._newStreamMessage();
      this._messages.push(msg);
    }
    if (data.thinking_clear) {
      msg.thinking = "";
    }
    if (data.skill) {
      msg.activeSkill = data.skill;
    }
    if (data.meta) {
      msg.turnMeta = this._mergeTurnMeta(msg.turnMeta, data.meta);
    }
    if (data.thinking) {
      msg.thinking = this._appendStreamText(msg.thinking, data.thinking);
    }
    if (data.content) {
      const hadContent = Boolean(String(msg.content || "").trim());
      msg.content = this._appendStreamText(msg.content, data.content);
      if (
        !hadContent &&
        String(msg.content || "").trim() &&
        !msg.thinkingUserToggled
      ) {
        msg.thinkingCollapsed = true;
      }
    }
    if (data.tool) {
      this._applyToolDelta(msg, data.tool);
    }
    this._scheduleChatRender();
  }

  _applyToolDelta(msg, tool) {
    msg.tools = msg.tools || [];
    const last = msg.tools[msg.tools.length - 1];
    if (
      tool.phase !== "start" &&
      last &&
      last.phase === "start" &&
      last.name === tool.name
    ) {
      const merged = { ...last, ...tool };
      if (merged.phase !== "start" && !merged.userToggled) {
        merged.collapsed = true;
      }
      msg.tools[msg.tools.length - 1] = merged;
      return;
    }
    msg.tools.push({
      ...tool,
      id: this._toolId++,
      collapsed: tool.phase !== "start",
      userToggled: false,
    });
  }

  _renderToolCall(tool) {
    const phase = tool.phase || "start";
    const labels = { start: "Running", done: "Done", error: "Failed" };
    const label = labels[phase] || phase;
    const name = tool.name || tool.call_name || "tool";
    const collapsed = Boolean(tool.collapsed);
    const chevron = collapsed ? "▸" : "▾";
    const preview = collapsed ? this._toolPreview(tool) : "";
    const args =
      tool.arguments && Object.keys(tool.arguments).length
        ? `<pre class="tool-args">${this._escape(
            JSON.stringify(tool.arguments, null, 2)
          )}</pre>`
        : "";
    const detail = tool.detail
      ? `<div class="tool-detail">${this._escape(tool.detail)}</div>`
      : "";
    return `
      <div class="tool-call tool-call--${phase} ${
        collapsed ? "collapsed" : "expanded"
      }">
        <button
          type="button"
          class="tool-call-toggle"
          data-tool-toggle
          data-tool-id="${tool.id}"
          aria-expanded="${collapsed ? "false" : "true"}"
        >
          <span class="tool-call-toggle-icon">${chevron}</span>
          <span class="tool-call-label">${label}</span>
          <code class="tool-call-name">${this._escape(name)}</code>
          ${
            preview
              ? `<span class="tool-call-preview">${this._escape(preview)}</span>`
              : ""
          }
        </button>
        <div class="tool-call-body">
          ${args}
          ${detail}
        </div>
      </div>`;
  }

  _scheduleChatRender() {
    if (this._chatRenderPending) return;
    this._chatRenderPending = true;
    requestAnimationFrame(() => {
      this._chatRenderPending = false;
      if (this._tab !== "chat") {
        return;
      }
      if (this.shadowRoot?.querySelector(".messages")) {
        this._updateChatMessages();
        return;
      }
      this._render();
    });
  }

  _shouldShowTypingIndicator() {
    if (!this._streaming) return false;
    const last = this._messages[this._messages.length - 1];
    if (!last || last.role !== "assistant") return true;
    return !(
      String(last.thinking || "").trim() ||
      String(last.content || "").trim() ||
      (last.tools && last.tools.length)
    );
  }

  _renderMessageListHtml() {
    const messages = this._messages
      .filter((m) => this._hasRenderableMessage(m))
      .map((m) => {
        if (m.role === "user") {
          return `<div class="bubble user">${this._escape(m.content)}</div>`;
        }

        const skillBadge =
          !m.turnMeta?.skill && m.activeSkill
            ? `<div class="skill-badge">Skill: ${this._escape(
                m.activeSkill.title || m.activeSkill.slug || "unknown"
              )}</div>`
            : "";
        const turnMeta = this._renderTurnMeta(m);
        const thinking = this._renderThinkingPanel(m);
        const toolBlocks = (m.tools || [])
          .map((tool) => this._renderToolCall(tool))
          .join("");
        const tools = toolBlocks
          ? `<div class="tools-stack">${toolBlocks}</div>`
          : "";
        const body = this._renderAssistantBody(m.content);
        const bubble = String(m.content || "").trim()
          ? `<div class="bubble assistant">${body}</div>`
          : "";

        return `<div class="assistant-turn">${turnMeta}${skillBadge}${thinking}${tools}${bubble}</div>`;
      })
      .join("");

    const typing = this._shouldShowTypingIndicator()
      ? '<div class="bubble assistant typing">Thinking…</div>'
      : "";

    const empty = !messages && !typing
      ? `<div class="empty-chat">No messages in this chat yet.${
          this._threads.length ? " History may have been cleared after a restart unless memory persistence is enabled in Settings." : ""
        }</div>`
      : "";

    return `${messages}${typing}${empty}`;
  }

  _updateChatMessages() {
    const messagesEl = this.shadowRoot?.querySelector(".messages");
    if (!messagesEl) {
      this._render();
      return;
    }
    const saved = this._captureMessagesScroll();
    messagesEl.innerHTML = this._renderMessageListHtml();
    this._bindMessagesScroll(messagesEl);
    this._bindMessagesInteractions(messagesEl);
    this._restoreMessagesScroll(saved);
  }

  _bindMessagesInteractions(el) {
    if (this._messagesInteractEl === el) return;
    this._messagesInteractEl = el;
    el.addEventListener("click", (ev) => {
      const toolToggle = ev.target.closest("[data-tool-toggle]");
      if (toolToggle) {
        const tool = this._findToolById(toolToggle.getAttribute("data-tool-id"));
        if (tool) {
          tool.collapsed = !tool.collapsed;
          tool.userToggled = true;
          this._updateChatMessages();
        }
        return;
      }
      const toggle = ev.target.closest("[data-thinking-toggle]");
      if (!toggle) return;
      const msg = this._findMessageById(toggle.getAttribute("data-msg-id"));
      if (!msg) return;
      msg.thinkingCollapsed = !msg.thinkingCollapsed;
      msg.thinkingUserToggled = true;
      this._updateChatMessages();
    });
  }

  _capturePanelScroll() {
    const el = this.shadowRoot?.querySelector(".panel");
    if (!el || this._tab === "chat") return null;
    return { scrollTop: el.scrollTop };
  }

  _restorePanelScroll(saved) {
    if (!saved) return;
    const el = this.shadowRoot?.querySelector(".panel");
    if (!el) return;
    const apply = () => {
      const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
      el.scrollTop = Math.min(saved.scrollTop, maxScroll);
    };
    apply();
    requestAnimationFrame(apply);
  }

  _captureMessagesScroll() {
    const el = this.shadowRoot?.querySelector(".messages");
    if (!el) {
      return { scrollTop: 0, stickToBottom: this._stickToBottom };
    }
    const distanceFromBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight;
    return {
      scrollTop: el.scrollTop,
      stickToBottom: distanceFromBottom < 48,
    };
  }

  _restoreMessagesScroll(saved) {
    const el = this.shadowRoot?.querySelector(".messages");
    if (!el) return;
    this._bindMessagesScroll(el);
    if (saved.stickToBottom) {
      this._stickToBottom = true;
      el.scrollTop = el.scrollHeight;
      return;
    }
    this._stickToBottom = false;
    const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
    el.scrollTop = Math.min(saved.scrollTop, maxScroll);
  }

  _bindMessagesScroll(el) {
    if (this._messagesScrollEl === el) return;
    if (this._messagesScrollEl) {
      this._messagesScrollEl.removeEventListener("scroll", this._onMessagesScroll);
    }
    this._messagesScrollEl = el;
    el.addEventListener("scroll", this._onMessagesScroll, { passive: true });
  }

  _onMessagesScroll = () => {
    const el = this._messagesScrollEl;
    if (!el) return;
    const distanceFromBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight;
    this._stickToBottom = distanceFromBottom < 48;
  };

  async _handleChatDone(data) {
    if (data.entry_id && data.entry_id !== this._entryId) return;
    if (data.conversation_id !== this._conversationId) return;
    this._clearTurnTimeout();
    this._closeOpenStreamMessages();
    this._streaming = false;
    for (let index = this._messages.length - 1; index >= 0; index -= 1) {
      const msg = this._messages[index];
      if (msg.role !== "assistant") continue;
      const patch = {};
      if (data.turn_meta) {
        Object.assign(patch, data.turn_meta);
      }
      if (data.active_skill && data.active_skill !== "none") {
        if (!msg.activeSkill) {
          msg.activeSkill = { title: data.active_skill };
        }
        patch.skill = data.active_skill;
      }
      if (data.last_route) {
        patch.route = data.last_route;
      }
      if (Object.keys(patch).length) {
        msg.turnMeta = this._mergeTurnMeta(msg.turnMeta, patch);
      }
      break;
    }
    if (data.error) {
      this._messages.push({
        role: "assistant",
        content: `Error: ${data.error}`,
        thinking: "",
      });
    }
    this._render();
    await Promise.all([this._loadThreads(), this._loadPendingDraft()]);
    await this._refreshStatus();
  }

  async _sendMessage(text) {
    if (!text.trim() || this._streaming) return;
    await this._ensureEventSubscription();
    const turnId = this._conversationId;
    this._messages.push({ role: "user", content: text.trim(), thinking: "" });
    this._messages.push(this._newStreamMessage());
    this._streaming = true;
    this._stickToBottom = true;
    this._armTurnTimeout();
    this._render();
    try {
      const result = await this._call("ha_agent/chat/send", {
        entry_id: this._entryId,
        conversation_id: turnId,
        text: text.trim(),
      });
      if (result?.history) {
        this._clearTurnTimeout();
        this._closeOpenStreamMessages();
        this._streaming = false;
        this._applyHistory(result.history);
        await Promise.all([this._loadThreads(), this._loadPendingDraft()]);
        await this._refreshStatus();
        this._render();
      } else if (!result?.started) {
        this._clearTurnTimeout();
        this._closeOpenStreamMessages();
        this._streaming = false;
        await this._loadHistory();
        this._render();
      }
    } catch (err) {
      this._clearTurnTimeout();
      this._closeOpenStreamMessages();
      this._streaming = false;
      this._messages.push({
        role: "assistant",
        content: `Error: ${err?.message || err}`,
        thinking: "",
      });
      this._render();
    }
  }

  _styles() {
    return `
      :host { display: block; height: 100%; font-family: var(--ha-font-family-body); }
      .wrap { display: flex; flex-direction: column; height: 100%; padding: 16px; box-sizing: border-box; }
      .header { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
      .header-spacer { flex: 1; min-width: 8px; }
      .reload-notice { font-size: 0.9em; opacity: 0.85; }
      .header-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
      .header-actions button:disabled { opacity: 0.55; cursor: not-allowed; }
      .tabs { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
      .tab, button, select, input, textarea {
        font: inherit;
      }
      .tab {
        border: 1px solid var(--divider-color, #ccc);
        background: var(--card-background-color, #fff);
        padding: 8px 12px;
        border-radius: 8px;
        cursor: pointer;
      }
      .tab.active { background: var(--primary-color); color: var(--text-primary-color, #fff); }
      .panel { flex: 1; min-height: 0; overflow: auto; overscroll-behavior: contain; overflow-anchor: none; border: 1px solid var(--divider-color, #ccc); border-radius: 12px; padding: 12px; }
      .activity-hint { margin-top: 10px; opacity: 0.75; font-size: 0.9rem; }
      .skill-notice {
        padding: 10px 12px;
        border-radius: 8px;
        background: color-mix(in srgb, var(--primary-color) 14%, transparent);
        margin-bottom: 12px;
      }
      .skill-detail, .skill-editor {
        margin-top: 16px;
        padding: 14px;
        border-radius: 10px;
        border: 1px solid var(--divider-color, #444);
        background: var(--card-background-color, #1c1c1c);
      }
      .skill-detail h3, .skill-editor h3 { margin: 0 0 8px; }
      .skill-meta { opacity: 0.8; font-size: 0.9rem; margin-bottom: 10px; }
      .skill-body {
        white-space: pre-wrap;
        line-height: 1.45;
        margin: 10px 0;
        padding: 10px;
        border-radius: 8px;
        background: color-mix(in srgb, var(--primary-text-color, #fff) 4%, transparent);
      }
      .skill-triggers { margin: 8px 0 0; padding-left: 1.2em; }
      .skill-tool-steps {
        margin-top: 10px;
        padding: 10px;
        border-radius: 8px;
        overflow-x: auto;
        font-family: var(--code-font-family, monospace);
        font-size: 0.85em;
        background: color-mix(in srgb, var(--primary-text-color, #fff) 6%, transparent);
      }
      .panel.chat-panel { overflow: hidden; display: flex; flex-direction: column; padding: 0; }
      .chat-layout { display: grid; grid-template-columns: ${this._narrow ? "1fr" : "200px 1fr"}; gap: 12px; flex: 1; min-height: 0; height: 100%; padding: 12px; box-sizing: border-box; }
      .thread-sidebar { display: flex; flex-direction: column; gap: 8px; min-height: 0; }
      .thread-search {
        width: 100%;
        box-sizing: border-box;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid var(--divider-color, #444);
        background: var(--card-background-color, #1c1c1c);
      }
      .thread-list { display: flex; flex-direction: column; gap: 6px; overflow-y: auto; min-height: 0; flex: 1; }
      .thread-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 4px;
        align-items: stretch;
      }
      .thread-row .thread { min-width: 0; }
      .thread-delete {
        border: 1px solid var(--divider-color, #444);
        background: transparent;
        color: var(--primary-text-color, #e0e0e0);
        border-radius: 8px;
        width: 2rem;
        cursor: pointer;
        opacity: 0.65;
        padding: 0;
      }
      .thread-delete:hover {
        opacity: 1;
        border-color: var(--error-color, #cf6679);
        color: var(--error-color, #cf6679);
      }
      .thread-snippet {
        display: block;
        font-size: 0.78rem;
        opacity: 0.7;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .thread-list-title {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.65;
        margin: 0;
      }
      .thread {
        padding: 8px 10px;
        border-radius: 8px;
        cursor: pointer;
        border: 1px solid var(--divider-color, #444);
        background: var(--card-background-color, #1c1c1c);
        font-size: 0.9rem;
        line-height: 1.3;
      }
      .thread-title {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .thread:hover { border-color: var(--primary-color); }
      .thread.active {
        border-color: var(--primary-color);
        background: color-mix(in srgb, var(--primary-color) 18%, transparent);
      }
      .chat-main { display: flex; flex-direction: column; min-height: 0; height: 100%; overflow: hidden; }
      .messages {
        display: flex;
        flex-direction: column;
        gap: 10px;
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        overscroll-behavior: contain;
        overflow-anchor: none;
        padding-right: 4px;
      }
      .bubble { padding: 10px 12px; border-radius: 12px; max-width: 90%; word-break: break-word; }
      .bubble.user { align-self: flex-end; background: var(--primary-color); color: var(--text-primary-color, #fff); white-space: pre-wrap; }
      .assistant-turn {
        align-self: stretch;
        display: flex;
        flex-direction: column;
        gap: 6px;
        max-width: 100%;
      }
      .skill-badge {
        align-self: flex-start;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: var(--primary-color);
        background: color-mix(in srgb, var(--primary-color) 14%, transparent);
        border: 1px solid color-mix(in srgb, var(--primary-color) 35%, transparent);
      }
      .turn-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        align-items: center;
        align-self: stretch;
        width: 100%;
        padding: 8px 10px;
        border-radius: 10px;
        background: color-mix(in srgb, var(--primary-color) 8%, transparent);
        border: 1px solid color-mix(in srgb, var(--primary-color) 22%, transparent);
      }
      .turn-meta-heading {
        flex: 0 0 100%;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--primary-color);
        margin-bottom: 2px;
      }
      .turn-meta-chip {
        display: inline-flex;
        align-items: baseline;
        gap: 4px;
        max-width: 100%;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.72rem;
        line-height: 1.3;
        color: var(--secondary-text-color, #aaa);
        background: color-mix(in srgb, var(--primary-text-color, #fff) 6%, transparent);
        border: 1px solid var(--divider-color, #444);
      }
      .turn-meta-label {
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.65rem;
        opacity: 0.85;
      }
      .turn-meta-value {
        color: var(--primary-text-color, #e0e0e0);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 18rem;
      }
      .bubble.assistant { align-self: stretch; background: var(--secondary-background-color, #2a2a2a); color: var(--primary-text-color, #e0e0e0); }
      .bubble.assistant .md p { margin: 0.45em 0; }
      .bubble.assistant .md p:first-child { margin-top: 0; }
      .bubble.assistant .md p:last-child { margin-bottom: 0; }
      .bubble.assistant .md ul,
      .bubble.assistant .md ol { margin: 0.45em 0; padding-left: 1.35em; }
      .bubble.assistant .md ul ul,
      .bubble.assistant .md ol ul,
      .bubble.assistant .md ul ol { margin: 0.25em 0 0.35em; }
      .bubble.assistant .md li { margin: 0.3em 0; }
      .bubble.assistant .md strong { font-weight: 600; }
      .bubble.assistant .md em { font-style: italic; }
      .bubble.assistant .md code {
        font-family: var(--code-font-family, monospace);
        font-size: 0.92em;
        background: color-mix(in srgb, var(--primary-text-color, #fff) 12%, transparent);
        padding: 0.1em 0.35em;
        border-radius: 4px;
      }
      .bubble.assistant .md a { color: var(--primary-color); }
      .bubble.typing { opacity: 0.7; font-style: italic; }
      .empty-chat {
        align-self: center;
        margin: auto;
        max-width: 28rem;
        text-align: center;
        opacity: 0.75;
        line-height: 1.5;
        padding: 24px 12px;
      }
      .thinking-panel {
        border: 1px solid var(--divider-color, #444);
        border-radius: 10px;
        background: color-mix(in srgb, var(--primary-text-color, #fff) 5%, transparent);
        overflow: hidden;
      }
      .thinking-toggle {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        padding: 8px 10px;
        border: 0;
        background: transparent;
        color: var(--primary-text-color, #e0e0e0);
        font: inherit;
        text-align: left;
        cursor: pointer;
      }
      .thinking-toggle:hover {
        background: color-mix(in srgb, var(--primary-color) 10%, transparent);
      }
      .thinking-toggle-icon {
        color: var(--primary-color);
        font-size: 0.85em;
        flex: 0 0 auto;
      }
      .thinking-toggle-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.85;
        flex: 0 0 auto;
      }
      .thinking-preview {
        opacity: 0.65;
        font-size: 0.85em;
        font-style: italic;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 0;
      }
      .thinking-body {
        padding: 0 10px 10px 28px;
        opacity: 0.78;
        font-size: 0.88em;
        line-height: 1.45;
        white-space: pre-wrap;
        border-top: 1px solid color-mix(in srgb, var(--divider-color, #444) 70%, transparent);
      }
      .thinking-panel.collapsed .thinking-body {
        display: none;
      }
      .thinking-panel.collapsed .thinking-toggle {
        border-bottom: 0;
      }
      .thinking-panel.expanded .thinking-toggle {
        border-bottom: 1px solid color-mix(in srgb, var(--divider-color, #444) 70%, transparent);
      }
      .tools-stack {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .tool-call {
        border-radius: 10px;
        border: 1px solid var(--divider-color, #444);
        background: color-mix(in srgb, var(--primary-text-color, #fff) 4%, transparent);
        font-size: 0.85em;
        overflow: hidden;
      }
      .tool-call--start { border-color: var(--primary-color); }
      .tool-call--done { border-color: #4caf50; }
      .tool-call--error { border-color: var(--error-color, #cf6679); }
      .tool-call-toggle {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        padding: 8px 10px;
        border: 0;
        background: transparent;
        color: var(--primary-text-color, #e0e0e0);
        font: inherit;
        text-align: left;
        cursor: pointer;
        flex-wrap: wrap;
      }
      .tool-call-toggle:hover {
        background: color-mix(in srgb, var(--primary-color) 8%, transparent);
      }
      .tool-call-toggle-icon {
        color: var(--primary-color);
        font-size: 0.85em;
        flex: 0 0 auto;
      }
      .tool-call-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.8;
        flex: 0 0 auto;
      }
      .tool-call--start .tool-call-label { color: var(--primary-color); }
      .tool-call--done .tool-call-label { color: #4caf50; }
      .tool-call--error .tool-call-label { color: var(--error-color, #cf6679); }
      .tool-call-name {
        font-family: var(--code-font-family, monospace);
        font-size: 0.92em;
        word-break: break-all;
      }
      .tool-call-preview {
        opacity: 0.65;
        font-size: 0.85em;
        font-style: italic;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 0;
        flex: 1 1 120px;
      }
      .tool-call-body {
        padding: 0 10px 10px;
        border-top: 1px solid color-mix(in srgb, var(--divider-color, #444) 70%, transparent);
      }
      .tool-call.collapsed .tool-call-body {
        display: none;
      }
      .tool-call.expanded .tool-call-toggle {
        border-bottom: 1px solid color-mix(in srgb, var(--divider-color, #444) 70%, transparent);
      }
      .tool-args {
        margin: 6px 0 0;
        padding: 8px;
        border-radius: 6px;
        background: color-mix(in srgb, var(--primary-text-color, #fff) 6%, transparent);
        overflow-x: auto;
        white-space: pre-wrap;
        font-size: 0.85em;
      }
      .tool-detail {
        margin-top: 6px;
        opacity: 0.85;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .composer { display: flex; gap: 8px; margin-top: 12px; flex-shrink: 0; }
      .composer input { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid var(--divider-color, #ccc); }
      table { width: 100%; border-collapse: collapse; }
      th, td { text-align: left; padding: 8px; border-bottom: 1px solid var(--divider-color, #ddd); vertical-align: top; }
      tr.active-skill-row { background: color-mix(in srgb, var(--primary-color) 12%, transparent); }
      .status-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
      .chip { padding: 6px 10px; border-radius: 999px; background: var(--secondary-background-color, #eee); font-size: 0.9em; }
      .banner { padding: 10px; border-radius: 8px; background: #fff4d6; margin-bottom: 12px; }
      .form-grid { display: grid; gap: 10px; }
      .form-grid label { display: grid; gap: 4px; }
      .discover-section {
        gap: 12px;
      }
      .discover-section .stacked-field {
        display: grid;
        gap: 6px;
      }
      .discover-section .stacked-field input {
        width: 100%;
        box-sizing: border-box;
      }
      .discover-options {
        display: grid;
        gap: 8px;
      }
      .checkbox-inline {
        display: flex;
        align-items: flex-start;
        gap: 8px;
      }
      .field-hint {
        opacity: 0.7;
        font-size: 0.85rem;
        font-weight: normal;
      }
      .discover-proposals {
        margin: 0;
        padding-left: 0;
        list-style: none;
      }
      .discover-proposals li {
        margin-bottom: 12px;
        padding: 10px 12px;
        border: 1px solid var(--divider-color, #ccc);
        border-radius: 8px;
      }
      .discover-proposals label {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 8px;
        align-items: start;
      }
      .model-id-short,
      .discover-proposals strong {
        word-break: break-all;
        font-size: 0.92rem;
      }
      .discover-progress {
        position: relative;
        height: 10px;
        border-radius: 999px;
        background: color-mix(in srgb, var(--primary-color) 12%, transparent);
        overflow: hidden;
      }
      .discover-progress-bar {
        height: 100%;
        background: var(--primary-color);
        transition: width 0.3s ease;
      }
      .discover-progress-label {
        display: block;
        margin-top: 4px;
        font-size: 0.85rem;
        opacity: 0.8;
      }
      .eval-page {
        display: grid;
        gap: 16px;
      }
      .eval-section {
        padding: 14px 16px;
        border: 1px solid var(--divider-color, #ccc);
        border-radius: 12px;
        background: color-mix(in srgb, var(--card-background-color, #111) 92%, transparent);
      }
      .eval-section > h3 {
        margin: 0 0 12px;
        font-size: 1rem;
      }
      .eval-section > h4 {
        margin: 16px 0 8px;
        font-size: 0.95rem;
      }
      .eval-section > h4:first-child {
        margin-top: 0;
      }
      .eval-status-line {
        margin: 0 0 8px;
      }
      .eval-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
      }
      .eval-stat {
        padding: 6px 10px;
        border-radius: 999px;
        background: var(--secondary-background-color, #222);
        font-size: 0.85rem;
      }
      .eval-action-groups {
        display: grid;
        gap: 12px;
      }
      .eval-action-group {
        display: grid;
        gap: 8px;
      }
      .eval-action-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.65;
      }
      .eval-action-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .eval-action-row button {
        margin: 0;
      }
      .eval-table-wrap {
        overflow-x: auto;
      }
      .eval-table-wrap table {
        min-width: 640px;
      }
      .eval-section .activity-hint {
        margin-top: 0;
        margin-bottom: 8px;
      }
      .eval-progress-panel {
        margin: 10px 0 4px;
        padding: 10px 12px;
        border-radius: 8px;
        background: color-mix(in srgb, var(--primary-color) 8%, transparent);
        border: 1px solid color-mix(in srgb, var(--primary-color) 20%, transparent);
      }
      .eval-progress-phase {
        margin: 0 0 4px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.75;
      }
      .eval-progress-message {
        margin: 0 0 8px;
      }
      .eval-section.discover-section {
        margin-top: 0;
        padding-top: 14px;
        border-top: none;
      }
      textarea { min-height: 120px; }
      .actions { display: flex; gap: 8px; flex-wrap: wrap; }
      .thread-sidebar.hidden { display: none; }
    `;
  }

  _renderAssistantBody(content) {
    const trimmed = String(content || "").trim();
    if (!trimmed) return "";
    const md = this._formatMarkdown(trimmed);
    if (md.trim()) {
      return `<div class="md">${md}</div>`;
    }
    return `<div class="md"><p>${this._escape(trimmed)}</p></div>`;
  }

  _renderChat() {
    const threads = this._threads
      .map((t) => {
        const snippet = t.snippet
          ? `<span class="thread-snippet">${this._escape(t.snippet)}</span>`
          : "";
        return `
      <div class="thread-row">
        <div class="thread ${t.conversation_id === this._conversationId ? "active" : ""}"
             data-thread="${t.conversation_id}">
          <div class="thread-title">${t.pinned ? "📌 " : ""}${this._escape(t.title || t.conversation_id)}</div>
          ${snippet}
        </div>
        <button class="thread-delete" data-delete-thread="${t.conversation_id}" title="Delete chat" ${this._streaming ? "disabled" : ""}>×</button>
      </div>`;
      })
      .join("");

    const messageListHtml = this._renderMessageListHtml();

    const draft = this._pendingDraft
      ? `<div class="banner">Pending skill from last turn.
         <div class="actions">
           <button data-action="confirm-draft" ${this._streaming ? "disabled" : ""}>Save skill</button>
           <button data-action="dismiss-draft" ${this._streaming ? "disabled" : ""}>Dismiss</button>
         </div></div>`
      : "";

    const skillNotice = this._skillSaveNotice
      ? `<div class="banner skill-notice">${this._escape(this._skillSaveNotice)}</div>`
      : "";

    const emptyThreads = this._threadSearch.trim()
      ? "No chats match your search."
      : "No chats yet";

    return `
      <div class="chat-layout">
        <div class="thread-sidebar ${this._narrow ? "hidden" : ""}">
          <p class="thread-list-title">Chats</p>
          <button data-action="new-thread" ${this._streaming ? "disabled" : ""}>New chat</button>
          <input
            id="thread-search"
            class="thread-search"
            type="search"
            placeholder="Search chats…"
            value="${this._escape(this._threadSearch)}"
            ${this._streaming ? "disabled" : ""}
          />
          <div class="thread-list">${threads || `<div class="thread">${emptyThreads}</div>`}</div>
        </div>
        <div class="chat-main">
          ${skillNotice}
          ${draft}
          <div class="messages">${messageListHtml}</div>
          <div class="composer">
            <input id="chat-input" placeholder="Message HA Agent..." ${this._streaming ? "disabled" : ""} />
            <button data-action="send" ${this._streaming ? "disabled" : ""}>Send</button>
            <button data-action="clear-history" ${this._streaming ? "disabled" : ""}>Clear</button>
          </div>
        </div>
      </div>`;
  }

  _renderSkillDetail() {
    const skill = this._viewingSkill;
    if (!skill) return "";
    const triggers = (skill.triggers || [])
      .map((t) => `<li>${this._escape(t)}</li>`)
      .join("");
    const toolStepCount = (skill.tool_steps || []).length;
    return `
      <div class="skill-detail">
        <h3>${this._escape(skill.title || "Skill")}</h3>
        <div class="skill-meta">
          ${skill.enabled ? "Enabled" : "Disabled"} · used ${skill.use_count || 0} times
          ${skill.version ? ` · v${skill.version}` : ""}
          ${toolStepCount ? ` · ${toolStepCount} tool step(s)` : ""}
        </div>
        <p>${this._escape(skill.description || "")}</p>
        <div class="skill-body">${this._escape(skill.body || "")}</div>
        ${triggers ? `<ul class="skill-triggers">${triggers}</ul>` : ""}
        <div class="actions">
          <button data-action="skill-detail-edit">Edit</button>
          <button data-action="skill-detail-delete">Delete</button>
          <button data-action="skill-detail-close">Close</button>
        </div>
      </div>`;
  }

  _renderSkillEditor() {
    const skill = this._editingSkill;
    if (!skill) return "";
    const toolStepsJson = this._escape(
      JSON.stringify(skill.tool_steps || [], null, 2)
    );
    return `
      <div class="skill-editor">
        <h3>${skill.id ? "Edit skill" : "New skill"}</h3>
        <div class="form-grid">
          <label>Title<input id="skill-title" value="${this._escape(skill.title || "")}" /></label>
          <label>Description<textarea id="skill-description" rows="2">${this._escape(skill.description || "")}</textarea></label>
          <label>Triggers (one per line)<textarea id="skill-triggers" rows="3">${this._escape((skill.triggers || []).join("\n"))}</textarea></label>
          <label>Workflow<textarea id="skill-body" rows="12">${this._escape(skill.body || "")}</textarea></label>
          <p class="activity-hint">Write markdown steps. Mention tools in backticks (e.g. <code>mcp_news__news_curate</code>) — structured tool steps are derived automatically when you save.</p>
          <div class="actions">
            <button type="button" data-action="skill-derive-tool-steps">Recreate tool steps from workflow</button>
          </div>
          <details class="skill-advanced">
            <summary>Advanced: override tool steps JSON</summary>
            <label><input type="checkbox" id="skill-tool-steps-override" ${(skill.tool_steps || []).length ? "checked" : ""}/> Use manual tool steps instead of auto-derivation</label>
            <label>Tool steps JSON<textarea id="skill-tool-steps" rows="8">${toolStepsJson}</textarea></label>
          </details>
          <label><input type="checkbox" id="skill-enabled" ${skill.enabled !== false ? "checked" : ""}/> Enabled</label>
          <div class="actions">
            <button data-action="skill-save">Save</button>
            <button data-action="skill-cancel">Cancel</button>
          </div>
        </div>
      </div>`;
  }

  _renderSkills() {
    const notice = this._skillNotice
      ? `<div class="skill-notice">${this._escape(this._skillNotice)}</div>`
      : "";
    const rows = this._skills
      .map(
        (s) => `
      <tr class="${this._viewingSkill?.id === s.id ? "active-skill-row" : ""}">
        <td>${this._escape(s.title)}</td>
        <td>${s.enabled ? "Yes" : "No"}</td>
        <td>${s.use_count || 0}</td>
        <td class="actions">
          <button data-skill-view="${s.id}">View</button>
          <button data-skill-edit="${s.id}">Edit</button>
          <button data-skill-toggle="${s.id}">${s.enabled ? "Disable" : "Enable"}</button>
          <button data-skill-delete="${s.id}">Delete</button>
        </td>
      </tr>`
      )
      .join("");

    return `
      ${notice}
      <div class="actions" style="margin-bottom:12px">
        <input id="skill-search" placeholder="Search skills..." />
        <button data-action="skill-search">Search</button>
        <button data-action="skill-new">New skill</button>
        <button data-action="skill-export">Export</button>
        <button data-action="skill-import">Import</button>
      </div>
      <table>
        <thead><tr><th>Title</th><th>Enabled</th><th>Uses</th><th>Actions</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4">No skills yet.</td></tr>'}</tbody>
      </table>
      ${this._renderSkillDetail()}
      ${this._renderSkillEditor()}`;
  }

  _renderPlaybookEditor() {
    const pb = this._editingPlaybook;
    if (!pb) return "";
    const isNew = !pb.route;
    const isBuiltin = pb.is_builtin === true;
    const heading = isNew
      ? "New playbook rule"
      : `Edit playbook · ${this._escape(pb.title || pb.route)}`;
    return `
      <div class="skill-editor">
        <h3>${heading}</h3>
        <div class="form-grid">
          <label>Title<input id="playbook-title" value="${this._escape(pb.title || "")}" /></label>
          <label>When to apply (the model uses this to decide if the rule fires)<textarea id="playbook-match" rows="2">${this._escape(pb.match_text || "")}</textarea></label>
          <label>Workflow text<textarea id="playbook-body" rows="10">${this._escape(pb.body || "")}</textarea></label>
          <label><input type="checkbox" id="playbook-enabled" ${pb.enabled !== false ? "checked" : ""}/> Enabled</label>
          <div class="actions">
            <button data-action="playbook-save">Save</button>
            ${isNew ? "" : isBuiltin ? '<button data-action="playbook-reset">Reset to default</button>' : '<button data-action="playbook-delete">Delete</button>'}
            <button data-action="playbook-cancel">Cancel</button>
          </div>
        </div>
      </div>`;
  }

  _renderPlaybooks() {
    const notice = this._playbookNotice
      ? `<div class="skill-notice">${this._escape(this._playbookNotice)}</div>`
      : "";
    const rows = this._playbooks
      .map(
        (p) => `
      <tr class="${this._editingPlaybook?.route === p.route ? "active-skill-row" : ""}">
        <td>${this._escape(p.title || p.route)}</td>
        <td>${p.is_builtin ? this._escape(p.route) : "custom"}</td>
        <td>${p.enabled ? "Yes" : "No"}</td>
        <td>${p.is_builtin ? (p.is_default ? "Default" : "Customized") : "Custom"}</td>
        <td class="actions">
          <button data-playbook-edit="${this._escape(p.route)}">Edit</button>
          <button data-playbook-toggle="${this._escape(p.route)}">${p.enabled ? "Disable" : "Enable"}</button>
          ${p.is_builtin ? `<button data-playbook-reset="${this._escape(p.route)}">Reset</button>` : `<button data-playbook-delete="${this._escape(p.route)}">Delete</button>`}
        </td>
      </tr>`
      )
      .join("");

    return `
      ${notice}
      <p class="playbook-intro">Playbooks are workflow recipes injected into the prompt. Built-in playbooks map to routes; custom rules you add fire when the model decides their "when to apply" text matches the request. The model only runs that selection when at least one custom rule exists.</p>
      <div class="actions" style="margin-bottom:12px">
        <button data-action="playbook-new">Add playbook</button>
      </div>
      <table>
        <thead><tr><th>Title</th><th>Kind</th><th>Enabled</th><th>Status</th><th>Actions</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="5">No playbooks.</td></tr>'}</tbody>
      </table>
      ${this._renderPlaybookEditor()}`;
  }

  _openPlaybookEditor(route) {
    if (route === null) {
      this._editingPlaybook = {
        route: null,
        title: "",
        match_text: "",
        body: "",
        enabled: true,
        is_builtin: false,
      };
    } else {
      const pb = this._playbooks.find((p) => p.route === route);
      if (!pb) return;
      this._editingPlaybook = { ...pb };
    }
    this._tab = "playbooks";
    this._render();
  }

  _renderRouteEditor() {
    const route = this._editingRoute;
    if (!route) return "";
    const keywords = Array.isArray(route.keywords)
      ? route.keywords.join("\n")
      : "";
    return `
      <div class="skill-editor">
        <h3>Edit route · ${this._escape(route.title || route.route)}</h3>
        <div class="form-grid">
          <label>Trigger keywords (one per line; case-insensitive whole-word match)<textarea id="route-keywords" rows="8">${this._escape(keywords)}</textarea></label>
          <label><input type="checkbox" id="route-enabled" ${route.enabled !== false ? "checked" : ""}/> Use custom keywords (when off, shipped defaults apply)</label>
          <div class="actions">
            <button data-action="route-save">Save</button>
            <button data-action="route-reset">Reset to default</button>
            <button data-action="route-cancel">Cancel</button>
          </div>
        </div>
      </div>`;
  }

  _renderRoutes() {
    const notice = this._routeNotice
      ? `<div class="skill-notice">${this._escape(this._routeNotice)}</div>`
      : "";
    const rows = this._routeKeywords
      .map(
        (r) => `
      <tr class="${this._editingRoute?.route === r.route ? "active-skill-row" : ""}">
        <td>${this._escape(r.title || r.route)}</td>
        <td>${this._escape((r.keywords || []).join(", "))}</td>
        <td>${r.enabled ? "Yes" : "No"}</td>
        <td>${r.is_default ? "Default" : "Customized"}</td>
        <td class="actions">
          <button data-route-edit="${this._escape(r.route)}">Edit</button>
          <button data-route-toggle="${this._escape(r.route)}">${r.enabled ? "Disable" : "Enable"}</button>
          <button data-route-reset="${this._escape(r.route)}">Reset</button>
        </td>
      </tr>`
      )
      .join("");

    return `
      ${notice}
      <p class="playbook-intro">Route keywords decide which built-in workflow (email, news, or device action) a request triggers. Matching is case-insensitive whole-word. When a route's custom keywords are disabled, empty, or unchanged from the default, the shipped matcher is used.</p>
      <table>
        <thead><tr><th>Route</th><th>Keywords</th><th>Custom enabled</th><th>Status</th><th>Actions</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="5">No routes.</td></tr>'}</tbody>
      </table>
      ${this._renderRouteEditor()}`;
  }

  _openRouteEditor(route) {
    const r = this._routeKeywords.find((item) => item.route === route);
    if (!r) return;
    this._editingRoute = { ...r, keywords: [...(r.keywords || [])] };
    this._tab = "routes";
    this._render();
  }

  _renderHintEditor() {
    const hint = this._editingHint;
    if (!hint) return "";
    const isNew = !hint.rule_id;
    const isBuiltin = hint.is_builtin === true;
    const heading = isNew
      ? "New recovery hint"
      : `Edit recovery hint · ${this._escape(hint.title || hint.rule_id)}`;
    return `
      <div class="skill-editor">
        <h3>${heading}</h3>
        <div class="form-grid">
          <label>Title<input id="hint-title" value="${this._escape(hint.title || "")}" /></label>
          <label>Tool-name substring (optional; matches the failed tool name)<input id="hint-tool" value="${this._escape(hint.tool_substring || "")}" /></label>
          <label>Error-text pattern (optional; regex or keyword in the error)<input id="hint-pattern" value="${this._escape(hint.error_pattern || "")}" /></label>
          <label>Hint shown to the model<textarea id="hint-body" rows="6">${this._escape(hint.body || "")}</textarea></label>
          <label><input type="checkbox" id="hint-enabled" ${hint.enabled !== false ? "checked" : ""}/> Enabled</label>
          <div class="actions">
            <button data-action="hint-save">Save</button>
            ${isNew ? "" : isBuiltin ? '<button data-action="hint-reset">Reset to default</button>' : '<button data-action="hint-delete">Delete</button>'}
            <button data-action="hint-cancel">Cancel</button>
          </div>
        </div>
      </div>`;
  }

  _renderRecovery() {
    const notice = this._hintNotice
      ? `<div class="skill-notice">${this._escape(this._hintNotice)}</div>`
      : "";
    const rows = this._recoveryHints
      .map(
        (h) => `
      <tr class="${this._editingHint?.rule_id === h.rule_id ? "active-skill-row" : ""}">
        <td>${this._escape(h.title || h.rule_id)}</td>
        <td>${this._escape(h.tool_substring || "any")}</td>
        <td>${this._escape(h.error_pattern || "any")}</td>
        <td>${h.enabled ? "Yes" : "No"}</td>
        <td>${h.is_builtin ? (h.is_default ? "Default" : "Customized") : "Custom"}</td>
        <td class="actions">
          <button data-hint-edit="${this._escape(h.rule_id)}">Edit</button>
          <button data-hint-toggle="${this._escape(h.rule_id)}">${h.enabled ? "Disable" : "Enable"}</button>
          ${h.is_builtin ? `<button data-hint-reset="${this._escape(h.rule_id)}">Reset</button>` : `<button data-hint-delete="${this._escape(h.rule_id)}">Delete</button>`}
        </td>
      </tr>`
      )
      .join("");

    return `
      ${notice}
      <p class="playbook-intro">Recovery hints are appended to a failed tool result to help the model change strategy. A hint fires when its tool-name substring and error-text pattern both match (blank fields match anything).</p>
      <div class="actions" style="margin-bottom:12px">
        <button data-action="hint-new">Add recovery hint</button>
      </div>
      <table>
        <thead><tr><th>Title</th><th>Tool</th><th>Error pattern</th><th>Enabled</th><th>Status</th><th>Actions</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="6">No recovery hints.</td></tr>'}</tbody>
      </table>
      ${this._renderHintEditor()}`;
  }

  _openHintEditor(ruleId) {
    if (ruleId === null) {
      this._editingHint = {
        rule_id: null,
        title: "",
        tool_substring: "",
        error_pattern: "",
        body: "",
        enabled: true,
        is_builtin: false,
      };
    } else {
      const hint = this._recoveryHints.find((h) => h.rule_id === ruleId);
      if (!hint) return;
      this._editingHint = { ...hint };
    }
    this._tab = "recovery";
    this._render();
  }

  _bindRouteEvents() {
    this.shadowRoot.querySelectorAll("[data-route-edit]").forEach((el) => {
      el.onclick = () => this._openRouteEditor(el.getAttribute("data-route-edit"));
    });

    this.shadowRoot.querySelectorAll("[data-route-toggle]").forEach((el) => {
      el.onclick = async () => {
        const route = el.getAttribute("data-route-toggle");
        const r = this._routeKeywords.find((item) => item.route === route);
        await this._call("ha_agent/route_keywords/set_enabled", {
          entry_id: this._entryId,
          route,
          enabled: !r?.enabled,
        });
        this._routeNotice = `${r?.title || route} custom keywords ${r?.enabled ? "disabled" : "enabled"}.`;
        await this._loadRouteKeywords();
        this._render();
      };
    });

    this.shadowRoot.querySelectorAll("[data-route-reset]").forEach((el) => {
      el.onclick = async () => {
        const route = el.getAttribute("data-route-reset");
        await this._call("ha_agent/route_keywords/reset", {
          entry_id: this._entryId,
          route,
        });
        this._routeNotice = `Reset ${route} keywords to default.`;
        if (this._editingRoute?.route === route) this._editingRoute = null;
        await this._loadRouteKeywords();
        this._render();
      };
    });

    this.shadowRoot
      .querySelector('[data-action="route-save"]')
      ?.addEventListener("click", async () => {
        if (!this._editingRoute) return;
        const raw = this.shadowRoot.querySelector("#route-keywords")?.value || "";
        const keywords = raw
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.length > 0);
        const enabled = this.shadowRoot.querySelector("#route-enabled")?.checked;
        const route = this._editingRoute.route;
        try {
          await this._call("ha_agent/route_keywords/update", {
            entry_id: this._entryId,
            route,
            route_keywords: { keywords, enabled },
          });
          this._routeNotice = `Saved ${this._editingRoute.title || route} keywords.`;
          this._editingRoute = null;
        } catch (err) {
          this._routeNotice = `Could not save route: ${err?.message || err}`;
        }
        await this._loadRouteKeywords();
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="route-reset"]')
      ?.addEventListener("click", async () => {
        if (!this._editingRoute) return;
        const route = this._editingRoute.route;
        await this._call("ha_agent/route_keywords/reset", {
          entry_id: this._entryId,
          route,
        });
        this._routeNotice = `Reset ${route} keywords to default.`;
        this._editingRoute = null;
        await this._loadRouteKeywords();
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="route-cancel"]')
      ?.addEventListener("click", () => {
        this._editingRoute = null;
        this._render();
      });
  }

  _bindRecoveryEvents() {
    this.shadowRoot
      .querySelector('[data-action="hint-new"]')
      ?.addEventListener("click", () => this._openHintEditor(null));

    this.shadowRoot.querySelectorAll("[data-hint-edit]").forEach((el) => {
      el.onclick = () => this._openHintEditor(el.getAttribute("data-hint-edit"));
    });

    this.shadowRoot.querySelectorAll("[data-hint-toggle]").forEach((el) => {
      el.onclick = async () => {
        const ruleId = el.getAttribute("data-hint-toggle");
        const hint = this._recoveryHints.find((h) => h.rule_id === ruleId);
        await this._call("ha_agent/recovery_hints/set_enabled", {
          entry_id: this._entryId,
          rule_id: ruleId,
          enabled: !hint?.enabled,
        });
        this._hintNotice = `${hint?.title || ruleId} ${hint?.enabled ? "disabled" : "enabled"}.`;
        await this._loadRecoveryHints();
        this._render();
      };
    });

    this.shadowRoot.querySelectorAll("[data-hint-reset]").forEach((el) => {
      el.onclick = async () => {
        const ruleId = el.getAttribute("data-hint-reset");
        await this._call("ha_agent/recovery_hints/reset", {
          entry_id: this._entryId,
          rule_id: ruleId,
        });
        this._hintNotice = `Reset ${ruleId} to default.`;
        if (this._editingHint?.rule_id === ruleId) this._editingHint = null;
        await this._loadRecoveryHints();
        this._render();
      };
    });

    this.shadowRoot.querySelectorAll("[data-hint-delete]").forEach((el) => {
      el.onclick = async () => {
        const ruleId = el.getAttribute("data-hint-delete");
        await this._call("ha_agent/recovery_hints/delete", {
          entry_id: this._entryId,
          rule_id: ruleId,
        });
        this._hintNotice = "Deleted custom recovery hint.";
        if (this._editingHint?.rule_id === ruleId) this._editingHint = null;
        await this._loadRecoveryHints();
        this._render();
      };
    });

    this.shadowRoot
      .querySelector('[data-action="hint-save"]')
      ?.addEventListener("click", async () => {
        if (!this._editingHint) return;
        const title = this.shadowRoot.querySelector("#hint-title")?.value || "";
        const tool_substring =
          this.shadowRoot.querySelector("#hint-tool")?.value || "";
        const error_pattern =
          this.shadowRoot.querySelector("#hint-pattern")?.value || "";
        const body = this.shadowRoot.querySelector("#hint-body")?.value || "";
        const enabled = this.shadowRoot.querySelector("#hint-enabled")?.checked;
        const payload = { title, tool_substring, error_pattern, body, enabled };
        try {
          if (this._editingHint.rule_id) {
            await this._call("ha_agent/recovery_hints/update", {
              entry_id: this._entryId,
              rule_id: this._editingHint.rule_id,
              hint: payload,
            });
          } else {
            await this._call("ha_agent/recovery_hints/create", {
              entry_id: this._entryId,
              hint: payload,
            });
          }
          this._hintNotice = `Saved ${title || "recovery hint"}.`;
          this._editingHint = null;
        } catch (err) {
          this._hintNotice = `Could not save recovery hint: ${err?.message || err}`;
        }
        await this._loadRecoveryHints();
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="hint-delete"]')
      ?.addEventListener("click", async () => {
        if (!this._editingHint?.rule_id) return;
        const ruleId = this._editingHint.rule_id;
        await this._call("ha_agent/recovery_hints/delete", {
          entry_id: this._entryId,
          rule_id: ruleId,
        });
        this._hintNotice = "Deleted custom recovery hint.";
        this._editingHint = null;
        await this._loadRecoveryHints();
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="hint-reset"]')
      ?.addEventListener("click", async () => {
        if (!this._editingHint?.rule_id) return;
        const ruleId = this._editingHint.rule_id;
        await this._call("ha_agent/recovery_hints/reset", {
          entry_id: this._entryId,
          rule_id: ruleId,
        });
        this._hintNotice = `Reset ${ruleId} to default.`;
        this._editingHint = null;
        await this._loadRecoveryHints();
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="hint-cancel"]')
      ?.addEventListener("click", () => {
        this._editingHint = null;
        this._render();
      });
  }

  _openSkillEditor(skill = null) {
    this._viewingSkill = null;
    this._editingSkill = skill
      ? { ...skill }
      : {
          title: "",
          description: "",
          triggers: [],
          body:
            "1. Describe what the agent should do in numbered steps.\n" +
            "2. Name tools in backticks, e.g. `mcp_news__news_curate`.\n" +
            "3. Tool steps are derived automatically when you save.",
          tool_steps: [],
          enabled: true,
        };
    this._tab = "skills";
    this._render();
  }

  async _viewSkill(skillId) {
    const data = await this._call("ha_agent/skills/get", {
      entry_id: this._entryId,
      skill_id: skillId,
    });
    this._viewingSkill = data.skill;
    this._editingSkill = null;
    this._render();
  }

  async _deleteSkill(skillId) {
    if (!confirm("Delete this skill permanently?")) return;
    try {
      await this._call("ha_agent/skills/delete", {
        entry_id: this._entryId,
        skill_id: skillId,
      });
      if (this._viewingSkill?.id === skillId) {
        this._viewingSkill = null;
      }
      if (this._editingSkill?.id === skillId) {
        this._editingSkill = null;
      }
      this._skillNotice = "Skill deleted.";
      await this._loadSkills();
      this._render();
    } catch (err) {
      this._skillNotice = `Could not delete skill: ${err?.message || err}`;
      this._render();
    }
  }

  async _deriveSkillToolSteps() {
    const editor = this.shadowRoot.querySelector(".skill-editor");
    if (!editor) return;
    const body = editor.querySelector("#skill-body")?.value || "";
    try {
      const data = await this._call("ha_agent/skills/derive_tool_steps", {
        entry_id: this._entryId,
        body,
      });
      const steps = data.tool_steps || [];
      const stepsField = editor.querySelector("#skill-tool-steps");
      const override = editor.querySelector("#skill-tool-steps-override");
      const advanced = editor.querySelector(".skill-advanced");
      if (stepsField) {
        stepsField.value = JSON.stringify(steps, null, 2);
      }
      if (override) {
        override.checked = true;
      }
      if (advanced) {
        advanced.open = true;
      }
      this._skillNotice =
        steps.length > 0
          ? `Recreated ${steps.length} tool step(s) from workflow. Review and save.`
          : "No tool names found in workflow — add tools in backticks, then try again.";
      const notice = this.shadowRoot.querySelector(".skill-notice");
      if (notice) {
        notice.textContent = this._skillNotice;
      } else {
        this._render();
      }
    } catch (err) {
      this._skillNotice = `Could not derive tool steps: ${err?.message || err}`;
      this._render();
    }
  }

  async _saveSkillEditor() {
    const editor = this.shadowRoot.querySelector(".skill-editor");
    if (!editor || !this._editingSkill) return;
    const override = Boolean(
      editor.querySelector("#skill-tool-steps-override")?.checked
    );
    const payload = {
      title: editor.querySelector("#skill-title")?.value || "",
      description: editor.querySelector("#skill-description")?.value || "",
      triggers: String(editor.querySelector("#skill-triggers")?.value || "")
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      body: editor.querySelector("#skill-body")?.value || "",
      enabled: Boolean(editor.querySelector("#skill-enabled")?.checked),
    };
    if (override) {
      try {
        const toolSteps = JSON.parse(
          editor.querySelector("#skill-tool-steps")?.value || "[]"
        );
        if (!Array.isArray(toolSteps)) {
          throw new Error("Tool steps must be a JSON array");
        }
        payload.tool_steps = toolSteps;
      } catch (err) {
        this._skillNotice = `Invalid tool steps JSON: ${err?.message || err}`;
        this._render();
        return;
      }
    }

    try {
      if (this._editingSkill.id) {
        await this._call("ha_agent/skills/update", {
          entry_id: this._entryId,
          skill_id: this._editingSkill.id,
          skill: payload,
        });
        this._skillNotice = `Updated skill: ${payload.title}.`;
      } else {
        await this._call("ha_agent/skills/create", {
          entry_id: this._entryId,
          skill: payload,
        });
        this._skillNotice = `Created skill: ${payload.title}.`;
      }
      this._editingSkill = null;
      await this._loadSkills();
      this._render();
    } catch (err) {
      this._skillNotice = `Could not save skill: ${err?.message || err}`;
      this._render();
    }
  }

  _renderSettings() {
    const c = this._config || {};
    const s = this._status || {};
    return `
      <div class="status-row">
        <span class="chip">Route: ${this._escape(s.last_route || "—")}</span>
        <span class="chip">LLM: ${s.llm_reachable ? "online" : "offline"}</span>
        <span class="chip">MCP: ${s.mcp_reachable ? "online" : "offline"}</span>
        <span class="chip">Skills: ${s.skills_enabled || 0}/${s.skills_total || 0}</span>
      </div>
      <div class="form-grid">
        <label>Chat model<input data-config="llm_model" value="${this._escape(c.llm_model || "")}" /></label>
        <label>Thinking level
          <select data-config="thinking_level">
            ${["off", "low", "medium", "high", "infinite"]
              .map(
                (lvl) =>
                  `<option value="${lvl}" ${c.thinking_level === lvl ? "selected" : ""}>${lvl}</option>`
              )
              .join("")}
          </select>
        </label>
        <label><input type="checkbox" data-config-bool="classifier_model_enabled" ${c.classifier_model_enabled ? "checked" : ""}/> Use a dedicated playbook classifier model</label>
        <label>Classifier model<input data-config="classifier_llm_model" value="${this._escape(c.classifier_model || "")}" placeholder="defaults to chat model" /></label>
        <label>Classifier base URL<input data-config="classifier_llm_base_url" value="${this._escape(c.classifier_llm_base_url || "")}" placeholder="defaults to chat base URL" /></label>
        <label><input type="checkbox" data-config-bool="email_model_enabled" ${c.email_model_enabled ? "checked" : ""}/> Use a dedicated email-route model</label>
        <label>Email model<input data-config="email_llm_model" value="${this._escape(c.email_model || "")}" placeholder="defaults to chat model" /></label>
        <label>Email base URL<input data-config="email_llm_base_url" value="${this._escape(c.email_llm_base_url || "")}" placeholder="defaults to chat base URL" /></label>
        <label><input type="checkbox" data-config-bool="news_model_enabled" ${c.news_model_enabled ? "checked" : ""}/> Use a dedicated news-route model</label>
        <label>News model<input data-config="news_llm_model" value="${this._escape(c.news_model || "")}" placeholder="defaults to chat model" /></label>
        <label>News base URL<input data-config="news_llm_base_url" value="${this._escape(c.news_llm_base_url || "")}" placeholder="defaults to chat base URL" /></label>
        <label><input type="checkbox" data-config-bool="show_reasoning_in_chat" ${c.show_reasoning_in_chat ? "checked" : ""}/> Show model reasoning in chat</label>
        <label><input type="checkbox" data-config-bool="enable_streaming" ${c.enable_streaming ? "checked" : ""}/> Enable streaming</label>
        <label><input type="checkbox" data-config-bool="skills_learning_enabled" ${c.skills_learning_enabled ? "checked" : ""}/> Skill learning</label>
        <label><input type="checkbox" data-config-bool="skills_auto_save" ${c.skills_auto_save ? "checked" : ""}/> Skill auto-save</label>
        <label><input type="checkbox" data-config-bool="skills_use_enabled" ${c.skills_use_enabled ? "checked" : ""}/> Skill auto-use</label>
        <label><input type="checkbox" data-config-bool="memory_persist" ${c.memory_persist ? "checked" : ""}/> Persist conversation memory</label>
        <label>History turns<input type="number" data-config="history_turns" value="${c.history_turns || 10}" /></label>
        <label>Max skills per turn<input type="number" data-config="skills_max_inject" value="${c.skills_max_inject || 3}" /></label>
        <button data-action="save-config">Save settings</button>
      </div>`;
  }

  _renderActivity() {
    const rows = this._activity
      .map((t) => {
        const tools = (t.tool_calls || [])
          .map((call) => call.toolName || call.name || "tool")
          .join(", ");
        const verify = (t.verification_notes || []).join(" | ");
        const title = [tools, verify].filter(Boolean).join("\n");
        const promotable =
          !t.fallback &&
          !t.tool_errors &&
          (!t.outcome || t.outcome === "success" || t.outcome === "partial");
        const promoteBtn =
          t.timestamp && promotable
            ? `<button data-action="activity-promote" data-timestamp="${t.timestamp}">Promote</button>`
            : "—";
        return `
      <tr title="${this._escape(title)}">
        <td>${t.timestamp ? new Date(t.timestamp * 1000).toLocaleString() : "—"}</td>
        <td>${this._escape(t.route || "—")}</td>
        <td>${this._escape(t.user_text || "")}</td>
        <td>${this._escape(t.outcome || "—")}</td>
        <td>${t.iterations || 0}</td>
        <td>${(t.tool_calls || []).length}</td>
        <td>${t.tool_errors || 0}</td>
        <td>${promoteBtn}</td>
      </tr>`;
      })
      .join("");
    const notice = this._activityNotice
      ? `<p class="activity-hint">${this._escape(this._activityNotice)}</p>`
      : "";
    return `
      ${notice}
      <table>
        <thead><tr><th>Time</th><th>Route</th><th>User</th><th>Outcome</th><th>Iter</th><th>Tools</th><th>Errors</th><th>Eval</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="8">No activity yet.</td></tr>'}</tbody>
      </table>
      <p class="activity-hint">Promote successful turns to custom eval cases. They run with the built-in suite on the next eval (mock MCP, same scoring).</p>`;
  }

  _renderEval() {
    const run = this._evalStatus?.run || {};
    const progress = run.progress || {};
    const recommendation = run.settings_recommendation || {};
    const taskScores = run.task_scores || [];
    const caps = this._evalCapabilities || run.server_capabilities || {};
    const summary = caps.summary || {};
    const settings = (recommendation.recommendations || [])
      .map(
        (item) =>
          `<li><strong>${this._escape(item.setting)}</strong> = ${this._escape(item.value)} — ${this._escape(item.reason || "")}</li>`,
      )
      .join("");
    const assignments = Object.entries(recommendation.model_assignments || {})
      .map(
        ([task, item]) =>
          `<li><strong>${this._escape(task)}</strong>: ${this._escape(item.model || "")} — ${this._escape(item.reason || "")}</li>`,
      )
      .join("");
    const scoreRows = taskScores
      .map(
        (item) => `<tr>
          <td>${this._escape(item.task)}</td>
          <td>${this._escape(item.model)}</td>
          <td>${Number(item.score || 0).toFixed(2)}</td>
          <td>${item.passed_count || 0}/${item.case_count || 0}</td>
          <td>${item.avg_latency_ms ? Math.round(item.avg_latency_ms) : "—"}</td>
        </tr>`,
      )
      .join("");
    const presetIni = recommendation.preset_ini || "";
    const applyMode =
      recommendation.apply_mode || (summary.props_writable ? "props" : "preset");
    const pipeline = this._evalStatus?.pipeline;
    const discover = this._evalStatus?.discover;
    const running = this._evalStatus?.running
      ? pipeline === "discover"
        ? "Discover running"
        : "Eval running"
      : run.status || "idle";
    const evalRunning = Boolean(this._evalStatus?.running && pipeline === "eval");
    const progressPanel = this._renderEvalProgressPanel(progress, evalRunning);
    const statusDetail =
      pipeline === "discover" && discover?.progress?.message
        ? discover.progress.message
        : evalRunning && progress.message
          ? progress.message
          : progress.message
            ? progress.message
            : progress.phase
              ? `${this._evalPhaseLabel(progress.phase)}${progress.model ? ` · ${this._shortModelId(progress.model)}` : ""}`
              : "";
    const loadedModels = caps.loaded_models || [];
    const modelDetails = caps.model_details || [];
    const catalogRows = modelDetails
      .map((item) => {
        const source = item.is_preset ? "preset" : item.source || "—";
        const progressText = this._formatCatalogProgress(item.progress);
        const loaded = loadedModels.includes(item.model_id);
        const canRetry =
          item.failed ||
          item.status === "downloading" ||
          (item.status === "unloaded" && !item.is_preset && item.source !== "preset");
        const retryBtn = canRetry
          ? `<button data-action="discover-retry" data-model-id="${this._escape(item.model_id)}">Retry</button> `
          : "";
        const downloading = item.status === "downloading";
        const evalBtn = downloading
          ? `<button disabled title="Wait for download to finish">Eval</button> `
          : `<button data-action="eval-model" data-model-id="${this._escape(item.model_id)}" data-needs-preload="${loaded ? "false" : "true"}">Eval</button> `;
        const manageBtns = loaded
          ? `<button data-action="eval-unload-model" data-model-id="${this._escape(item.model_id)}">Unload</button> <button data-action="eval-delete-model" data-model-id="${this._escape(item.model_id)}">Delete</button>`
          : item.is_preset
            ? ""
            : `<button data-action="eval-delete-model" data-model-id="${this._escape(item.model_id)}">Delete</button>`;
        const actions = `${evalBtn}${retryBtn}${manageBtns}`;
        return `<tr>
          <td class="model-id-short">${this._escape(this._shortModelId(item.model_id))}</td>
          <td>${this._escape(item.status || "—")}</td>
          <td>${this._escape(source)}</td>
          <td>${this._escape(this._formatModalities(item.input_modalities))}</td>
          <td>${this._escape(this._formatModalities(item.output_modalities))}</td>
          <td>${this._escape(progressText)}</td>
          <td>${item.failed ? "failed" : "—"}</td>
          <td>${actions}</td>
        </tr>`;
      })
      .join("");
    const promotedCases = (this._evalCases || []).filter((item) => item.source === "promoted");
    const promotedRows = promotedCases
      .map(
        (item) => `<tr>
          <td>${this._escape(item.id)}</td>
          <td>${this._escape(item.task)}</td>
          <td>${this._escape(item.user_text || "")}</td>
          <td>${this._escape((item.expected_tool || "—") + (item.expected_text_contains?.length ? ` · text: ${item.expected_text_contains.join(", ")}` : ""))}</td>
          <td><button data-action="eval-case-delete" data-case-id="${this._escape(item.id)}">Delete</button></td>
        </tr>`,
      )
      .join("");
    const serverStats = [
      ["Models", summary.model_count ?? caps.models?.length ?? "—"],
      ["Loaded", summary.loaded_model_count ?? loadedModels.length ?? "—"],
      ["Slots", summary.total_slots ?? summary.max_instances ?? "—"],
      ["n_ctx", summary.n_ctx && summary.n_ctx > 0 ? summary.n_ctx : "—"],
      ["Apply", applyMode],
    ];
    if (summary.router_role) {
      serverStats.push(["Router", summary.router_role]);
    }
    const statChips = serverStats
      .map(
        ([label, value]) =>
          `<span class="eval-stat"><strong>${this._escape(label)}</strong> ${this._escape(String(value))}</span>`,
      )
      .join("");
    return `
      <div class="eval-page">
        <section class="eval-section eval-section-status">
          ${this._evalNotice && !evalRunning ? `<p class="activity-hint">${this._escape(this._evalNotice)}</p>` : ""}
          <p class="eval-status-line">Status: <strong>${this._escape(running)}</strong>${statusDetail && !evalRunning ? ` — ${this._escape(statusDetail)}` : ""}</p>
          ${progressPanel}
          ${run.error ? `<p class="banner">${this._escape(run.error)}</p>` : ""}
          <div class="eval-stats">${statChips}</div>
        </section>

        <section class="eval-section">
          <h3>Actions</h3>
          <div class="eval-action-groups">
            <div class="eval-action-group">
              <span class="eval-action-label">Benchmark</span>
              <div class="eval-action-row">
                <button data-action="eval-probe">Probe server</button>
                <button data-action="eval-start">Run eval suite</button>
                <button data-action="eval-start-preload">Run eval + preload</button>
                <button data-action="eval-cancel">Cancel pipeline</button>
              </div>
            </div>
            <div class="eval-action-group">
              <span class="eval-action-label">Models</span>
              <div class="eval-action-row">
                <button data-action="eval-preload">Preload recommended</button>
                <button data-action="discover-start">Discover models</button>
              </div>
            </div>
            <div class="eval-action-group">
              <span class="eval-action-label">Apply results</span>
              <div class="eval-action-row">
                <button data-action="eval-apply">Apply model picks</button>
                <button data-action="eval-apply-settings">${applyMode === "preset" ? "Copy preset + instructions" : "Apply server settings"}</button>
                <button data-action="eval-copy-preset">Copy preset</button>
              </div>
            </div>
          </div>
        </section>

        <section class="eval-section">
          <h3>Eval results</h3>
          <p class="activity-hint">Benchmarks loaded models by default. Preload or Discover to try catalog models on your router.</p>
          ${recommendation.summary ? `<p>${this._escape(recommendation.summary)}</p>` : ""}
          ${settings ? `<h4>Recommended server settings</h4><ul>${settings}</ul>` : ""}
          ${applyMode === "preset" ? `<p class="activity-hint">Router/Docker: edit the preset file on the llama volume, then restart the container.</p>` : ""}
          ${presetIni ? `<label>llama.cpp preset<textarea readonly rows="8">${this._escape(presetIni)}</textarea></label>` : ""}
          ${assignments ? `<h4>Recommended models per task</h4><ul>${assignments}</ul>` : ""}
          <div class="eval-table-wrap">
            <table>
              <thead><tr><th>Task</th><th>Model</th><th>Score</th><th>Passed</th><th>Latency ms</th></tr></thead>
              <tbody>${scoreRows || '<tr><td colspan="5">No eval results yet.</td></tr>'}</tbody>
            </table>
          </div>
        </section>

        <section class="eval-section">
          <h3>Model catalog</h3>
          <p class="activity-hint">All models known to the llama.cpp router. Use Eval to benchmark one model; Retry for failed downloads.</p>
          <div class="eval-table-wrap">
            <table>
              <thead><tr><th>Model</th><th>Status</th><th>Source</th><th>Input</th><th>Output</th><th>Progress</th><th>Failed</th><th></th></tr></thead>
              <tbody>${catalogRows || '<tr><td colspan="8">Probe the server to load the model catalog.</td></tr>'}</tbody>
            </table>
          </div>
        </section>

        <section class="eval-section">
          <h3>Promoted cases</h3>
          <p class="activity-hint">From Activity → Promote. Run with the built-in suite on the next eval.</p>
          <div class="eval-table-wrap">
            <table>
              <thead><tr><th>ID</th><th>Task</th><th>Prompt</th><th>Expectations</th><th></th></tr></thead>
              <tbody>${promotedRows || '<tr><td colspan="5">No promoted cases yet.</td></tr>'}</tbody>
            </table>
          </div>
        </section>

        ${this._renderDiscoverSection(discover)}
      </div>`;
  }

  _bindActivityEvents() {
    this.shadowRoot.querySelectorAll('[data-action="activity-promote"]').forEach((button) => {
      button.addEventListener("click", async () => {
        const raw = button.getAttribute("data-timestamp");
        if (!raw) return;
        await this._promoteActivityTurn(Number(raw));
      });
    });
  }

  _bindEvalEvents() {
    this.shadowRoot
      .querySelector('[data-action="eval-probe"]')
      ?.addEventListener("click", async () => {
        await this._probeEvalServer();
      });
    this.shadowRoot
      .querySelector('[data-action="eval-start"]')
      ?.addEventListener("click", async () => {
        await this._startEvalRun(false);
      });
    this.shadowRoot
      .querySelector('[data-action="eval-start-preload"]')
      ?.addEventListener("click", async () => {
        await this._startEvalRun(true);
      });
    this.shadowRoot
      .querySelector('[data-action="eval-preload"]')
      ?.addEventListener("click", async () => {
        await this._preloadEvalModels();
      });
    this.shadowRoot
      .querySelector('[data-action="eval-apply"]')
      ?.addEventListener("click", async () => {
        await this._applyEvalRecommendations();
      });
    this.shadowRoot
      .querySelector('[data-action="eval-apply-settings"]')
      ?.addEventListener("click", async () => {
        await this._applyEvalServerSettings();
      });
    this.shadowRoot
      .querySelector('[data-action="eval-copy-preset"]')
      ?.addEventListener("click", async () => {
        await this._copyEvalPreset();
      });
    this.shadowRoot.querySelectorAll('[data-action="eval-model"]').forEach((button) => {
      button.addEventListener("click", async () => {
        await this._evalCatalogModel(
          button.getAttribute("data-model-id"),
          button.getAttribute("data-needs-preload") === "true",
        );
      });
    });
    this.shadowRoot.querySelectorAll('[data-action="eval-unload-model"]').forEach((button) => {
      button.addEventListener("click", async () => {
        await this._unloadEvalModel(button.getAttribute("data-model-id"));
      });
    });
    this.shadowRoot.querySelectorAll('[data-action="eval-delete-model"]').forEach((button) => {
      button.addEventListener("click", async () => {
        await this._deleteEvalModel(button.getAttribute("data-model-id"));
      });
    });
    this.shadowRoot
      .querySelector('[data-action="eval-cancel"]')
      ?.addEventListener("click", async () => {
        await this._cancelPipeline();
      });
    this.shadowRoot
      .querySelector('[data-action="discover-start"]')
      ?.addEventListener("click", async () => {
        await this._startDiscoverPipeline();
      });
    this.shadowRoot
      .querySelector('[data-action="discover-approve-download"]')
      ?.addEventListener("click", async () => {
        await this._approveDiscoverDownloads();
      });
    this.shadowRoot
      .querySelector('[data-action="discover-approve-download-none"]')
      ?.addEventListener("click", async () => {
        await this._approveDiscoverDownloads([]);
      });
    this.shadowRoot.querySelectorAll('[data-action="discover-approve-trial"]').forEach((button) => {
      button.addEventListener("click", async () => {
        await this._approveDiscoverTrial(
          button.getAttribute("data-model-id"),
          button.getAttribute("data-approved") === "true",
        );
      });
    });
    this.shadowRoot.querySelectorAll('[data-action="discover-select-model"]').forEach((input) => {
      input.addEventListener("change", () => {
        const modelId = input.getAttribute("data-model-id");
        if (!modelId) return;
        if (input.checked) this._discoverSelectedModels.add(modelId);
        else this._discoverSelectedModels.delete(modelId);
      });
    });
    this.shadowRoot.querySelectorAll('[data-action="discover-retry"]').forEach((button) => {
      button.addEventListener("click", async () => {
        await this._retryDiscoverModel(button.getAttribute("data-model-id"));
      });
    });
    this.shadowRoot
      .querySelector('[data-action="discover-require-download"]')
      ?.addEventListener("change", (event) => {
        this._discoverRequireDownloadApproval = event.target.checked;
      });
    this.shadowRoot
      .querySelector('[data-action="discover-require-trial"]')
      ?.addEventListener("change", (event) => {
        this._discoverRequireTrialApproval = event.target.checked;
      });
    this.shadowRoot.querySelectorAll('[data-action="eval-case-delete"]').forEach((button) => {
      button.addEventListener("click", async () => {
        await this._deleteEvalCase(button.getAttribute("data-case-id"));
      });
    });
    this.shadowRoot
      .querySelector('[data-field="discover-models-dir"]')
      ?.addEventListener("input", (event) => {
        this._discoverModelsDir = event.target.value;
      });
    this.shadowRoot
      .querySelector('[data-field="discover-webhook-url"]')
      ?.addEventListener("input", (event) => {
        this._discoverWebhookUrl = event.target.value;
      });
  }

  _formatCatalogProgress(progress) {
    if (!progress || typeof progress !== "object") return "—";
    if (progress.percent != null) {
      const value = Number(progress.percent);
      return Number.isFinite(value)
        ? `${Math.round(value <= 1 ? value * 100 : value)}%`
        : "—";
    }
    if (progress.bytes_done != null && progress.bytes_total) {
      return `${Math.round((progress.bytes_done / progress.bytes_total) * 100)}%`;
    }
    return "—";
  }

  _formatModalities(modalities) {
    if (!Array.isArray(modalities) || !modalities.length) return "—";
    return modalities.join(", ");
  }

  _escape(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  _shouldBoldSpan(inner) {
    const text = String(inner || "").trim();
    if (!text) return false;
    if (text.endsWith(":") && text.length <= 72) return true;
    const words = text.split(/\s+/);
    return text.length <= 36 && words.length <= 4;
  }

  _formatInlineMarkdown(text) {
    let html = this._escape(text);
    html = html.replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    html = html.replace(/\*\*([^*]+)\*\*/g, (_, inner) =>
      this._shouldBoldSpan(inner) ? `<strong>${inner}</strong>` : inner
    );
    html = html.replace(/\*([^*]{1,48})\*/g, (_, inner) => {
      if (inner.split(/\s+/).length > 5) return `*${inner}*`;
      return `<em>${inner}</em>`;
    });
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    return html;
  }

  _isSectionHeader(text) {
    const plain = String(text || "").trim();
    const colonIdx = plain.indexOf(":");
    if (colonIdx === -1) return false;
    return plain.slice(colonIdx + 1).trim() === "";
  }

  _listLineMatch(line) {
    const bullet = line.match(/^(\s*)([-*]|\d+\.)\s+(.+)$/);
    if (!bullet) return null;
    const ordered = /^\d+\.$/.test(bullet[2]);
    return {
      indent: Math.floor(bullet[1].length / 2),
      ordered,
      content: bullet[3],
    };
  }

  _nestSectionHeaders(nodes) {
    const nested = [];
    let index = 0;
    while (index < nodes.length) {
      const node = nodes[index];
      if (this._isSectionHeader(node.content) && node.children.length === 0) {
        const children = [];
        index += 1;
        while (
          index < nodes.length &&
          !this._isSectionHeader(nodes[index].content)
        ) {
          children.push(nodes[index]);
          index += 1;
        }
        node.children = children;
        nested.push(node);
      } else {
        nested.push(node);
        index += 1;
      }
    }
    return nested;
  }

  _buildListTree(items) {
    const root = { children: [] };
    const stack = [{ node: root, indent: -1 }];
    for (const item of items) {
      while (stack.length > 1 && stack[stack.length - 1].indent >= item.indent) {
        stack.pop();
      }
      const parent = stack[stack.length - 1].node;
      const node = { content: item.content, children: [] };
      parent.children.push(node);
      stack.push({ node, indent: item.indent });
    }
    return this._nestSectionHeaders(root.children);
  }

  _renderListTree(nodes, ordered = false) {
    if (!nodes.length) return "";
    const tag = ordered ? "ol" : "ul";
    const items = nodes
      .map((node) => {
        const childHtml = node.children.length
          ? this._renderListTree(node.children, false)
          : "";
        return `<li>${this._formatInlineMarkdown(node.content)}${childHtml}</li>`;
      })
      .join("");
    return `<${tag}>${items}</${tag}>`;
  }

  _formatMarkdown(text) {
    const lines = String(text || "").split("\n");
    const blocks = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      const trimmed = line.trim();
      if (!trimmed) {
        index += 1;
        continue;
      }

      const listMatch = this._listLineMatch(line);
      if (listMatch) {
        const items = [];
        let ordered = listMatch.ordered;
        while (index < lines.length) {
          const current = lines[index];
          if (!current.trim()) {
            index += 1;
            break;
          }
          const match = this._listLineMatch(current);
          if (!match) break;
          if (items.length === 0) {
            ordered = match.ordered;
          } else if (match.ordered !== ordered) {
            break;
          }
          items.push(match);
          index += 1;
        }
        blocks.push(this._renderListTree(this._buildListTree(items), ordered));
        continue;
      }

      blocks.push(`<p>${this._formatInlineMarkdown(trimmed)}</p>`);
      index += 1;
    }

    return blocks.join("");
  }

  _render() {
    if (!this.shadowRoot) return;
    const tabs = [
      "chat",
      "skills",
      "playbooks",
      "routes",
      "recovery",
      "settings",
      "eval",
      "activity",
    ];
    const tabLabels = { recovery: "Recovery hints" };
    const tabButtons = tabs
      .map((t) => {
        const label = tabLabels[t] || `${t[0].toUpperCase()}${t.slice(1)}`;
        const busy = this._streaming && t === "chat" ? " …" : "";
        return `<button class="tab ${this._tab === t ? "active" : ""}" data-tab="${t}">${label}${busy}</button>`;
      })
      .join("");

    const savedScroll =
      this._tab === "chat" ? this._captureMessagesScroll() : this._capturePanelScroll();

    let body = "";
    if (this._bootstrapError) {
      body = `<div class="banner">Failed to connect: ${this._escape(this._bootstrapError)}</div>`;
    } else if (this._tab === "chat") body = this._renderChat();
    if (this._tab === "skills") body = this._renderSkills();
    if (this._tab === "playbooks") body = this._renderPlaybooks();
    if (this._tab === "routes") body = this._renderRoutes();
    if (this._tab === "recovery") body = this._renderRecovery();
    if (this._tab === "settings") body = this._renderSettings();
    if (this._tab === "eval") body = this._renderEval();
    if (this._tab === "activity") body = this._renderActivity();

    const panelClass = this._tab === "chat" ? "panel chat-panel" : "panel";

    const hacsEnabled = this._hacsControlsEnabled();
    const headerBusy = this._reloading || this._hacsBusy;
    const hacsButtons = hacsEnabled
      ? `<div class="header-actions">
          <button data-action="hacs-check" title="Force check GitHub for a new version" ${headerBusy ? "disabled" : ""}>Check</button>
          <button data-action="hacs-update" title="Check for updates, download if available. Alt+click to force redownload." ${headerBusy ? "disabled" : ""}>${this._escape(this._hacsUpdateLabel())}</button>
        </div>`
      : "";

    this.shadowRoot.innerHTML = `
      <style>${this._styles()}</style>
      <div class="wrap">
        <div class="header">
          <strong>HA Agent Console</strong>
          <span class="chip">${this._escape(this._config?.title || "")}</span>
          <span class="header-spacer"></span>
          ${this._headerNotice ? `<span class="reload-notice">${this._escape(this._headerNotice)}</span>` : ""}
          ${hacsButtons}
          <button class="header-reload" data-action="reload-integration" title="Reload HA Agent integration (Devices &amp; services)" ${headerBusy ? "disabled" : ""}>Reload</button>
        </div>
        <div class="tabs">${tabButtons}</div>
        <div class="${panelClass}">${body}</div>
      </div>`;

    if (savedScroll) {
      if (this._tab === "chat") {
        this._restoreMessagesScroll(savedScroll);
      } else {
        this._restorePanelScroll(savedScroll);
      }
    }
    if (this._tab === "chat") {
      const messagesEl = this.shadowRoot?.querySelector(".messages");
      if (messagesEl) {
        this._bindMessagesScroll(messagesEl);
        this._bindMessagesInteractions(messagesEl);
      }
    }

    this.shadowRoot.querySelectorAll("[data-tab]").forEach((el) => {
      el.onclick = async () => {
        this._tab = el.getAttribute("data-tab");
        if (this._tab === "activity") await this._loadActivity();
        if (this._tab === "skills") await this._loadSkills();
        if (this._tab === "playbooks") await this._loadPlaybooks();
        if (this._tab === "routes") await this._loadRouteKeywords();
        if (this._tab === "recovery") await this._loadRecoveryHints();
        if (this._tab === "eval") await this._loadEvalStatus();
        this._render();
        if (this._tab === "chat" && (this._streaming || this._findOpenStreamMessage())) {
          this._scheduleChatRender();
        }
      };
    });

    this.shadowRoot.querySelector('[data-action="reload-integration"]')?.addEventListener("click", async () => {
      await this._reloadIntegration();
    });

    this.shadowRoot.querySelector('[data-action="hacs-check"]')?.addEventListener("click", async () => {
      await this._checkHacsUpdate();
    });

    this.shadowRoot.querySelector('[data-action="hacs-update"]')?.addEventListener("click", async (ev) => {
      await this._hacsUpdate(Boolean(ev.altKey));
    });

    const sendBtn = this.shadowRoot.querySelector('[data-action="send"]');
    const input = this.shadowRoot.querySelector("#chat-input");
    if (sendBtn && input) {
      const submit = () => {
        const value = input.value;
        input.value = "";
        this._sendMessage(value);
      };
      sendBtn.onclick = submit;
      input.onkeydown = (ev) => {
        if (ev.key === "Enter") submit();
      };
    }

    this.shadowRoot.querySelector('[data-action="clear-history"]')?.addEventListener("click", async () => {
      await this._call("ha_agent/chat/history/clear", {
        conversation_id: this._conversationId,
      });
      this._messages = [];
      this._render();
    });

    this.shadowRoot.querySelector('[data-action="new-thread"]')?.addEventListener("click", async () => {
      this._conversationId = this._newConversationId();
      this._messages = [];
      await this._loadHistory();
      this._render();
    });

    this.shadowRoot.querySelector("#thread-search")?.addEventListener("input", (ev) => {
      this._onThreadSearchInput(ev.target.value);
    });

    this.shadowRoot.querySelectorAll("[data-delete-thread]").forEach((el) => {
      el.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        await this._deleteThread(el.getAttribute("data-delete-thread"));
      });
    });

    this.shadowRoot.querySelectorAll("[data-thread]").forEach((el) => {
      el.onclick = async () => {
        if (this._streaming) return;
        this._conversationId = el.getAttribute("data-thread");
        this._stickToBottom = true;
        await this._loadHistory();
        this._render();
      };
    });

    this.shadowRoot.querySelector('[data-action="confirm-draft"]')?.addEventListener("click", async () => {
      if (this._streaming) return;
      this._skillSaveNotice = null;
      try {
        const data = await this._call("ha_agent/skills/pending_confirm", {
          entry_id: this._entryId,
          conversation_id: this._conversationId,
        });
        this._pendingDraft = null;
        this._skillSaveNotice = `Saved skill: ${data.skill?.title || "Skill"}. Open the Skills tab to review or edit it.`;
        this._skillNotice = this._skillSaveNotice;
        await this._loadSkills();
      } catch (err) {
        this._skillSaveNotice = `Could not save skill: ${err?.message || err}`;
        await this._loadPendingDraft();
      }
      this._render();
    });

    this.shadowRoot.querySelector('[data-action="dismiss-draft"]')?.addEventListener("click", async () => {
      if (this._streaming) return;
      try {
        await this._call("ha_agent/skills/pending_dismiss", {
          entry_id: this._entryId,
          conversation_id: this._conversationId,
        });
        this._pendingDraft = null;
        this._skillSaveNotice = "Pending skill dismissed.";
      } catch (err) {
        this._skillSaveNotice = `Could not dismiss skill: ${err?.message || err}`;
      }
      this._render();
    });

    this.shadowRoot.querySelector('[data-action="skill-search"]')?.addEventListener("click", async () => {
      const q = this.shadowRoot.querySelector("#skill-search")?.value || "";
      if (!q.trim()) {
        await this._loadSkills();
      } else {
        const data = await this._call("ha_agent/skills/search", {
          entry_id: this._entryId,
          query: q,
        });
        this._skills = data.skills || [];
      }
      this._render();
    });

    this.shadowRoot.querySelector('[data-action="skill-new"]')?.addEventListener("click", () => {
      this._openSkillEditor();
    });

    this.shadowRoot.querySelector('[data-action="skill-export"]')?.addEventListener("click", async () => {
      const data = await this._call("ha_agent/skills/export", { entry_id: this._entryId });
      const blob = new Blob([JSON.stringify(data.skills, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "ha-agent-skills.json";
      a.click();
      URL.revokeObjectURL(url);
    });

    this.shadowRoot.querySelector('[data-action="skill-import"]')?.addEventListener("click", async () => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = "application/json";
      input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) return;
        const text = await file.text();
        const skills = JSON.parse(text);
        await this._call("ha_agent/skills/import", {
          entry_id: this._entryId,
          skills,
        });
        await this._loadSkills();
        this._render();
      };
      input.click();
    });

    this.shadowRoot.querySelectorAll("[data-skill-toggle]").forEach((el) => {
      el.onclick = async () => {
        const id = el.getAttribute("data-skill-toggle");
        const skill = this._skills.find((s) => s.id === id);
        await this._call("ha_agent/skills/set_enabled", {
          entry_id: this._entryId,
          skill_id: id,
          enabled: !skill?.enabled,
        });
        await this._loadSkills();
        this._render();
      };
    });

    this.shadowRoot.querySelectorAll("[data-skill-delete]").forEach((el) => {
      el.onclick = async () => {
        await this._deleteSkill(el.getAttribute("data-skill-delete"));
      };
    });

    this.shadowRoot.querySelectorAll("[data-skill-view]").forEach((el) => {
      el.onclick = async () => {
        await this._viewSkill(el.getAttribute("data-skill-view"));
      };
    });

    this.shadowRoot.querySelectorAll("[data-skill-edit]").forEach((el) => {
      el.onclick = async () => {
        const id = el.getAttribute("data-skill-edit");
        const data = await this._call("ha_agent/skills/get", {
          entry_id: this._entryId,
          skill_id: id,
        });
        this._openSkillEditor(data.skill);
      };
    });

    this.shadowRoot
      .querySelector('[data-action="skill-detail-edit"]')
      ?.addEventListener("click", () => {
        if (!this._viewingSkill) return;
        this._openSkillEditor(this._viewingSkill);
      });

    this.shadowRoot
      .querySelector('[data-action="skill-detail-delete"]')
      ?.addEventListener("click", async () => {
        if (!this._viewingSkill?.id) return;
        await this._deleteSkill(this._viewingSkill.id);
      });

    this.shadowRoot
      .querySelector('[data-action="skill-detail-close"]')
      ?.addEventListener("click", () => {
        this._viewingSkill = null;
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="skill-derive-tool-steps"]')
      ?.addEventListener("click", () => void this._deriveSkillToolSteps());

    this.shadowRoot
      .querySelector('[data-action="skill-save"]')
      ?.addEventListener("click", () => void this._saveSkillEditor());

    this.shadowRoot
      .querySelector('[data-action="skill-cancel"]')
      ?.addEventListener("click", () => {
        this._editingSkill = null;
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="playbook-new"]')
      ?.addEventListener("click", () => this._openPlaybookEditor(null));

    this.shadowRoot.querySelectorAll("[data-playbook-edit]").forEach((el) => {
      el.onclick = () => this._openPlaybookEditor(el.getAttribute("data-playbook-edit"));
    });

    this.shadowRoot.querySelectorAll("[data-playbook-delete]").forEach((el) => {
      el.onclick = async () => {
        const route = el.getAttribute("data-playbook-delete");
        await this._call("ha_agent/playbooks/delete", {
          entry_id: this._entryId,
          route,
        });
        this._playbookNotice = "Deleted custom playbook.";
        if (this._editingPlaybook?.route === route) this._editingPlaybook = null;
        await this._loadPlaybooks();
        this._render();
      };
    });

    this.shadowRoot.querySelectorAll("[data-playbook-toggle]").forEach((el) => {
      el.onclick = async () => {
        const route = el.getAttribute("data-playbook-toggle");
        const pb = this._playbooks.find((p) => p.route === route);
        await this._call("ha_agent/playbooks/set_enabled", {
          entry_id: this._entryId,
          route,
          enabled: !pb?.enabled,
        });
        this._playbookNotice = `${pb?.title || route} ${pb?.enabled ? "disabled" : "enabled"}.`;
        await this._loadPlaybooks();
        this._render();
      };
    });

    this.shadowRoot.querySelectorAll("[data-playbook-reset]").forEach((el) => {
      el.onclick = async () => {
        const route = el.getAttribute("data-playbook-reset");
        await this._call("ha_agent/playbooks/reset", {
          entry_id: this._entryId,
          route,
        });
        this._playbookNotice = `Reset ${route} playbook to default.`;
        if (this._editingPlaybook?.route === route) this._editingPlaybook = null;
        await this._loadPlaybooks();
        this._render();
      };
    });

    this.shadowRoot
      .querySelector('[data-action="playbook-save"]')
      ?.addEventListener("click", async () => {
        if (!this._editingPlaybook) return;
        const title = this.shadowRoot.querySelector("#playbook-title")?.value || "";
        const match_text = this.shadowRoot.querySelector("#playbook-match")?.value || "";
        const body = this.shadowRoot.querySelector("#playbook-body")?.value || "";
        const enabled = this.shadowRoot.querySelector("#playbook-enabled")?.checked;
        try {
          if (this._editingPlaybook.route) {
            await this._call("ha_agent/playbooks/update", {
              entry_id: this._entryId,
              route: this._editingPlaybook.route,
              playbook: { title, match_text, body, enabled },
            });
          } else {
            await this._call("ha_agent/playbooks/create", {
              entry_id: this._entryId,
              playbook: { title, match_text, body, enabled },
            });
          }
          this._playbookNotice = `Saved ${title || "playbook"}.`;
          this._editingPlaybook = null;
        } catch (err) {
          this._playbookNotice = `Could not save playbook: ${err?.message || err}`;
        }
        await this._loadPlaybooks();
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="playbook-delete"]')
      ?.addEventListener("click", async () => {
        if (!this._editingPlaybook?.route) return;
        const route = this._editingPlaybook.route;
        await this._call("ha_agent/playbooks/delete", {
          entry_id: this._entryId,
          route,
        });
        this._playbookNotice = "Deleted custom playbook.";
        this._editingPlaybook = null;
        await this._loadPlaybooks();
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="playbook-reset"]')
      ?.addEventListener("click", async () => {
        if (!this._editingPlaybook) return;
        const route = this._editingPlaybook.route;
        await this._call("ha_agent/playbooks/reset", {
          entry_id: this._entryId,
          route,
        });
        this._playbookNotice = `Reset ${route} playbook to default.`;
        this._editingPlaybook = null;
        await this._loadPlaybooks();
        this._render();
      });

    this.shadowRoot
      .querySelector('[data-action="playbook-cancel"]')
      ?.addEventListener("click", () => {
        this._editingPlaybook = null;
        this._render();
      });

    this._bindRouteEvents();
    this._bindRecoveryEvents();
    this._bindEvalEvents();
    this._bindActivityEvents();

    this.shadowRoot.querySelector('[data-action="save-config"]')?.addEventListener("click", async () => {
      const updates = {};
      this.shadowRoot.querySelectorAll("[data-config]").forEach((el) => {
        updates[el.getAttribute("data-config")] = el.value;
      });
      this.shadowRoot.querySelectorAll("[data-config-bool]").forEach((el) => {
        updates[el.getAttribute("data-config-bool")] = el.checked;
      });
      const data = await this._call("ha_agent/config/set", {
        entry_id: this._entryId,
        updates,
      });
      this._config = data.config;
      this._render();
    });
  }
}

customElements.define("ha-agent-panel", HaAgentPanel);
