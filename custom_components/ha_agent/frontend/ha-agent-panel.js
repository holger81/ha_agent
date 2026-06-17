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
    this._unsubEvents = null;
    this._eventsReady = null;
    this._bootstrapError = null;
    this._turnTimeout = null;
    this._stickToBottom = true;
    this._chatRenderPending = false;
    this._threadSearch = "";
    this._threadSearchTimer = null;
    this._messagesScrollEl = null;
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

  disconnectedCallback() {
    this._clearTurnTimeout();
    this._clearThreadSearchTimer();
    if (this._messagesScrollEl) {
      this._messagesScrollEl.removeEventListener("scroll", this._onMessagesScroll);
      this._messagesScrollEl = null;
    }
    if (this._unsubEvents) {
      void this._unsubEvents();
      this._unsubEvents = null;
    }
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

  _applyHistory(history) {
    this._messages = (history || []).map((item) => ({
      role: item.role,
      content: item.content,
      thinking: "",
      tools: [],
    }));
  }

  async _loadHistory() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/chat/history/list", {
      entry_id: this._entryId,
      conversation_id: this._conversationId,
    });
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

  _handleDelta(data) {
    if (data.entry_id && data.entry_id !== this._entryId) return;
    if (data.conversation_id !== this._conversationId) return;
    let msg = this._messages[this._messages.length - 1];
    if (!msg || msg.role !== "assistant" || !this._streaming) {
      msg = { role: "assistant", content: "", thinking: "", tools: [] };
      this._messages.push(msg);
    }
    if (data.thinking) {
      msg.thinking += data.thinking;
    }
    if (data.content) {
      msg.content += data.content;
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
      msg.tools[msg.tools.length - 1] = { ...last, ...tool };
      return;
    }
    msg.tools.push({ ...tool });
  }

  _renderToolCall(tool) {
    const phase = tool.phase || "start";
    const labels = { start: "Running", done: "Done", error: "Failed" };
    const label = labels[phase] || phase;
    const name = tool.name || tool.call_name || "tool";
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
      <div class="tool-call tool-call--${phase}">
        <div class="tool-call-header">
          <span class="tool-call-label">${label}</span>
          <code class="tool-call-name">${this._escape(name)}</code>
        </div>
        ${args}
        ${detail}
      </div>`;
  }

  _scheduleChatRender() {
    if (this._chatRenderPending) return;
    this._chatRenderPending = true;
    requestAnimationFrame(() => {
      this._chatRenderPending = false;
      this._render();
    });
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
    el.scrollTop = saved.scrollTop;
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
    this._streaming = false;
    if (data.error) {
      this._messages.push({
        role: "assistant",
        content: `Error: ${data.error}`,
        thinking: "",
      });
    } else {
      await this._loadHistory();
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
    this._streaming = true;
    this._stickToBottom = true;
    this._clearTurnTimeout();
    this._turnTimeout = setTimeout(
      () => void this._recoverStuckTurn(),
      this._turnTimeoutMs()
    );
    this._render();
    try {
      const result = await this._call("ha_agent/chat/send", {
        entry_id: this._entryId,
        conversation_id: turnId,
        text: text.trim(),
      });
      if (result?.history) {
        this._clearTurnTimeout();
        this._applyHistory(result.history);
        this._streaming = false;
        await Promise.all([this._loadThreads(), this._loadPendingDraft()]);
        await this._refreshStatus();
        this._render();
      } else if (!result?.started) {
        this._clearTurnTimeout();
        await this._loadHistory();
        this._streaming = false;
        this._render();
      }
    } catch (err) {
      this._clearTurnTimeout();
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
      .panel { flex: 1; min-height: 0; overflow: auto; border: 1px solid var(--divider-color, #ccc); border-radius: 12px; padding: 12px; }
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
      .bubble.assistant { align-self: flex-start; background: var(--secondary-background-color, #2a2a2a); color: var(--primary-text-color, #e0e0e0); }
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
      .thinking { opacity: 0.75; font-size: 0.9em; margin-bottom: 6px; border-left: 3px solid var(--primary-color); padding-left: 8px; white-space: pre-wrap; }
      .tool-call {
        margin: 8px 0;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid var(--divider-color, #444);
        background: color-mix(in srgb, var(--primary-text-color, #fff) 4%, transparent);
        font-size: 0.85em;
      }
      .tool-call--start { border-color: var(--primary-color); }
      .tool-call--done { border-color: #4caf50; }
      .tool-call--error { border-color: var(--error-color, #cf6679); }
      .tool-call-header {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
      }
      .tool-call-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.8;
      }
      .tool-call--start .tool-call-label { color: var(--primary-color); }
      .tool-call--done .tool-call-label { color: #4caf50; }
      .tool-call--error .tool-call-label { color: var(--error-color, #cf6679); }
      .tool-call-name {
        font-family: var(--code-font-family, monospace);
        font-size: 0.92em;
        word-break: break-all;
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
      .status-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
      .chip { padding: 6px 10px; border-radius: 999px; background: var(--secondary-background-color, #eee); font-size: 0.9em; }
      .banner { padding: 10px; border-radius: 8px; background: #fff4d6; margin-bottom: 12px; }
      .form-grid { display: grid; gap: 10px; }
      .form-grid label { display: grid; gap: 4px; }
      textarea { min-height: 120px; }
      .actions { display: flex; gap: 8px; flex-wrap: wrap; }
      .thread-sidebar.hidden { display: none; }
    `;
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

    const messages = this._messages
      .filter((m) => m.content || m.thinking || (m.tools && m.tools.length))
      .map((m) => {
        const thinking = m.thinking
          ? `<div class="thinking">${this._escape(m.thinking)}</div>`
          : "";
        const tools = (m.tools || [])
          .map((tool) => this._renderToolCall(tool))
          .join("");
        const body =
          m.role === "assistant"
            ? `<div class="md">${this._formatMarkdown(m.content)}</div>`
            : this._escape(m.content);
        return `<div class="bubble ${m.role}">${thinking}${tools}${body}</div>`;
      })
      .join("");

    const typing = this._streaming
      ? '<div class="bubble assistant typing">Thinking…</div>'
      : "";

    const empty = !messages && !typing
      ? `<div class="empty-chat">No messages in this chat yet.${
          this._threads.length ? " History may have been cleared after a restart unless memory persistence is enabled in Settings." : ""
        }</div>`
      : "";

    const draft = this._pendingDraft
      ? `<div class="banner">Pending skill from last turn.
         <div class="actions">
           <button data-action="confirm-draft">Save skill</button>
           <button data-action="dismiss-draft">Dismiss</button>
         </div></div>`
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
          ${draft}
          <div class="messages">${messages}${typing}${empty}</div>
          <div class="composer">
            <input id="chat-input" placeholder="Message HA Agent..." ${this._streaming ? "disabled" : ""} />
            <button data-action="send" ${this._streaming ? "disabled" : ""}>Send</button>
            <button data-action="clear-history" ${this._streaming ? "disabled" : ""}>Clear</button>
          </div>
        </div>
      </div>`;
  }

  _renderSkills() {
    const rows = this._skills
      .map(
        (s) => `
      <tr>
        <td>${this._escape(s.title)}</td>
        <td>${s.enabled ? "Yes" : "No"}</td>
        <td>${s.use_count || 0}</td>
        <td class="actions">
          <button data-skill-view="${s.id}">View</button>
          <button data-skill-toggle="${s.id}">${s.enabled ? "Disable" : "Enable"}</button>
          <button data-skill-edit="${s.id}">Edit</button>
          <button data-skill-delete="${s.id}">Delete</button>
        </td>
      </tr>`
      )
      .join("");

    return `
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
      <div id="skill-editor" hidden></div>`;
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
      .map(
        (t) => `
      <tr>
        <td>${t.timestamp ? new Date(t.timestamp * 1000).toLocaleString() : "—"}</td>
        <td>${this._escape(t.user_text || "")}</td>
        <td>${t.iterations || 0}</td>
        <td>${t.tool_errors || 0}</td>
        <td>${(t.tool_calls || []).length}</td>
      </tr>`
      )
      .join("");
    return `
      <table>
        <thead><tr><th>Time</th><th>User</th><th>Iterations</th><th>Errors</th><th>Tools</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="5">No activity yet.</td></tr>'}</tbody>
      </table>`;
  }

  _escape(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  _formatInlineMarkdown(text) {
    let html = this._escape(text);
    html = html.replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
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
    const tabs = ["chat", "skills", "settings", "activity"];
    const tabButtons = tabs
      .map(
        (t) =>
          `<button class="tab ${this._tab === t ? "active" : ""}" data-tab="${t}">${t[0].toUpperCase()}${t.slice(1)}</button>`
      )
      .join("");

    const savedScroll =
      this._tab === "chat" ? this._captureMessagesScroll() : null;

    let body = "";
    if (this._bootstrapError) {
      body = `<div class="banner">Failed to connect: ${this._escape(this._bootstrapError)}</div>`;
    } else if (this._tab === "chat") body = this._renderChat();
    if (this._tab === "skills") body = this._renderSkills();
    if (this._tab === "settings") body = this._renderSettings();
    if (this._tab === "activity") body = this._renderActivity();

    const panelClass = this._tab === "chat" ? "panel chat-panel" : "panel";

    this.shadowRoot.innerHTML = `
      <style>${this._styles()}</style>
      <div class="wrap">
        <div class="header">
          <strong>HA Agent Console</strong>
          <span class="chip">${this._escape(this._config?.title || "")}</span>
        </div>
        <div class="tabs">${tabButtons}</div>
        <div class="${panelClass}">${body}</div>
      </div>`;

    if (savedScroll) {
      this._restoreMessagesScroll(savedScroll);
    }

    this.shadowRoot.querySelectorAll("[data-tab]").forEach((el) => {
      el.onclick = async () => {
        this._tab = el.getAttribute("data-tab");
        if (this._tab === "activity") await this._loadActivity();
        this._render();
      };
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
        await this._loadHistory();
        this._render();
      };
    });

    this.shadowRoot.querySelector('[data-action="confirm-draft"]')?.addEventListener("click", async () => {
      await this._call("ha_agent/skills/pending_confirm", {
        entry_id: this._entryId,
        conversation_id: this._conversationId,
      });
      this._pendingDraft = null;
      await this._loadSkills();
      this._render();
    });

    this.shadowRoot.querySelector('[data-action="dismiss-draft"]')?.addEventListener("click", async () => {
      await this._call("ha_agent/skills/pending_dismiss", {
        entry_id: this._entryId,
        conversation_id: this._conversationId,
      });
      this._pendingDraft = null;
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
        const id = el.getAttribute("data-skill-delete");
        if (!confirm("Delete this skill?")) return;
        await this._call("ha_agent/skills/delete", {
          entry_id: this._entryId,
          skill_id: id,
        });
        await this._loadSkills();
        this._render();
      };
    });

    this.shadowRoot.querySelectorAll("[data-skill-view]").forEach((el) => {
      el.onclick = async () => {
        const id = el.getAttribute("data-skill-view");
        const data = await this._call("ha_agent/skills/get", {
          entry_id: this._entryId,
          skill_id: id,
        });
        alert(`${data.skill.title}\n\n${data.skill.body}`);
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

  _openSkillEditor(skill = null) {
    const editor = this.shadowRoot.querySelector("#skill-editor");
    if (!editor) return;
    editor.hidden = false;
    editor.innerHTML = `
      <h3>${skill ? "Edit skill" : "New skill"}</h3>
      <div class="form-grid">
        <label>Title<input id="skill-title" value="${this._escape(skill?.title || "")}" /></label>
        <label>Description<textarea id="skill-description">${this._escape(skill?.description || "")}</textarea></label>
        <label>Triggers (one per line)<textarea id="skill-triggers">${this._escape((skill?.triggers || []).join("\n"))}</textarea></label>
        <label>Body<textarea id="skill-body">${this._escape(skill?.body || "")}</textarea></label>
        <label>Tool steps JSON<textarea id="skill-tool-steps">${this._escape(JSON.stringify(skill?.tool_steps || [], null, 2))}</textarea></label>
        <div class="actions">
          <button id="skill-save">Save</button>
          <button id="skill-cancel">Cancel</button>
        </div>
      </div>`;
    editor.querySelector("#skill-cancel").onclick = () => {
      editor.hidden = true;
      editor.innerHTML = "";
      this._render();
    };
    editor.querySelector("#skill-save").onclick = async () => {
      const payload = {
        title: editor.querySelector("#skill-title").value,
        description: editor.querySelector("#skill-description").value,
        triggers: editor.querySelector("#skill-triggers").value.split("\n").map((s) => s.trim()).filter(Boolean),
        body: editor.querySelector("#skill-body").value,
        tool_steps: JSON.parse(editor.querySelector("#skill-tool-steps").value || "[]"),
      };
      if (skill?.id) {
        await this._call("ha_agent/skills/update", {
          entry_id: this._entryId,
          skill_id: skill.id,
          skill: payload,
        });
      } else {
        await this._call("ha_agent/skills/create", {
          entry_id: this._entryId,
          skill: payload,
        });
      }
      editor.hidden = true;
      await this._loadSkills();
      this._render();
    };
  }
}

customElements.define("ha-agent-panel", HaAgentPanel);
