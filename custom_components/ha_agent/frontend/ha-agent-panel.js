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
    if (this._unsubEvents) {
      void this._unsubEvents();
      this._unsubEvents = null;
    }
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
    await this._ensureEventSubscription();
    const data = await this._call("ha_agent/subscribe", {});
    this._entryId = data.entry_id;
    this._config = data.config;
    this._status = data.status || {};
    await Promise.all([
      this._loadThreads(),
      this._loadSkills(),
      this._loadHistory(),
      this._loadPendingDraft(),
    ]);
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

  async _loadThreads() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/threads/list", {
      entry_id: this._entryId,
    });
    this._threads = data.threads || [];
  }

  async _loadHistory() {
    if (!this._entryId) return;
    const data = await this._call("ha_agent/chat/history/list", {
      entry_id: this._entryId,
      conversation_id: this._conversationId,
    });
    this._messages = (data.history || []).map((item) => ({
      role: item.role,
      content: item.content,
      thinking: "",
    }));
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
      msg = { role: "assistant", content: "", thinking: "" };
      this._messages.push(msg);
    }
    if (data.thinking) {
      msg.thinking += data.thinking;
    }
    if (data.content) {
      msg.content += data.content;
    }
    this._render();
  }

  async _handleChatDone(data) {
    if (data.entry_id && data.entry_id !== this._entryId) return;
    if (data.conversation_id !== this._conversationId) return;
    this._streaming = false;
    if (data.error) {
      this._messages.push({
        role: "assistant",
        content: `Error: ${data.error}`,
        thinking: "",
      });
    } else {
      const last = this._messages[this._messages.length - 1];
      if (!last || last.role !== "assistant" || (!last.content && !last.thinking)) {
        await this._loadHistory();
      }
    }
    this._render();
    await this._loadPendingDraft();
    await this._refreshStatus();
  }

  async _sendMessage(text) {
    if (!text.trim() || this._streaming) return;
    await this._ensureEventSubscription();
    this._messages.push({ role: "user", content: text.trim(), thinking: "" });
    this._streaming = true;
    this._render();
    try {
      await this._call("ha_agent/chat/send", {
        entry_id: this._entryId,
        conversation_id: this._conversationId,
        text: text.trim(),
      });
      await this._loadThreads();
    } catch (err) {
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
      .panel { flex: 1; overflow: auto; border: 1px solid var(--divider-color, #ccc); border-radius: 12px; padding: 12px; }
      .chat-layout { display: grid; grid-template-columns: ${this._narrow ? "1fr" : "220px 1fr"}; gap: 12px; height: 100%; }
      .thread { padding: 8px; border-radius: 8px; cursor: pointer; margin-bottom: 6px; border: 1px solid transparent; }
      .thread.active { border-color: var(--primary-color); background: rgba(0,0,0,0.04); }
      .messages { display: flex; flex-direction: column; gap: 10px; min-height: 200px; }
      .bubble { padding: 10px 12px; border-radius: 12px; max-width: 90%; white-space: pre-wrap; }
      .bubble.user { align-self: flex-end; background: var(--primary-color); color: var(--text-primary-color, #fff); }
      .bubble.assistant { align-self: flex-start; background: var(--secondary-background-color, #f4f4f4); }
      .thinking { opacity: 0.75; font-size: 0.9em; margin-bottom: 6px; border-left: 3px solid var(--primary-color); padding-left: 8px; }
      .composer { display: flex; gap: 8px; margin-top: 12px; }
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
    `;
  }

  _renderChat() {
    const threads = this._threads
      .map(
        (t) => `
      <div class="thread ${t.conversation_id === this._conversationId ? "active" : ""}"
           data-thread="${t.conversation_id}">
        ${t.pinned ? "📌 " : ""}${t.title || t.conversation_id}
      </div>`
      )
      .join("");

    const messages = this._messages
      .map((m) => {
        const thinking = m.thinking
          ? `<div class="thinking">${this._escape(m.thinking)}</div>`
          : "";
        return `<div class="bubble ${m.role}">${thinking}${this._escape(m.content)}</div>`;
      })
      .join("");

    const draft = this._pendingDraft
      ? `<div class="banner">Pending skill from last turn.
         <div class="actions">
           <button data-action="confirm-draft">Save skill</button>
           <button data-action="dismiss-draft">Dismiss</button>
         </div></div>`
      : "";

    return `
      <div class="chat-layout">
        <div>
          <div class="actions" style="margin-bottom:8px">
            <button data-action="new-thread">New chat</button>
          </div>
          ${threads}
        </div>
        <div>
          ${draft}
          <div class="messages">${messages}</div>
          <div class="composer">
            <input id="chat-input" placeholder="Message HA Agent..." ${this._streaming ? "disabled" : ""} />
            <button data-action="send" ${this._streaming ? "disabled" : ""}>Send</button>
            <button data-action="clear-history">Clear</button>
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
        <label><input type="checkbox" data-config-bool="show_reasoning_in_chat" ${c.show_reasoning_in_chat ? "checked" : ""}/> Show reasoning in chat</label>
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

  _render() {
    if (!this.shadowRoot) return;
    const tabs = ["chat", "skills", "settings", "activity"];
    const tabButtons = tabs
      .map(
        (t) =>
          `<button class="tab ${this._tab === t ? "active" : ""}" data-tab="${t}">${t[0].toUpperCase()}${t.slice(1)}</button>`
      )
      .join("");

    let body = "";
    if (this._tab === "chat") body = this._renderChat();
    if (this._tab === "skills") body = this._renderSkills();
    if (this._tab === "settings") body = this._renderSettings();
    if (this._tab === "activity") body = this._renderActivity();

    this.shadowRoot.innerHTML = `
      <style>${this._styles()}</style>
      <div class="wrap">
        <div class="header">
          <strong>HA Agent Console</strong>
          <span class="chip">${this._escape(this._config?.title || "")}</span>
        </div>
        <div class="tabs">${tabButtons}</div>
        <div class="panel">${body}</div>
      </div>`;

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

    this.shadowRoot.querySelectorAll("[data-thread]").forEach((el) => {
      el.onclick = async () => {
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
