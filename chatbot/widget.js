/**
 * Superset Chatbot Widget
 *
 * A floating chatbot overlay that can be injected into Superset UI.
 * Communicates with the chatbot API server to process natural language queries.
 *
 * Usage:
 * 1. Include this script in your page
 * 2. Call: SupersetChatbot.init({ apiUrl: 'http://localhost:5050' })
 */

(function() {
    'use strict';

    const CHATBOT_CONFIG = {
        apiUrl: 'http://localhost:5050',
        position: 'bottom-right',
        theme: {
            primary: '#1FA8C9',      // Superset blue
            secondary: '#484848',
            background: '#ffffff',
            text: '#484848',
            border: '#e0e0e0'
        }
    };

    // Generate unique session ID
    const SESSION_ID = 'chat_' + Math.random().toString(36).substr(2, 9);

    // Styles
    const STYLES = `
        #superset-chatbot-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 99999;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }

        #superset-chatbot-toggle {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #1FA8C9 0%, #1A85A0 100%);
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(31, 168, 201, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        #superset-chatbot-toggle:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 16px rgba(31, 168, 201, 0.5);
        }

        #superset-chatbot-toggle svg {
            width: 28px;
            height: 28px;
            fill: white;
        }

        #superset-chatbot-window {
            display: none;
            position: absolute;
            bottom: 70px;
            right: 0;
            width: 380px;
            height: 520px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            flex-direction: column;
            overflow: hidden;
        }

        #superset-chatbot-window.open {
            display: flex;
        }

        .chatbot-header {
            background: linear-gradient(135deg, #1FA8C9 0%, #1A85A0 100%);
            color: white;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .chatbot-header h3 {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        }

        .chatbot-header-actions {
            display: flex;
            gap: 8px;
        }

        .chatbot-header button {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }

        .chatbot-header button:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .chatbot-model-select {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            outline: none;
        }

        .chatbot-model-select option {
            background: #1FA8C9;
            color: white;
        }

        .chatbot-chart-types {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 8px;
        }

        .chatbot-chart-type {
            background: white;
            border: 1px solid #e0e0e0;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            text-align: center;
            transition: all 0.2s;
        }

        .chatbot-chart-type:hover {
            border-color: #1FA8C9;
            background: #f8fcfd;
        }

        .chatbot-chart-type.selected {
            border-color: #1FA8C9;
            background: #e8f7fa;
        }

        .chatbot-chart-type .icon {
            font-size: 24px;
            margin-bottom: 4px;
        }

        .chatbot-chart-type .name {
            font-weight: 500;
            font-size: 12px;
        }

        .chatbot-messages {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .chatbot-message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.5;
        }

        .chatbot-message.bot {
            background: #f5f5f5;
            color: #484848;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }

        .chatbot-message.user {
            background: #1FA8C9;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }

        .chatbot-message code {
            background: rgba(0, 0, 0, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
        }

        .chatbot-message pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 8px 0;
            font-size: 12px;
        }

        .chatbot-message pre code {
            background: none;
            padding: 0;
            color: inherit;
        }

        .chatbot-message a {
            color: #1FA8C9;
            text-decoration: none;
            font-weight: 500;
        }

        .chatbot-message a:hover {
            text-decoration: underline;
        }

        .chatbot-choices {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 8px;
        }

        .chatbot-choice {
            background: white;
            border: 1px solid #e0e0e0;
            padding: 10px 14px;
            border-radius: 8px;
            cursor: pointer;
            text-align: left;
            font-size: 13px;
            transition: border-color 0.2s, background 0.2s;
        }

        .chatbot-choice:hover {
            border-color: #1FA8C9;
            background: #f8fcfd;
        }

        .chatbot-choice .label {
            font-weight: 500;
            color: #484848;
        }

        .chatbot-choice .reason {
            font-size: 11px;
            color: #888;
            margin-top: 2px;
        }

        .chatbot-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }

        .chatbot-action {
            background: #1FA8C9;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: background 0.2s;
        }

        .chatbot-action:hover {
            background: #1A85A0;
        }

        .chatbot-action.secondary {
            background: #f5f5f5;
            color: #484848;
        }

        .chatbot-action.secondary:hover {
            background: #e8e8e8;
        }

        .chatbot-input-area {
            padding: 16px;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 8px;
        }

        .chatbot-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }

        .chatbot-input:focus {
            border-color: #1FA8C9;
        }

        .chatbot-input:disabled {
            background: #f5f5f5;
            cursor: not-allowed;
        }

        .chatbot-send {
            background: #1FA8C9;
            color: white;
            border: none;
            width: 44px;
            height: 44px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }

        .chatbot-send:hover {
            background: #1A85A0;
        }

        .chatbot-send:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .chatbot-send svg {
            width: 20px;
            height: 20px;
            fill: white;
        }

        .chatbot-typing {
            display: flex;
            gap: 4px;
            padding: 12px 16px;
        }

        .chatbot-typing span {
            width: 8px;
            height: 8px;
            background: #ccc;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .chatbot-typing span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .chatbot-typing span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
    `;

    // HTML Template
    const HTML_TEMPLATE = `
        <div id="superset-chatbot-container">
            <div id="superset-chatbot-window">
                <div class="chatbot-header">
                    <h3>Query Assistant</h3>
                    <div class="chatbot-header-actions">
                        <select id="chatbot-model-select" class="chatbot-model-select" title="Select AI Model">
                            <option value="openai">GPT-5</option>
                            <option value="claude">Claude</option>
                        </select>
                        <button id="chatbot-reset" title="New conversation">
                            <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                                <path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
                            </svg>
                        </button>
                        <button id="chatbot-close" title="Close">
                            <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="chatbot-messages" id="chatbot-messages">
                    <div class="chatbot-message bot">
                        Hi! I can help you query your data using natural language.
                        Just describe what you're looking for, like:
                        <br><br>
                        <em>"Show me total assets per user"</em>
                        <br>
                        <em>"Count of events by type this month"</em>
                        <br><br>
                        <small>Using: <span id="chatbot-current-model">GPT-4</span></small>
                    </div>
                </div>
                <div class="chatbot-input-area">
                    <input type="text" class="chatbot-input" id="chatbot-input"
                           placeholder="Type a query, or refine the current one..." />
                    <button class="chatbot-send" id="chatbot-send">
                        <svg viewBox="0 0 24 24">
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                        </svg>
                    </button>
                </div>
            </div>
            <button id="superset-chatbot-toggle">
                <svg viewBox="0 0 24 24">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                </svg>
            </button>
        </div>
    `;

    // State
    let isOpen = false;
    let isWaiting = false;

    // API calls
    async function sendMessage(message, action = null, selection = null) {
        const payload = {
            session_id: SESSION_ID,
            message: message || '',
            action: action,
            selection: selection
        };

        const response = await fetch(`${CHATBOT_CONFIG.apiUrl}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        return response.json();
    }

    async function resetSession() {
        await fetch(`${CHATBOT_CONFIG.apiUrl}/api/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: SESSION_ID })
        });
    }

    // UI functions
    function addMessage(content, isUser = false) {
        const messagesEl = document.getElementById('chatbot-messages');
        const messageEl = document.createElement('div');
        messageEl.className = `chatbot-message ${isUser ? 'user' : 'bot'}`;
        messageEl.innerHTML = formatMessage(content);
        messagesEl.appendChild(messageEl);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return messageEl;
    }

    function formatMessage(text) {
        // Convert markdown-like syntax to HTML
        return text
            // Code blocks
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Bold
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            // Links
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
            // Line breaks
            .replace(/\n/g, '<br>');
    }

    function addChoices(choices, messageEl) {
        const choicesEl = document.createElement('div');
        choicesEl.className = 'chatbot-choices';

        choices.forEach((choice, index) => {
            const choiceEl = document.createElement('button');
            choiceEl.className = 'chatbot-choice';
            choiceEl.innerHTML = `
                <div class="label">${choice.label || choice.table_name}</div>
                ${choice.reason ? `<div class="reason">${choice.reason}</div>` : ''}
            `;
            choiceEl.onclick = () => handleChoice(index);
            choicesEl.appendChild(choiceEl);
        });

        messageEl.appendChild(choicesEl);
    }

    function addActions(actions, messageEl) {
        const actionsEl = document.createElement('div');
        actionsEl.className = 'chatbot-actions';

        const actionLabels = {
            'confirm_joins': 'Accept Joins',
            'reject_joins': 'Skip Joins',
            'execute': 'Execute & Save',
            'save_only': 'Save Only',
            'get_url': 'Get URL',
            'cancel': 'Cancel'
        };

        const secondaryActions = ['reject_joins', 'cancel', 'get_url'];

        actions.forEach(action => {
            const btn = document.createElement('button');
            btn.className = `chatbot-action ${secondaryActions.includes(action) ? 'secondary' : ''}`;
            btn.textContent = actionLabels[action] || action;
            btn.onclick = () => handleAction(action);
            actionsEl.appendChild(btn);
        });

        messageEl.appendChild(actionsEl);
    }

    function showTyping() {
        const messagesEl = document.getElementById('chatbot-messages');
        const typingEl = document.createElement('div');
        typingEl.className = 'chatbot-message bot';
        typingEl.id = 'chatbot-typing';
        typingEl.innerHTML = '<div class="chatbot-typing"><span></span><span></span><span></span></div>';
        messagesEl.appendChild(typingEl);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function hideTyping() {
        const typingEl = document.getElementById('chatbot-typing');
        if (typingEl) typingEl.remove();
    }

    function setInputEnabled(enabled) {
        const input = document.getElementById('chatbot-input');
        const send = document.getElementById('chatbot-send');
        input.disabled = !enabled;
        send.disabled = !enabled;
        isWaiting = !enabled;
    }

    // Handlers
    async function handleSend() {
        const input = document.getElementById('chatbot-input');
        const message = input.value.trim();

        if (!message || isWaiting) return;

        input.value = '';
        addMessage(message, true);
        setInputEnabled(false);
        showTyping();

        try {
            const response = await sendMessage(message);
            hideTyping();
            processResponse(response);
        } catch (error) {
            hideTyping();
            addMessage(`Error: ${error.message}`);
            setInputEnabled(true);
        }
    }

    async function handleChoice(index) {
        setInputEnabled(false);
        showTyping();

        try {
            const response = await sendMessage('', 'select', index);
            hideTyping();
            processResponse(response);
        } catch (error) {
            hideTyping();
            addMessage(`Error: ${error.message}`);
            setInputEnabled(true);
        }
    }

    async function handleAction(action) {
        setInputEnabled(false);
        showTyping();

        try {
            const response = await sendMessage('', action);
            hideTyping();
            processResponse(response);
        } catch (error) {
            hideTyping();
            addMessage(`Error: ${error.message}`);
            setInputEnabled(true);
        }
    }

    function processResponse(response) {
        const messageEl = addMessage(response.response);

        // Add choices if present
        if (response.choices && response.choices.length > 0) {
            addChoices(response.choices, messageEl);
        }

        // Add action buttons if present
        if (response.actions && response.actions.length > 0) {
            addActions(response.actions, messageEl);
        }

        // Handle redirect to SQL Lab - auto-open without confirmation
        if (response.redirect && response.sqllab_url) {
            setTimeout(() => {
                window.open(response.sqllab_url, '_blank');
            }, 300);
        }

        // ALWAYS enable input - users should always be able to type
        // They can type a new query, modify the current one, or use buttons
        setInputEnabled(true);
    }

    async function handleReset() {
        await resetSession();
        const messagesEl = document.getElementById('chatbot-messages');
        messagesEl.innerHTML = `
            <div class="chatbot-message bot">
                Conversation reset. What would you like to query?
            </div>
        `;
        setInputEnabled(true);
    }

    function toggleWindow() {
        const window = document.getElementById('superset-chatbot-window');
        isOpen = !isOpen;
        window.classList.toggle('open', isOpen);

        if (isOpen) {
            document.getElementById('chatbot-input').focus();
        }
    }

    // Model switching
    async function setProvider(provider) {
        try {
            await fetch(`${CHATBOT_CONFIG.apiUrl}/api/set-provider`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: SESSION_ID, provider: provider })
            });
            const modelLabel = document.getElementById('chatbot-current-model');
            const labels = { 'openai': 'GPT-5.2', 'claude': 'Claude Sonnet' };
            if (modelLabel) modelLabel.textContent = labels[provider] || provider;
        } catch (error) {
            console.error('Failed to set provider:', error);
        }
    }

    // Load available providers
    async function loadProviders() {
        try {
            const response = await fetch(`${CHATBOT_CONFIG.apiUrl}/api/providers`);
            const data = await response.json();
            const selectEl = document.getElementById('chatbot-model-select');
            const modelLabel = document.getElementById('chatbot-current-model');
            if (selectEl && data.providers) {
                selectEl.innerHTML = '';
                data.providers.forEach(p => {
                    const option = document.createElement('option');
                    option.value = p.id;
                    option.textContent = p.name;
                    option.disabled = !p.available;
                    // Select the default provider
                    if (p.id === data.default) {
                        option.selected = true;
                    }
                    selectEl.appendChild(option);
                });
                // Update model label to show default
                if (modelLabel && data.default) {
                    const defaultProvider = data.providers.find(p => p.id === data.default);
                    if (defaultProvider) {
                        modelLabel.textContent = defaultProvider.model;
                    }
                }
            }
        } catch (error) {
            console.log('Could not load providers:', error);
        }
    }

    // Initialize
    function init(config = {}) {
        // Merge config
        Object.assign(CHATBOT_CONFIG, config);

        // Inject styles
        const styleEl = document.createElement('style');
        styleEl.textContent = STYLES;
        document.head.appendChild(styleEl);

        // Inject HTML
        const containerEl = document.createElement('div');
        containerEl.innerHTML = HTML_TEMPLATE;
        document.body.appendChild(containerEl.firstElementChild);

        // Bind events
        document.getElementById('superset-chatbot-toggle').onclick = toggleWindow;
        document.getElementById('chatbot-close').onclick = toggleWindow;
        document.getElementById('chatbot-reset').onclick = handleReset;
        document.getElementById('chatbot-send').onclick = handleSend;

        document.getElementById('chatbot-input').onkeypress = (e) => {
            if (e.key === 'Enter') handleSend();
        };

        // Model selector
        const modelSelect = document.getElementById('chatbot-model-select');
        if (modelSelect) modelSelect.onchange = (e) => setProvider(e.target.value);

        // Load available providers
        loadProviders();

        console.log('Superset Chatbot initialized');
    }

    // Expose API
    window.SupersetChatbot = {
        init: init,
        open: () => { if (!isOpen) toggleWindow(); },
        close: () => { if (isOpen) toggleWindow(); },
        reset: handleReset,
        setProvider: setProvider
    };

})();