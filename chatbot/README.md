# Superset Query Assistant Chatbot

A floating chatbot overlay that integrates with Superset to provide natural language query capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SUPERSET UI                             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                     Your Dashboards                     │ │
│  │                                                        │ │
│  │                                                        │ │
│  │                                        ┌─────────────┐ │ │
│  │                                        │  Chatbot    │ │ │
│  │                                        │  Widget     │ │ │
│  │                                        │             │ │ │
│  │                                        │  [Ask...]   │ │ │
│  │                                        └─────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP API
                              ▼
                    ┌───────────────────┐
                    │  api_server.py    │
                    │  (Flask backend)  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  orchestrator.py  │
                    │  (Query flow)     │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  MCP Server       │
                    │  (Superset API)   │
                    └───────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors
```

### 2. Start the MCP Server (if not already running)

Make sure your `.env` file has the required Superset credentials:

```env
SUPERSET_URL=http://your-superset:8088
SUPERSET_USERNAME=admin
SUPERSET_PASSWORD=admin
OPENAI_API_KEY=sk-...
```

### 3. Start the Chatbot API Server

```bash
cd chatbot
python api_server.py
```

The server will start on `http://localhost:5050`

### 4. Test with Demo Page

Open `demo.html` in your browser to test the chatbot.

## Integration with Superset

### Option 1: Bookmarklet (Recommended for Quick Setup)

**Easy Installation:**

1. Start the chatbot server: `python chatbot/api_server.py`
2. Open the installation page: **http://localhost:5050/install**
3. Drag the "SQL Chatbot" button to your bookmarks bar
4. Navigate to Superset and click the bookmarklet

**Manual Installation:**

Create a bookmark with this URL:

```javascript
javascript:(function(){var a='http://localhost:5050';if(window.SupersetChatbot){window.SupersetChatbot.open();return}var s=document.createElement('script');s.src=a+'/static/widget.js';s.onload=function(){if(window.SupersetChatbot){window.SupersetChatbot.init({apiUrl:a});setTimeout(function(){window.SupersetChatbot.open()},100)}};s.onerror=function(){alert('Failed to load Superset Chatbot. Make sure the chatbot server is running at '+a)};document.head.appendChild(s)})();
```

Click the bookmarklet while on any Superset page to load the chatbot.

### Option 2: Superset Custom Template (Permanent)

1. Find your Superset templates directory:
   - Usually: `superset/templates/`
   - Or create: `superset_config_templates/`

2. Create or edit `tail_js_custom_extra.html`:

```html
<script src="http://YOUR_CHATBOT_SERVER:5050/static/widget.js"></script>
<script>
  SupersetChatbot.init({
    apiUrl: 'http://YOUR_CHATBOT_SERVER:5050'
  });
</script>
```

3. In your `superset_config.py`, add:

```python
TEMPLATES_EXTRA_PATH = '/path/to/superset_config_templates'
```

4. Restart Superset

### Option 3: Browser Extension

For development, you can use a browser extension like "User JavaScript and CSS" to inject the widget script on Superset pages.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Bookmarklet installation page |
| `/install` | GET | Alias for installation page |
| `/api/health` | GET | Health check |
| `/api/info` | GET | Server and semantic matcher info |
| `/api/chat` | POST | Main chat endpoint |
| `/api/reset` | POST | Reset chat session |
| `/static/widget.js` | GET | Chatbot widget JavaScript |

### Chat API Request

```json
{
  "session_id": "unique-session-id",
  "message": "Show me total assets per user",
  "action": null,
  "selection": null
}
```

### Chat API Response

```json
{
  "response": "Bot message with **markdown** support",
  "step": "sql_ready",
  "sql": "SELECT ...",
  "sqllab_url": "http://superset/sqllab?...",
  "choices": [...],
  "actions": ["execute", "save_only", "get_url", "cancel"],
  "needs_input": true
}
```

## User Flow

1. **User asks a question** → "Show me users with their asset counts"

2. **Bot analyzes query** → Identifies tables: `user`, `asset`

3. **Bot suggests joins** (if multiple tables) → User confirms

4. **Bot generates SQL** → Using OpenAI with schema context

5. **User chooses action**:
   - **Execute & Save** → Runs query, saves to SQL Lab, redirects
   - **Save Only** → Saves to SQL Lab without running
   - **Get URL** → Provides link to open in SQL Lab

6. **Bot provides SQL Lab URL** → User clicks to view/modify query

## Configuration

Environment variables for `api_server.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_PORT` | 5050 | Port for the API server |
| `SUPERSET_URL` | - | Superset base URL |
| `SUPERSET_USERNAME` | - | Superset username |
| `SUPERSET_PASSWORD` | - | Superset password |
| `OPENAI_API_KEY` | - | OpenAI API key for SQL generation |

## Files

```
chatbot/
├── api_server.py      # Flask backend that wraps orchestrator
├── widget.js          # Floating chatbot UI component
├── bookmarklet.html   # Bookmarklet installation page (served at /install)
├── bookmarklet.js     # Bookmarklet source code
├── demo.html          # Demo/test page
└── README.md          # This file
```

## Customization

### Widget Appearance

Edit the `CHATBOT_CONFIG` and `STYLES` in `widget.js`:

```javascript
const CHATBOT_CONFIG = {
    apiUrl: 'http://localhost:5050',
    position: 'bottom-right',  // or 'bottom-left'
    theme: {
        primary: '#1FA8C9',    // Change to match your branding
        // ...
    }
};
```

### Adding Features

The chatbot flow is controlled in `api_server.py`. Key functions:

- `analyze_query()` - Initial query analysis
- `generate_sql()` - SQL generation with OpenAI
- `execute_query()` - Execute and save to SQL Lab
- `processResponse()` in `widget.js` - Handle bot responses

## Troubleshooting

### "API Disconnected"

1. Make sure `api_server.py` is running
2. Check the port (default 5050)
3. Verify CORS is enabled

### "OpenAI API key not configured"

Set `OPENAI_API_KEY` in your `.env` file

### "Could not identify tables"

Make sure:
1. Semantic matcher is trained (`train_semantic_matcher.py`)
2. Config exists at `config/semantic_config.json`

### Widget not appearing in Superset

1. Check browser console for errors
2. Verify the script URL is accessible
3. Try the bookmarklet method first to test