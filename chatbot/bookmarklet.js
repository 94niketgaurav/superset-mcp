/**
 * Superset Chatbot Bookmarklet
 *
 * This script can be converted to a bookmarklet to inject the chatbot
 * into any Superset page.
 *
 * To create a bookmarklet:
 * 1. Minify this code
 * 2. Prefix with "javascript:"
 * 3. Save as a browser bookmark
 */
(function() {
    // Configuration - change this to your chatbot server URL
    var API_URL = 'http://localhost:5050';

    // Check if chatbot is already loaded
    if (window.SupersetChatbot) {
        // Already loaded, just open it
        window.SupersetChatbot.open();
        return;
    }

    // Create script element to load the widget
    var script = document.createElement('script');
    script.src = API_URL + '/static/widget.js';
    script.onload = function() {
        // Initialize chatbot after script loads
        if (window.SupersetChatbot) {
            window.SupersetChatbot.init({ apiUrl: API_URL });
            // Auto-open the chatbot
            setTimeout(function() {
                window.SupersetChatbot.open();
            }, 100);
        }
    };
    script.onerror = function() {
        alert('Failed to load Superset Chatbot. Make sure the chatbot server is running at ' + API_URL);
    };

    document.head.appendChild(script);
})();
