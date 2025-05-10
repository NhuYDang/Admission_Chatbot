// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const clearButton = document.getElementById('clear-button');
const typingIndicator = document.createElement('div');

// Set up processing indicator
typingIndicator.classList.add('processing-indicator', 'bot-message');
typingIndicator.innerHTML = `
<div class="processing-steps">
  <div class="step"><div class="step-dot"></div><div class="step-text">Tìm kiếm tài liệu liên quan...</div></div>
  <div class="step"><div class="step-dot"></div><div class="step-text">Phân tích nội dung...</div></div>
  <div class="step"><div class="step-dot"></div><div class="step-text">Tổng hợp thông tin...</div></div>
  <div class="step"><div class="step-dot"></div><div class="step-text">Soạn câu trả lời...</div></div>
</div>`;
typingIndicator.style.display = 'none';

// Initialize chat
document.addEventListener('DOMContentLoaded', function() {
    scrollToBottom();
    
    // Send message on Enter key
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Send message on button click
    sendButton.addEventListener('click', sendMessage);
    
    // Clear chat on button click
    if (clearButton) {
        clearButton.addEventListener('click', clearChat);
    }
});

// Function to send a message
function sendMessage() {
    const message = messageInput.value.trim();
    
    if (message === '') {
        return;
    }
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input field
    messageInput.value = '';
    
    // Show typing indicator
    chatMessages.appendChild(typingIndicator);
    typingIndicator.style.display = 'inline-block';
    scrollToBottom();
    
    // Send message to server
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Hide processing indicator
        typingIndicator.style.display = 'none';
        
        // Add bot response to chat
        if (data.error) {
            addMessage('Error: ' + data.error, 'bot', true);
        } else {
            addMessage(data.response, 'bot');
        }
    })
    .catch(error => {
        // Hide processing indicator
        typingIndicator.style.display = 'none';
        
        // Add error message
        addMessage('Sorry, an error occurred while processing your request. Please try again.', 'bot', true);
        console.error('Error:', error);
    });
    
    // Monitor SSE or simulate processing steps
    // For demo, we'll simulate the steps with timeouts
    setTimeout(() => {
        document.querySelectorAll('.processing-steps .step')[0].style.opacity = '1';
    }, 500);
    
    setTimeout(() => {
        document.querySelectorAll('.processing-steps .step')[1].style.opacity = '1';
    }, 1500);
    
    setTimeout(() => {
        document.querySelectorAll('.processing-steps .step')[2].style.opacity = '1';
    }, 2500);
    
    setTimeout(() => {
        document.querySelectorAll('.processing-steps .step')[3].style.opacity = '1';
    }, 3500);
}

// Function to add a message to the chat
function addMessage(text, sender, isError = false) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender + '-message');
    
    if (isError) {
        messageElement.classList.add('text-danger');
    }
    
    // For bot messages, directly use HTML content
    // For user messages, escape HTML and process new lines
    if (sender === 'bot') {
        // Allow HTML content from bot (including tables and formatted text)
        messageElement.innerHTML = text;
    } else {
        // Process text for links and new lines for user messages
        const processedText = text
            .replace(/\n/g, '<br>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
        
        messageElement.innerHTML = processedText;
    }
    
    // Add message to chat
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    scrollToBottom();
}

// Function to scroll to bottom of chat
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to clear chat
function clearChat() {
    // Confirm before clearing
    if (confirm('Are you sure you want to clear the chat history?')) {
        // Clear chat UI
        while (chatMessages.firstChild) {
            chatMessages.removeChild(chatMessages.firstChild);
        }
        
        // Clear chat history on server
        fetch('/clear_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to clear chat history on server');
            }
            return response.json();
        })
        .then(data => {
            console.log('Chat history cleared');
            // Add welcome message
            addMessage('How can I help you with admissions today?', 'bot');
        })
        .catch(error => {
            console.error('Error clearing chat history:', error);
        });
    }
}

// File upload preview (for upload page)
if (document.getElementById('pdf_file')) {
    document.getElementById('pdf_file').addEventListener('change', function(e) {
        const fileName = e.target.files[0]?.name;
        if (fileName) {
            document.getElementById('file-name').textContent = fileName;
            document.getElementById('file-preview').style.display = 'block';
        }
    });
}

// Drag and drop functionality for upload area
if (document.querySelector('.upload-area')) {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.getElementById('pdf_file');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }
    
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        fileInput.files = files;
        
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }
    
    // Click to select file
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
}
