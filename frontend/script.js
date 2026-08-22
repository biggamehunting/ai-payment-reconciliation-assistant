// Change this if your FastAPI backend runs on a different host/port.
const API_URL = "http://127.0.0.1:8000/api/chat";

// A simple per-browser-tab session id so Gemini keeps conversation context.
const SESSION_ID = crypto.randomUUID();

const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatMessages = document.getElementById("chat-messages");

function addMessage(text, sender) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", sender);

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.textContent = text;

  messageDiv.appendChild(bubble);
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage(message) {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: SESSION_ID }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const data = await response.json();
    return data.reply;
  } catch (error) {
    console.error("Error contacting chatbot API:", error);
    return "Sorry, I couldn't reach the server. Please make sure the backend is running.";
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";

  const submitButton = chatForm.querySelector("button");
  submitButton.disabled = true;

  const reply = await sendMessage(message);
  addMessage(reply, "bot");

  submitButton.disabled = false;
  userInput.focus();
});
