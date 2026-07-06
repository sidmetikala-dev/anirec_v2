const usernameInput = document.getElementById("username");
const fetchButton = document.getElementById("fetch");
const results = document.getElementById("results");

function renderMessage(message) {
    results.innerHTML = "";
    const messageElement = document.createElement("p");
    messageElement.className = "status-message";
    messageElement.textContent = message;
    results.appendChild(messageElement);
}

function renderRecommendations(recommendations) {
    results.innerHTML = "";

    recommendations.forEach((title, index) => {
        const card = document.createElement("article");
        card.className = "result-card";

        const heading = document.createElement("h3");
        heading.textContent = `${index + 1}. ${title}`;

        card.appendChild(heading);
        results.appendChild(card);
    });
}

async function fetchData() {
    const username = usernameInput.value.trim();
    if (!username) {
        renderMessage("Enter a MAL username.");
        usernameInput.focus();
        return;
    }

    renderMessage("Loading...");
    fetchButton.disabled = true;

    try {
        const response = await fetch("/recs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username }),
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Could not fetch recommendations.");
        }

        renderRecommendations(data.recommendations || []);
        console.log("From Python:", data);
    } catch (error) {
        renderMessage(`Error: ${error.message}`);
        console.error(error);
    } finally {
        fetchButton.disabled = false;
    }
}

fetchButton.addEventListener("click", fetchData);
usernameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        fetchData();
    }
});
