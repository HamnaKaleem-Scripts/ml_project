document.getElementById("predictForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const form = new FormData(this);
  const payload = {
    title: form.get("title"),
    tags: form.get("tags"),
    category: form.get("category"),
    duration: parseFloat(form.get("duration")),
    publish_time: form.get("publish_time")
  };

  document.getElementById("result").innerText = "⏳ Predicting...";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (response.ok) {
      document.getElementById("result").innerText = `✅ Predicted Like Ratio: ${data.predicted_like_ratio.toFixed(4)}`;
    } else {
      document.getElementById("result").innerText = `❌ Error: ${data.detail}`;
    }
  } catch (error) {
    console.error("Error:", error);
    document.getElementById("result").innerText = "❌ Failed to connect to backend";
  }
});
