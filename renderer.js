const vegetables = [
  "Tomato","Edo","Squash","Thai Chili","Florida Long Chili",
  "Tomato Round","Tomato Roma","Dosakai","Green Papaya","Orange Papaya",
  "Cilantro","Turai","Tindora","Okra","Karela","Beans","String Beans",
  "Guava","Chasttor","Flat Velor","Parval"
];

let cart = [];
let stream;

const grid = document.getElementById("vegGrid");
const cartDiv = document.getElementById("cart");
const searchInput = document.getElementById("search");

// ================= UI =================

// Render vegetables
function renderVegetables(list) {
  grid.innerHTML = "";
  list.forEach(item => {
    const btn = document.createElement("button");
    btn.innerText = item;
    btn.onclick = () => addToCart(item);
    grid.appendChild(btn);
  });
}

// Add to cart
function addToCart(item) {
  cart.push(item);
  renderCart();
  closeScanner(); // close webcam after selection ✅
}

// Render cart
function renderCart() {
  cartDiv.innerHTML = cart.map((item, i) =>
    `${item} <button onclick="removeItem(${i})">❌</button>`
  ).join("<br>");
}

// Remove item
function removeItem(index) {
  cart.splice(index, 1);
  renderCart();
}

// Search
searchInput.addEventListener("input", () => {
  const value = searchInput.value.toLowerCase();
  const filtered = vegetables.filter(v => v.toLowerCase().includes(value));
  renderVegetables(filtered);
});


// ================= WEBCAM =================

async function openScanner() {
  const scanner = document.getElementById("scanner");
  const video = document.getElementById("video");

  scanner.style.display = "block";

  stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
}


// ================= AI CALL =================

async function capture() {
  const video = document.getElementById("video");
  const predictionDiv = document.getElementById("prediction");

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "image.jpg");

    try {
      predictionDiv.innerHTML = "Checking AI...";

      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      if (!res.ok || data.status === "error") {
        throw new Error(data.message || "Prediction failed");
      }

      predictionDiv.innerHTML = "";

      const title = document.createElement("div");
      title.innerText = "Top predictions:";
      predictionDiv.appendChild(title);

      const items = data.top_predictions || (data.predictions || []).map(label => ({ label }));

      if (!items.length && data.prediction) {
        items.push({ label: data.prediction, confidence: data.confidence });
      }

      items.slice(0, 3).forEach(item => {
        const btn = document.createElement("button");
        const confidenceText = item.confidence !== undefined
          ? ` (${Math.round(item.confidence * 100)}%)`
          : "";
        btn.innerText = `${item.label}${confidenceText} ➕`;
        btn.style.display = "block";
        btn.style.marginTop = "5px";
        btn.onclick = () => addToCart(item.label);
        predictionDiv.appendChild(btn);
      });

    } catch (err) {
      console.error(err);
      predictionDiv.innerText = "Error connecting to AI";
    }

  }, "image/jpeg");
}


// ================= CLOSE =================

function closeScanner() {
  document.getElementById("scanner").style.display = "none";

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
}


// ================= INIT =================

renderVegetables(vegetables);
renderCart();