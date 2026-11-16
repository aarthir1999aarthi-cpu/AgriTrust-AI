const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

let cameraStream = null;
let capturedFile = null;

// Image upload
imageInput.addEventListener("change", function () {
    const file = this.files[0];

    capturedFile = null;

    document.getElementById("result").style.display = "none";
    document.getElementById("error").textContent = "";

    if (!file) {
        preview.style.display = "none";
        return;
    }

    if (!["image/jpeg", "image/png"].includes(file.type)) {
        document.getElementById("error").textContent =
            "Please select a JPG or PNG image.";

        imageInput.value = "";
        preview.style.display = "none";
        return;
    }

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
});

// Start camera
async function startCamera() {
    try {
        document.getElementById("error").textContent = "";

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "environment"
            },
            audio: false
        });

        const camera = document.getElementById("camera");
        camera.srcObject = cameraStream;

        document.getElementById("captureBtn").disabled = false;
        document.getElementById("startCameraBtn").disabled = true;
        document.getElementById("stopCameraBtn").disabled = false;

    } catch (error) {
        document.getElementById("error").textContent =
            "Camera access failed. Please allow camera permission and try again.";
    }
}

// Capture image
function captureImage() {
    const camera = document.getElementById("camera");
    const canvas = document.getElementById("canvas");

    if (!camera.videoWidth || !camera.videoHeight) {
        document.getElementById("error").textContent =
            "Camera is not ready. Please wait and try again.";
        return;
    }

    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;

    const context = canvas.getContext("2d");
    context.drawImage(camera, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(function (blob) {
        if (!blob) {
            document.getElementById("error").textContent =
                "Unable to capture image. Please try again.";
            return;
        }

        capturedFile = new File(
            [blob],
            "captured_leaf.jpg",
            { type: "image/jpeg" }
        );

        imageInput.value = "";

        preview.src = URL.createObjectURL(blob);
        preview.style.display = "block";

        document.getElementById("result").style.display = "none";
        document.getElementById("error").textContent =
            "Leaf image captured successfully! Click Predict Disease.";

    }, "image/jpeg", 0.95);
}

// Stop camera
function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    document.getElementById("camera").srcObject = null;

    document.getElementById("captureBtn").disabled = true;
    document.getElementById("startCameraBtn").disabled = false;
    document.getElementById("stopCameraBtn").disabled = true;
}

// Predict disease
async function predictDisease() {
    const file = capturedFile || imageInput.files[0];

    if (!file) {
        document.getElementById("error").textContent =
            "Please upload or capture a leaf image first.";
        return;
    }

    const loading = document.getElementById("loading");
    const result = document.getElementById("result");
    const error = document.getElementById("error");
    const button = document.getElementById("predictBtn");

    const formData = new FormData();
    formData.append("file", file);

    loading.style.display = "block";
    result.style.display = "none";
    error.textContent = "";
    button.disabled = true;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Prediction failed.");
        }

        document.getElementById("crop").textContent = data.crop;
        document.getElementById("disease").textContent = data.disease;
        document.getElementById("confidence").textContent =
            data.confidence + "%";
        document.getElementById("reliability").textContent =
            data.reliability;
        document.getElementById("message").textContent = data.message;
        document.getElementById("disclaimer").textContent = data.disclaimer;

        result.style.display = "block";

    } catch (err) {
        error.textContent = err.message ||
            "Unable to connect to the prediction server.";
    } finally {
        loading.style.display = "none";
        button.disabled = false;
    }
}