const form = document.getElementById("predictForm");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const data = {
        age: formData.get("age"),
        bmi: formData.get("bmi"),
        children: formData.get("children"),
        smoker: formData.get("smoker")
    };

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();
    if(result.error){
        resultDiv.textContent = "Error: " + result.error;
    } else {
        resultDiv.textContent = "Predicted Insurance Price: $" + result.prediction;
    }
});
