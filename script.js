document.getElementById("riskForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const formData = {
    age: document.getElementById("age").value,
    bmi: document.getElementById("bmi").value,
    bloodPressureSystolic: document.getElementById("bloodPressureSystolic").value,
    bloodPressureDiastolic: document.getElementById("bloodPressureDiastolic").value,
    bloodSugar: document.getElementById("bloodSugar").value,
    hemoglobin: document.getElementById("hemoglobin").value,
    proteinInUrine: document.getElementById("proteinInUrine").value,
    hydrationLevel: document.getElementById("hydrationLevel").value,
    dietQuality: document.getElementById("dietQuality").value,
    historyOfDisease: document.getElementById("historyOfDisease").value,
    ironIntake: document.getElementById("ironIntake").value,
    hormoneLevels: document.getElementById("hormoneLevels").value,
    numberOfPregnancies: document.getElementById("numberOfPregnancies").value,
    nauseaIntensity: document.getElementById("nauseaIntensity").value,
    presenceOfPain: document.getElementById("presenceOfPain").value,
    fetalWeight: document.getElementById("fetalWeight").value,
    amnioticFluidIndex: document.getElementById("amnioticFluidIndex").value,
    smokingStatus: document.getElementById("smokingStatus").value,
  };

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });
    const result = await response.json();
    alert(`Prediction: ${JSON.stringify(result)}`);
  } catch (error) {
    console.error("Error:", error);
  }
});
