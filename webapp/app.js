async function run(e) {
  e.preventDefault();
  
  const MODEL_URL = "./data/model.json";
  const model = await tf.loadLayersModel(MODEL_URL);
  console.log("Model loaded");

  const review = String(document.getElementById("review").value);
  const input = tf.tensor2d([[review]]);
  const result = model.predict(input).arraySync()[0];

  document.getElementById("output").innerHTML = result;
}

run();