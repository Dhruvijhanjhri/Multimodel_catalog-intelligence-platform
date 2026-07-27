const form = document.getElementById("predictForm");

const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const titleBox = document.getElementById("title");
const resultBox = document.getElementById("result");

// ----------------------
// Image Preview
// ----------------------

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) return;

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

});

// ----------------------
// Auto Detect Title
// ----------------------

imageInput.addEventListener("change", async function () {

    const image = imageInput.files[0];

    if (!image) return;

    const formData = new FormData();
    formData.append("image", image);

    try {

        const response = await fetch("http://127.0.0.1:5000/get-title", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        titleBox.value = data.title;

    } catch (err) {

        console.error(err);

    }

});

// ----------------------
// Predict
// ----------------------

form.addEventListener("submit", async function (e) {

    e.preventDefault();

    const image = imageInput.files[0];
    const title = titleBox.value;

    if (!image) {
        alert("Please choose an image.");
        return;
    }

    // Predict

    const predictForm = new FormData();

    predictForm.append("image", image);
    predictForm.append("title", title);

    const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
            method: "POST",
            body: predictForm
        }
    );

    const result = await response.json();

    // Duplicate Search

    let duplicateResult = { results: [] };

    try {

        const duplicateResponse = await fetch(
            "http://127.0.0.1:8000/find-duplicates",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    query: title,
                    top_k: 5
                })
            }
        );

        if (duplicateResponse.ok) {
            duplicateResult = await duplicateResponse.json();
        }

    } catch (err) {

        console.log(err);

    }

    // Build Duplicate Table

    let duplicateRows = "";

    duplicateResult.results.forEach(item => {

        duplicateRows += `
        <tr>
            <td>${item.title}</td>
            <td>${item.category}</td>
            <td>${item.similarity}</td>
        </tr>
        `;

    });

    // Render Result

    resultBox.innerHTML = `

<div class="row g-3">

    <div class="col-md-6">

        <div class="card shadow-sm">

            <div class="card-body text-center">

                <h6 class="text-muted">
                    📦 Category
                </h6>

                <h2 class="text-primary">
                    ${result.category}
                </h2>

            </div>

        </div>

    </div>

    <div class="col-md-6">

        <div class="card shadow-sm">

            <div class="card-body text-center">

                <h6 class="text-muted">
                    🎯 Confidence
                </h6>

                <h2 class="text-success">
                    ${(result.confidence*100).toFixed(1)}%
                </h2>

                <div class="progress mt-3">

                    <div
                        class="progress-bar bg-success"
                        style="width:${result.confidence*100}%">

                    </div>

                </div>

            </div>

        </div>

    </div>

    <div class="col-md-6">

        <div class="card shadow-sm">

            <div class="card-body text-center">

                <h6 class="text-muted">
                    🖼 Image-Text Similarity
                </h6>

                <h2>

                    ${result.image_title_similarity.toFixed(4)}

                </h2>

            </div>

        </div>

    </div>

    <div class="col-md-6">

        <div class="card shadow-sm">

            <div class="card-body text-center">

                <h6 class="text-muted">

                    Status

                </h6>

                ${
                    result.mismatch

                    ?

                    `<span class="badge bg-danger fs-5">
                        ❌ Mismatch
                    </span>`

                    :

                    `<span class="badge bg-success fs-5">
                        ✅ Match
                    </span>`
                }

            </div>

        </div>

    </div>

</div>

<div class="card shadow mt-4">

<div class="card-body">

<h4 class="mb-3">

🔍 Similar Products

</h4>

<table class="table table-hover align-middle">

<thead>

<tr>

<th>Product</th>

<th>Category</th>

<th>Similarity</th>

</tr>

</thead>

<tbody>

${duplicateRows}

</tbody>

</table>

</div>

</div>

`;

});