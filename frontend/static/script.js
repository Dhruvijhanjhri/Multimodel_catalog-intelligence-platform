// =====================================================
// Enterprise Catalog Intelligence
// Main Script
// =====================================================

// -----------------------------------------------------
// API URLs
// -----------------------------------------------------

const FLASK_URL = "http://127.0.0.1:5000";
const FASTAPI_URL = "http://127.0.0.1:8000";

// -----------------------------------------------------
// DOM Elements
// -----------------------------------------------------

const catalogMode = document.getElementById("catalogMode");
const uploadMode = document.getElementById("uploadMode");

const catalogSection = document.getElementById("catalogSection");
const uploadSection = document.getElementById("uploadSection");

const categorySelect = document.getElementById("categorySelect");
const productSelect = document.getElementById("productSelect");

const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const catalogPreview = document.getElementById("catalogPreview");

const titleBox = document.getElementById("title");

const uploadForm = document.getElementById("predictForm");

const catalogAnalyzeBtn =
document.getElementById("catalogAnalyzeBtn");

// Dashboard

const predictedCategory =
document.getElementById("predictedCategory");

const predictionConfidence =
document.getElementById("predictionConfidence");

const confidenceBar =
document.getElementById("confidenceBar");

const similarityScore =
document.getElementById("similarityScore");

const reviewStatus =
document.getElementById("reviewStatus");

const summaryCategory =
document.getElementById("summaryCategory");

const summaryConfidence =
document.getElementById("summaryConfidence");

const summaryMatch =
document.getElementById("summaryMatch");

const summaryReview =
document.getElementById("summaryReview");

const summarySimilarity =
document.getElementById("summarySimilarity");

const similarProducts =
document.getElementById("similarProducts");

const duplicateTable =
document.getElementById("duplicateTable");

const similarCount =
document.getElementById("similarCount");

const analysisSection =
document.getElementById("analysisSection");

// -----------------------------------------------------
// Global Variables
// -----------------------------------------------------

let selectedProduct = null;

// -----------------------------------------------------
// Source Mode
// -----------------------------------------------------

catalogMode.addEventListener("change", () => {

    catalogSection.style.display = "block";

    uploadSection.style.display = "none";

});

uploadMode.addEventListener("change", () => {

    catalogSection.style.display = "none";

    uploadSection.style.display = "block";

});

// -----------------------------------------------------
// Load Categories
// -----------------------------------------------------

async function loadCategories() {

    categorySelect.innerHTML =
    `<option>Loading...</option>`;

    try {

        const response =
        await fetch(`${FLASK_URL}/categories`);

        const categories =
        await response.json();

        categorySelect.innerHTML =
        `<option value="">Select Category</option>`;

        categories.forEach(category=>{

            const option =
            document.createElement("option");

            option.value = category;

            option.textContent =
            category.replaceAll("_"," ");

            categorySelect.appendChild(option);

        });

    }

    catch(error){

        console.log(error);

        categorySelect.innerHTML =
        `<option>Error Loading</option>`;

    }

}

// -----------------------------------------------------
// Load Products
// -----------------------------------------------------

categorySelect.addEventListener("change", async ()=>{

    productSelect.disabled = true;

    productSelect.innerHTML =
    `<option>Loading...</option>`;

    selectedProduct = null;

    if(categorySelect.value==="")
        return;

    const response =
    await fetch(

        `${FLASK_URL}/products/${categorySelect.value}`

    );

    const products =
    await response.json();

    productSelect.innerHTML =
    `<option value="">Select Product</option>`;

    products.forEach(product=>{

        const option =
        document.createElement("option");

        option.value =
        JSON.stringify(product);

        option.textContent =
        product.title;

        productSelect.appendChild(option);

    });

    productSelect.disabled = false;

});

// -----------------------------------------------------
// Product Selected
// -----------------------------------------------------

productSelect.addEventListener("change", ()=>{

    if(productSelect.value==="")
        return;

    selectedProduct =
    JSON.parse(productSelect.value);

    titleBox.value =
    selectedProduct.title;

    catalogPreview.src =
    `${FLASK_URL}/image/${selectedProduct.image}`;

    catalogPreview.style.display = "block";

});

// -----------------------------------------------------
// Upload Preview
// -----------------------------------------------------

imageInput.addEventListener("change", ()=>{

    if(!imageInput.files.length)
        return;

    preview.src =
    URL.createObjectURL(imageInput.files[0]);

    preview.style.display = "block";

});

// -----------------------------------------------------
// Start
// -----------------------------------------------------

loadCategories();

// =====================================================
// Upload Image
// =====================================================

imageInput.addEventListener("change", async () => {

    const image = imageInput.files[0];

    if (!image)
        return;

    preview.src = URL.createObjectURL(image);

    preview.style.display = "block";

    const formData = new FormData();

    formData.append("image", image);

    try {

        const response = await fetch(

            `${FLASK_URL}/get-title`,

            {

                method: "POST",

                body: formData

            }

        );

        const data = await response.json();

        titleBox.value = data.title;

    }

    catch (err) {

        console.log(err);

    }

});

// =====================================================
// Convert Catalog Image → File
// =====================================================

async function getCatalogImageFile(filename) {

    const response = await fetch(

        `${FLASK_URL}/image/${filename}`

    );

    const blob = await response.blob();

    return new File(

        [blob],

        filename,

        {

            type: blob.type

        }

    );

}

// =====================================================
// Catalog Analyze Button
// =====================================================

catalogAnalyzeBtn.addEventListener("click", async () => {

    if (!selectedProduct) {

        alert("Please select a catalog product.");

        return;

    }

    const image = await getCatalogImageFile(

        selectedProduct.image

    );

    analyzeProduct(

        image,

        selectedProduct.title

    );

});

// =====================================================
// Upload Analyze Button
// =====================================================

uploadForm.addEventListener("submit", async (e) => {

    e.preventDefault();

    const image = imageInput.files[0];

    if (!image) {

        alert("Please choose an image.");

        return;

    }

    analyzeProduct(

        image,

        titleBox.value

    );

});

// =====================================================
// Analyze Product
// =====================================================

async function analyzeProduct(image, title) {

    const button =

        catalogMode.checked

        ? catalogAnalyzeBtn

        : document.getElementById("uploadAnalyzeBtn");

    button.disabled = true;

    button.innerHTML = "Analyzing...";

    try {

        //------------------------------------------------
        // Prediction
        //------------------------------------------------

        const formData = new FormData();

        formData.append("image", image);

        formData.append("title", title);

        const predictResponse = await fetch(

            `${FASTAPI_URL}/predict`,

            {

                method: "POST",

                body: formData

            }

        );

        if (!predictResponse.ok)

            throw new Error("Prediction failed");

        const prediction = await predictResponse.json();

        //------------------------------------------------
        // Duplicate Search
        //------------------------------------------------

        const duplicateResponse = await fetch(

            `${FASTAPI_URL}/find-duplicates`,

            {

                method: "POST",

                headers: {

                    "Content-Type": "application/json"

                },

                body: JSON.stringify({

                    query: title,

                    category: prediction.category,

                    top_k: 5

                })

            }

        );

        let duplicates = [];

        if (duplicateResponse.ok) {

            const duplicateData = await duplicateResponse.json();

            console.log("Duplicate Result:");
            console.log(duplicateData);

            duplicates = duplicateData.results;

        }

        //------------------------------------------------
        // Render Dashboard
        //------------------------------------------------

        renderDashboard(

            prediction,

            duplicates

        );

    }

    catch(error){

        console.error(error);

        alert(

            "Unable to analyze product."

        );

    }

    finally{

        button.disabled = false;

        button.innerHTML =

            catalogMode.checked

            ? "Analyze Catalog Product"

            : "Analyze Product";

    }

}

// =====================================================
// Render Dashboard
// =====================================================

function renderDashboard(prediction, duplicates) {

    analysisSection.style.display = "block";

    //------------------------------------------------
    // Summary Cards
    //------------------------------------------------

    predictedCategory.textContent =
        prediction.category.replaceAll("_", " ");

    predictionConfidence.textContent =
        (prediction.confidence * 100).toFixed(1) + "%";

    confidenceBar.style.width =
        (prediction.confidence * 100) + "%";

    similarityScore.textContent =
        prediction.image_title_similarity.toFixed(4);

    reviewStatus.innerHTML =

        prediction.mismatch || prediction.confidence < 0.70

        ? "<span class='text-danger fw-bold'>Required</span>"

        : "<span class='text-success fw-bold'>Not Required</span>";

    //------------------------------------------------
    // Validation Table
    //------------------------------------------------

    summaryCategory.textContent =
        prediction.category.replaceAll("_"," ");

    summaryConfidence.textContent =
        (prediction.confidence * 100).toFixed(1) + "%";

    summaryMatch.innerHTML =

        prediction.mismatch

        ? "<span class='text-danger'>Mismatch</span>"

        : "<span class='text-success'>Match</span>";

    summaryReview.innerHTML =

        prediction.mismatch || prediction.confidence < 0.70

        ? "<span class='text-danger fw-bold'>Required</span>"

        : "<span class='text-success fw-bold'>Not Required</span>";

    summarySimilarity.textContent =
        prediction.image_title_similarity.toFixed(4);

    //------------------------------------------------
    // Similar Product Cards
    //------------------------------------------------

    similarProducts.innerHTML = "";

    similarCount.textContent =
        `${duplicates.length} Results`;

    duplicates.forEach(item => {

        similarProducts.innerHTML += `

        <div class="card shadow-sm border-0 mb-3">

            <div class="row g-0 align-items-center">

                <div class="col-3 text-center p-2">

                    <img

                        src="${FLASK_URL}/image/${item.image}"

                        class="img-fluid rounded"

                        style="
                            height:90px;
                            width:90px;
                            object-fit:contain;
                        "

                        onerror="this.src='https://placehold.co/90x90?text=No+Image'"

                    >

                </div>

                <div class="col-9">

                    <div class="card-body py-2">

                        <h6 class="fw-bold mb-2">

                            ${item.title}

                        </h6>

                        <span class="badge bg-primary">

                            ${item.category.replaceAll("_"," ")}

                        </span>

                        <span class="badge bg-success ms-2">

                            ${(item.similarity*100).toFixed(2)}% Match

                        </span>

                    </div>

                </div>

            </div>

        </div>

        `;

    });

    //------------------------------------------------
    // Duplicate Table
    //------------------------------------------------

    duplicateTable.innerHTML = "";

    duplicates.forEach(item => {

        duplicateTable.innerHTML += `

        <tr>

            <td>

                ${item.title}

            </td>

            <td>

                ${item.category.replaceAll("_"," ")}

            </td>

            <td>

                ${(item.similarity * 100).toFixed(2)}%

            </td>

        </tr>

        `;

    });

}