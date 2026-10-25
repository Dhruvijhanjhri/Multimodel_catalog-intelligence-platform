// =====================================================
// Enterprise Catalog Intelligence
// Main Script
// =====================================================

// -----------------------------------------------------
// API URLs
// -----------------------------------------------------

const FLASK_URL = "http://127.0.0.1:5000";
const FASTAPI_URL = "http://127.0.0.1:8001";

// -----------------------------------------------------
// DOM Elements
// -----------------------------------------------------

const catalogSection = document.getElementById("catalogSection");
const uploadSection = document.getElementById("uploadSection");

const categorySelect = document.getElementById("categorySelect");
const productSelect = document.getElementById("productSelect");
const productSearch = document.getElementById("productSearch");
let catalogProducts = [];

const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const catalogPreview = document.getElementById("catalogPreview");

const titleBox = document.getElementById("title");

const uploadForm = document.getElementById("predictForm");

const catalogAnalyzeBtn =
document.getElementById("catalogAnalyzeBtn");

const catalogHeroBtn = document.getElementById("catalogHeroBtn");
const uploadHeroBtn = document.getElementById("uploadHeroBtn");

console.log("Catalog Hero:", catalogHeroBtn);
console.log("Upload Hero :", uploadHeroBtn);

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

// Explainable AI Assessment
const xaiDecision = document.getElementById("xaiDecision");
const xaiTaxonomyStatus = document.getElementById("xaiTaxonomyStatus");
const xaiTaxonomySupport = document.getElementById("xaiTaxonomySupport");
const xaiTaxonomyMargin = document.getElementById("xaiTaxonomyMargin");
const xaiDuplicateScore = document.getElementById("xaiDuplicateScore");
const xaiAlternativeCategory = document.getElementById("xaiAlternativeCategory");
const xaiAlternativeSupport = document.getElementById("xaiAlternativeSupport");
const xaiReviewReasons = document.getElementById("xaiReviewReasons");

// -----------------------------------------------------
// Global Variables
// -----------------------------------------------------

let selectedProduct = null;



//-----------------------------------------------------
// Hero Buttons
//-----------------------------------------------------

catalogHeroBtn.addEventListener("click", () => {

    catalogSection.style.display = "block";
    uploadSection.style.display = "none";

    catalogSection.scrollIntoView({
        behavior: "smooth"
    });

});

uploadHeroBtn.addEventListener("click", () => {

    uploadSection.style.display = "block";
    catalogSection.style.display = "none";

    uploadSection.scrollIntoView({
        behavior: "smooth"
    });

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

function renderProductOptions(searchText = "") {
    const query = searchText.trim().toLowerCase();
    const filteredProducts = catalogProducts.filter(product =>
        (product.title || "").toLowerCase().includes(query)
    );

    productSelect.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = filteredProducts.length
        ? "Select Product"
        : "No matching products";
    productSelect.appendChild(placeholder);

    filteredProducts.forEach(product => {
        const option = document.createElement("option");
        option.value = JSON.stringify(product);
        option.textContent = product.title;
        productSelect.appendChild(option);
    });

    productSelect.disabled = catalogProducts.length === 0;
}

categorySelect.addEventListener("change", async () => {
    productSelect.disabled = true;
    productSearch.disabled = true;
    productSearch.value = "";
    productSelect.innerHTML = '<option value="">Loading...</option>';
    catalogProducts = [];
    selectedProduct = null;

    if (categorySelect.value === "") {
        productSelect.innerHTML = '<option value="">Select Product</option>';
        return;
    }

    try {
        const response = await fetch(
            `${FLASK_URL}/products/${categorySelect.value}`
        );

        if (!response.ok) {
            throw new Error(`Product request failed: ${response.status}`);
        }

        catalogProducts = await response.json();
        renderProductOptions();
        productSearch.disabled = false;
    } catch (error) {
        console.error("Unable to load catalog products:", error);
        productSelect.innerHTML = '<option value="">Error loading products</option>';
        productSelect.disabled = true;
    }
});

productSearch.addEventListener("input", () => {
    selectedProduct = null;
    renderProductOptions(productSearch.value);
});

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

    const isCatalog =
        catalogSection.style.display !== "none";

    const button = isCatalog
        ? document.getElementById("catalogAnalyzeBtn")
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

    catch (error) {

        console.error(error);

        alert("Unable to analyze product.");

    }

    finally {

        button.disabled = false;

        button.innerHTML = isCatalog
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

        // Use the backend decision instead of duplicating its rules.
    const decision = String(prediction.decision ?? "UNKNOWN").toUpperCase();
    const reviewRequired = decision === "REVIEW";

    const formatScore = value => {
        if (value === null || value === undefined || value === "") {
            return "N/A";
        }

        const number = Number(value);
        return Number.isFinite(number) ? number.toFixed(4) : "N/A";
    };

    const displayText = value =>
        value === null || value === undefined || value === ""
            ? "N/A"
            : String(value).replaceAll("_", " ");

    reviewStatus.innerHTML =
        decision === "REVIEW"
            ? "<span class='text-danger fw-bold'>Required</span>"
            : decision === "PASS"
                ? "<span class='text-success fw-bold'>Not Required</span>"
                : "<span class='text-secondary fw-bold'>Unavailable</span>";

    // Validation Table
    summaryCategory.textContent =
        displayText(prediction.category);

    summaryConfidence.textContent =
        (Number(prediction.confidence) * 100).toFixed(1) + "%";

    summaryMatch.innerHTML =
        prediction.mismatch
            ? "<span class='text-danger'>Mismatch</span>"
            : "<span class='text-success'>Match</span>";

    summaryReview.innerHTML =
        decision === "REVIEW"
            ? "<span class='text-danger fw-bold'>Required</span>"
            : decision === "PASS"
                ? "<span class='text-success fw-bold'>Not Required</span>"
                : "<span class='text-secondary fw-bold'>Unavailable</span>";

    summarySimilarity.textContent =
        formatScore(prediction.image_title_similarity);

    // Explainable AI Assessment
    xaiDecision.textContent = decision;
    xaiTaxonomyStatus.textContent =
        displayText(prediction.taxonomy_status);

    xaiTaxonomySupport.textContent =
        formatScore(prediction.taxonomy_support);

    xaiTaxonomyMargin.textContent =
        formatScore(prediction.taxonomy_margin);

    xaiDuplicateScore.textContent =
        formatScore(prediction.duplicate_score);

    xaiAlternativeCategory.textContent =
        displayText(prediction.best_alternative_category);

    xaiAlternativeSupport.textContent =
        "Alternative support: " +
        formatScore(prediction.best_alternative_support);

    xaiReviewReasons.innerHTML = "";

    const reasons = Array.isArray(prediction.review_reasons)
        ? prediction.review_reasons
        : [];

    const reasonDescriptions = {
        "Low Confidence":
            "Prediction confidence is below the configured 70% threshold.",
        "Image-Text Mismatch":
            "The product image and title may describe different things.",
        "Taxonomy Uncertainty":
            "Taxonomy validation requires review. Check confidence, support margin, and image-text consistency.",
        "Possible Duplicate":
            "The decision engine's duplicate similarity score is above 0.90."
    };

    if (reasons.length > 0) {
        reasons.forEach(reason => {
            const item = document.createElement("li");
            const heading = document.createElement("strong");
            heading.textContent = reason;

            item.appendChild(heading);

            const description = document.createElement("div");
            description.className = "small";
            description.textContent =
                reasonDescriptions[reason] ??
                "This signal requires further inspection.";

            item.appendChild(description);
            xaiReviewReasons.appendChild(item);
        });
    } else {
        const item = document.createElement("li");

        item.textContent =
            decision === "PASS"
                ? "No configured review triggers were detected. This does not guarantee that the prediction is correct."
                : "The backend did not return review reasons for a recognized decision.";

        xaiReviewReasons.appendChild(item);
    }

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


// Dashboard tab navigation: only controls which existing panel is visible.
(function initializeAnalysisTabs() {
    const tabs = document.querySelectorAll("[data-analysis-tab]");
    const panels = {
        overview: document.getElementById("overviewPanel"),
        evidence: document.getElementById("evidencePanel"),
        similar: document.getElementById("similarPanel"),
        details: document.getElementById("detailsPanel")
    };

    if (!tabs.length || Object.values(panels).some(panel => !panel)) {
        console.warn("Dashboard tabs were not initialized: a tab or panel is missing.");
        return;
    }

    function activateTab(name) {
        tabs.forEach(tab => {
            const selected = tab.dataset.analysisTab === name;
            tab.classList.toggle("active", selected);
            tab.setAttribute("aria-selected", String(selected));
            tab.setAttribute("tabindex", selected ? "0" : "-1");
        });

        Object.entries(panels).forEach(([panelName, panel]) => {
            panel.hidden = panelName !== name;
        });
    }

    tabs.forEach((tab, index) => {
        tab.addEventListener("click", () => {
            activateTab(tab.dataset.analysisTab);
        });

        tab.addEventListener("keydown", event => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
                return;
            }

            event.preventDefault();

            let nextIndex = index;
            if (event.key === "ArrowRight") {
                nextIndex = (index + 1) % tabs.length;
            } else if (event.key === "ArrowLeft") {
                nextIndex = (index - 1 + tabs.length) % tabs.length;
            } else if (event.key === "Home") {
                nextIndex = 0;
            } else if (event.key === "End") {
                nextIndex = tabs.length - 1;
            }

            tabs[nextIndex].focus();
            activateTab(tabs[nextIndex].dataset.analysisTab);
        });
    });

    activateTab("overview");
})();

