// =====================================================
// Seller Portal
// AI Product Analysis
// =====================================================

const FASTAPI_URL = "http://127.0.0.1:8000";

const imageInput =
    document.getElementById("sellerProductImage");

const imagePreview =
    document.getElementById("sellerImagePreview");

const analyzeBtn =
    document.getElementById("sellerAnalyzeBtn");

let latestAIResult = null;
let latestImageName = null;


// =====================================================
// Image Preview
// =====================================================

imageInput.addEventListener("change", () => {

    const image = imageInput.files[0];

    if (!image)
        return;

    imagePreview.src =
        URL.createObjectURL(image);

    imagePreview.style.display = "block";

});


// =====================================================
// AI Analysis
// =====================================================

analyzeBtn.addEventListener("click", async () => {

    const image =
        imageInput.files[0];

    const title =
        document.getElementById("sellerProductTitle").value.trim();


    if (!image) {

        alert("Please upload a product image.");

        return;

    }


    if (!title) {

        alert("Please enter a product title.");

        return;

    }


    analyzeBtn.disabled = true;

    analyzeBtn.innerHTML =
        "Analyzing...";


    try {

        const formData =
            new FormData();

        formData.append(
            "image",
            image
        );

        formData.append(
            "title",
            title
        );


        const response =
            await fetch(
                `${FASTAPI_URL}/predict`,
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                "AI prediction failed"
            );

        }


        const result =
            await response.json();
        
        latestAIResult = result;
        latestImageName = result.image_name;

        console.log(
            "Seller AI Result:",
            result
        );


        displayAIResult(result);

    }

    catch (error) {

        console.error(error);

        alert(
            "Unable to analyze the product. Please try again."
        );

    }

    finally {

        analyzeBtn.disabled = false;

        analyzeBtn.innerHTML =
            "Analyze Product with AI";

    }

});


// =====================================================
// Display AI Result
// =====================================================

function displayAIResult(result) {

    let resultSection =
        document.getElementById(
            "sellerAIResult"
        );


    if (!resultSection) {

        resultSection =
            document.createElement("div");

        resultSection.id =
            "sellerAIResult";

        resultSection.className =
            "card border-0 shadow-sm mt-4";

        document
            .getElementById("addProduct")
            .appendChild(resultSection);

    }


    const confidence =
        (result.confidence * 100)
            .toFixed(1);


    const similarity =
        result.image_title_similarity
            .toFixed(4);


    const reviewRequired =
        result.mismatch ||
        result.confidence < 0.70;


    resultSection.innerHTML = `

        <div class="card-body p-4">

            <span class="badge bg-success mb-3">
                AI Analysis Complete
            </span>

            <h4 class="fw-bold mb-4">
                AI Product Assessment
            </h4>

            <div class="row g-4">

                <div class="col-md-4">

                    <div class="border rounded p-3">

                        <small class="text-muted">
                            Predicted Category
                        </small>

                        <h5 class="fw-bold mt-2 mb-0">
                            ${result.category.replaceAll("_", " ")}
                        </h5>

                    </div>

                </div>


                <div class="col-md-4">

                    <div class="border rounded p-3">

                        <small class="text-muted">
                            Prediction Confidence
                        </small>

                        <h5 class="fw-bold mt-2 mb-0">
                            ${confidence}%
                        </h5>

                    </div>

                </div>


                <div class="col-md-4">

                    <div class="border rounded p-3">

                        <small class="text-muted">
                            Image–Text Similarity
                        </small>

                        <h5 class="fw-bold mt-2 mb-0">
                            ${similarity}
                        </h5>

                    </div>

                </div>

            </div>


            <div class="alert ${
                reviewRequired
                    ? "alert-warning"
                    : "alert-success"
            } mt-4 mb-0">

                <strong>
                    ${reviewRequired
                        ? "Manual Review Required"
                        : "Ready for Seller Review"
                    }
                </strong>

                <div class="small mt-1">

                    ${
                        result.mismatch
                            ? "The product image and title may not be consistent."
                            : "No image-title mismatch was detected."
                    }

                </div>

            </div>

        </div>

        <div class="mt-4">
            <button
                type="button"
                id="submitSellerProductBtn"
                class="btn btn-success">
                Submit Product to Catalog
            </button>
        </div>

    `;

}

document.addEventListener("click", async (e) => {

    if (e.target.id !== "submitSellerProductBtn")
        return;

    if (!latestAIResult || !latestImageName) {
        alert("Please analyze the product first.");
        return;
    }

    const title =
        document.getElementById("sellerProductTitle").value.trim();

    const brand =
        document.getElementById("sellerProductBrand").value.trim();

    const description =
        document.getElementById("sellerProductDescription").value.trim();

    e.target.disabled = true;
    e.target.innerHTML = "Submitting...";

    try {

        const response =
            await fetch(
                `${FASTAPI_URL}/seller/products`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        title: title,
                        brand: brand || null,
                        description: description || null,
                        image_name: latestImageName,
                        category: latestAIResult.category,
                        confidence: latestAIResult.confidence,
                        model_version: "multimodal_classifier_v1"
                    })
                }
            );

        const data =
            await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Product submission failed."
            );
        }

        alert(data.message);

        e.target.innerHTML = "Product Submitted";

    }

    catch (error) {

        console.error(error);

        alert(
            "Unable to submit the product. Please try again."
        );

        e.target.disabled = false;
        e.target.innerHTML = "Submit Product to Catalog";

    }

});

async function loadSellerProducts() {

    const loading =
        document.getElementById("sellerProductsLoading");

    const empty =
        document.getElementById("sellerProductsEmpty");

    const wrapper =
        document.getElementById("sellerProductsTableWrapper");

    const tableBody =
        document.getElementById("sellerProductsTableBody");

    try {

        const response =
            await fetch(`${FASTAPI_URL}/seller/products`);

        const data =
            await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Unable to load seller products."
            );
        }

        loading.style.display = "none";

        if (!data.items || data.items.length === 0) {
            empty.style.display = "block";
            return;
        }

        wrapper.style.display = "block";

        tableBody.innerHTML = data.items.map(product => {

            const reviewStatus =
                product.review_decision || "Not Reviewed";

            return `
                <tr>
                    <td title="${product.title || ""}">
                        ${product.title || "-"}
                    </td>

                    <td>
                        ${product.brand || "-"}
                    </td>

                    <td>
                        ${product.corrected_category || product.category || "-"}
                    </td>

                    <td>
                        ${product.confidence != null
                            ? `${(product.confidence * 100).toFixed(1)}%`
                            : "-"}
                    </td>

                    <td>
                        ${reviewStatus}
                    </td>
                </tr>
            `;

        }).join("");

    }

    catch (error) {

        console.error(error);

        loading.innerHTML =
            "Unable to load seller products.";

    }
}

document.addEventListener("DOMContentLoaded", () => {
    loadSellerProducts();
});