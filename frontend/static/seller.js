// =====================================================
// Seller Portal
// AI Product Analysis
// =====================================================

const FASTAPI_URL = "http://127.0.0.1:8001";

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

    invalidateSellerAnalysis();

    const image = imageInput.files[0];

    if (!image)
        return;

    imagePreview.src =
        URL.createObjectURL(image);

    imagePreview.style.display = "block";

});

document.getElementById("sellerProductTitle").addEventListener("input", () => {
    invalidateSellerAnalysis();
});

function invalidateSellerAnalysis() {
    latestAIResult = null;
    latestImageName = null;

    const resultSection = document.getElementById("sellerAIResult");
    if (resultSection) {
        resultSection.style.display = "none";
    }
}


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

        // Queue the item only after the seller saves it.
        formData.append("enqueue_review", "false");


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

        const escapeHtml = value =>
        String(value ?? "").replace(/[&<>"']/g, char => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        })[char]);

    const confidence =
        Number.isFinite(Number(result.confidence))
            ? (Number(result.confidence) * 100).toFixed(1)
            : "N/A";

    const similarity =
        Number.isFinite(Number(result.image_title_similarity))
            ? Number(result.image_title_similarity).toFixed(4)
            : "N/A";

    const duplicateScore =
        Number.isFinite(Number(result.duplicate_score))
            ? Number(result.duplicate_score).toFixed(4)
            : "N/A";

    const taxonomySupport =
        Number.isFinite(Number(result.taxonomy_support))
            ? Number(result.taxonomy_support).toFixed(4)
            : "N/A";

    const taxonomyMargin =
        Number.isFinite(Number(result.taxonomy_margin))
            ? Number(result.taxonomy_margin).toFixed(4)
            : "N/A";

    const decision = String(result.decision ?? "UNKNOWN").toUpperCase();
    const reviewRequired = decision === "REVIEW";

    const reviewReasons = Array.isArray(result.review_reasons)
        ? result.review_reasons
        : [];

    const reasonDescriptions = {
        "Low Confidence":
            `Prediction confidence is ${confidence}% (review threshold: below 70%).`,
        "Image-Text Mismatch":
            "The image and product title may describe different things.",
        "Taxonomy Uncertainty":
            `Taxonomy validation requires review. Support margin: ${taxonomyMargin} (configured threshold: 0.10).`,
        "Possible Duplicate":
            `Duplicate similarity score: ${duplicateScore} (review threshold: above 0.90).`
    };

    const reasonsHtml = reviewReasons.length
        ? `<ul class="mb-0 mt-2">${
            reviewReasons.map(reason => `
                <li class="mb-2">
                    <strong>${escapeHtml(reason)}</strong>
                    <div class="small">${escapeHtml(
                        reasonDescriptions[reason] ??
                        "This signal requires further inspection."
                    )}</div>
                </li>
            `).join("")
        }</ul>`
        : `<p class="mb-0 mt-2">No review triggers were returned by the decision engine.</p>`;

    const alternativeCategory = result.best_alternative_category
        ? escapeHtml(result.best_alternative_category.replaceAll("_", " "))
        : "None identified";

    const alternativeSupport =
        Number.isFinite(Number(result.best_alternative_support))
            ? Number(result.best_alternative_support).toFixed(4)
            : "N/A";

    resultSection.innerHTML = `
        <div class="card-body p-4">
            <span class="badge ${reviewRequired ? "bg-warning text-dark" : "bg-success"} mb-3">
                AI Analysis Complete
            </span>

            <h4 class="fw-bold mb-4">AI Product Assessment</h4>

            <div class="row g-3">
                <div class="col-md-4">
                    <div class="border rounded p-3 h-100">
                        <small class="text-muted">Predicted Category</small>
                        <h5 class="fw-bold mt-2 mb-0">
                            ${escapeHtml(String(result.category ?? "Unknown").replaceAll("_", " "))}
                        </h5>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="border rounded p-3 h-100">
                        <small class="text-muted">Prediction Confidence</small>
                        <h5 class="fw-bold mt-2 mb-0">${confidence}%</h5>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="border rounded p-3 h-100">
                        <small class="text-muted">Image–Text Similarity</small>
                        <h5 class="fw-bold mt-2 mb-0">${similarity}</h5>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="border rounded p-3 h-100">
                        <small class="text-muted">Duplicate Similarity Score</small>
                        <h5 class="fw-bold mt-2 mb-0">${duplicateScore}</h5>
                        <small class="text-muted">Review threshold: above 0.90</small>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="border rounded p-3 h-100">
                        <small class="text-muted">Taxonomy Status</small>
                        <h5 class="fw-bold mt-2 mb-0">${escapeHtml(result.taxonomy_status ?? "N/A")}</h5>
                        <small class="text-muted">Support: ${taxonomySupport} · Margin: ${taxonomyMargin}</small>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="border rounded p-3 h-100">
                        <small class="text-muted">Best Alternative Category</small>
                        <h6 class="fw-bold mt-2 mb-1">${alternativeCategory}</h6>
                        <small class="text-muted">Alternative support: ${alternativeSupport}</small>
                    </div>
                </div>
            </div>

            <div class="alert ${reviewRequired ? "alert-warning" : decision === "PASS" ? "alert-success" : "alert-secondary"} mt-4 mb-0">
                <strong>
                    ${reviewRequired
                        ? "Human Review Required"
                        : decision === "PASS"
                            ? "Passed Automated Review Checks"
                            : "Decision Status Unavailable"}
                </strong>

                <div class="small mt-1">
                    ${reviewRequired
                        ? "One or more decision rules require human inspection. Review the evidence below."
                        : decision === "PASS"
                            ? "No configured review triggers were detected. This does not guarantee that the prediction is correct."
                            : "The backend did not return a recognized decision. Verify the result before proceeding."}
                </div>

                <hr>

                <strong>Decision Explanation</strong>
                ${reasonsHtml}
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
                        image_title_similarity: latestAIResult.image_title_similarity,
                        mismatch: latestAIResult.mismatch,
                        duplicate_score: latestAIResult.duplicate_score,
                        taxonomy_status: latestAIResult.taxonomy_status,
                        taxonomy_margin: latestAIResult.taxonomy_margin,
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

        invalidateSellerAnalysis();

        try {
            await loadSellerProducts();
        } catch (refreshError) {
            console.error(
                "Product saved, but catalog refresh failed:",
                refreshError
            );
        }

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
        
        document.getElementById("myProductsCount").textContent =
            data.total_items;

        document.getElementById("pendingReviewCount").textContent =
            data.items.filter(product => product.workflow_status === "Pending Review").length;

        document.getElementById("approvedProductsCount").textContent =
            data.items.filter(product => product.workflow_status === "Approved").length;

        if (!data.items || data.items.length === 0) {
            empty.style.display = "block";
            return;
        }

        wrapper.style.display = "block";

        const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        })[char]);

        tableBody.innerHTML = data.items.map(product => {
            const reviewStatus = product.workflow_status || "Not Reviewed";
            const details = [];

            if (product.corrected_category) {
                details.push(`Corrected category: ${escapeHtml(product.corrected_category)}`);
            }

            if (product.review_feedback) {
                details.push(escapeHtml(product.review_feedback));
            }

            return `
                <tr>
                    <td title="${escapeHtml(product.title || "")}">
                        ${escapeHtml(product.title || "-")}
                    </td>
                    <td>${escapeHtml(product.brand || "-")}</td>
                    <td>${escapeHtml(product.category || "-")}</td>
                    <td>${product.confidence != null
                        ? `${(product.confidence * 100).toFixed(1)}%`
                        : "-"}</td>
                    <td>${escapeHtml(reviewStatus)}</td>
                    <td>${details.length ? details.map(detail => `<div>${detail}</div>`).join("") : "-"}</td>
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

const bulkAnalyzeBtn = document.getElementById("bulkAnalyzeBtn");
const bulkCatalogCsv = document.getElementById("bulkCatalogCsv");
const bulkCatalogImages = document.getElementById("bulkCatalogImages");
const bulkUploadStatus = document.getElementById("bulkUploadStatus");

if (bulkAnalyzeBtn) {
    bulkAnalyzeBtn.addEventListener("click", async () => {

        if (!bulkCatalogCsv.files.length) {
            alert("Please select a catalog CSV file.");
            return;
        }

        if (!bulkCatalogImages.files.length) {
            alert("Please select product images.");
            return;
        }

        bulkUploadStatus.style.display = "block";
        bulkUploadStatus.textContent =
            "Uploading catalog and images for AI processing...";

        bulkAnalyzeBtn.disabled = true;

        const formData = new FormData();

        formData.append("catalog", bulkCatalogCsv.files[0]);

        for (const image of bulkCatalogImages.files) {
            formData.append("images", image);
        }

        try {

            const response = await fetch(
                `${FASTAPI_URL}/seller/bulk-products`,
                {
                    method: "POST",
                    body: formData
                }
            );

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(
                    result.message || "Bulk catalog processing failed."
                );
            }

            const successfulRows =
                result.successful_rows ?? result.processed_products ?? 0;
            const failedRows = result.failed_rows ?? 0;
            const skippedRows = result.skipped_rows ?? 0;

            bulkUploadStatus.textContent =
                `Batch completed: ${successfulRows} successful, ${failedRows} failed, ${skippedRows} skipped out of ${result.total_products} products.`;

            if (failedRows > 0 || skippedRows > 0) {
                alert(
                    `Bulk upload completed with some issues.\n` +
                    `Successful: ${successfulRows}\n` +
                    `Failed: ${failedRows}\n` +
                    `Skipped: ${skippedRows}`
                );
            } else {
                alert(
                    `Bulk catalog processing completed successfully!\n` +
                    `${successfulRows} products processed.`
                );
            }

            loadSellerProducts();

        } catch (error) {

            console.error("Bulk catalog error:", error);

            bulkUploadStatus.textContent =
                "Bulk catalog processing failed.";

            alert(error.message);

        } finally {

            bulkAnalyzeBtn.disabled = false;

        }
    });
}
