const API = "http://127.0.0.1:8001";

async function loadDashboard() {

    // Core operational metrics
    const metricsResponse = await fetch(`${API}/metrics`);

    if (!metricsResponse.ok) {
        throw new Error(`Metrics API returned HTTP ${metricsResponse.status}`);
    }

    const metricsData = await metricsResponse.json();

    document.getElementById("totalProducts").innerText =
        metricsData.dataset.total_products.toLocaleString();

    const monitoring = metricsData.prediction_monitoring || {};

    document.getElementById("livePredictions").innerText =
        (monitoring.total_predictions ?? 0).toLocaleString();

    const ingestion = metricsData.ingestion_monitoring || {};

    document.getElementById("failedIngestionEvents").innerText =
        (ingestion.failed_events ?? 0).toLocaleString();

    const queueMonitoring = metricsData.review_queue_monitoring || {};

    document.getElementById("pendingReviews").innerText =
        (queueMonitoring.pending_reviews ?? 0).toLocaleString();

    // Ingestion History
    //-----------------------------------

    const ingestionEvents =
        await fetch(`${API}/ingestion-events`);

    const ingestionData =
        await ingestionEvents.json();

    const ingestionTable =
        document.getElementById("ingestionHistoryTable");

    ingestionTable.innerHTML =
        ingestionData.events.map(event => {

            const payload = event.payload || {};

            return `
                <tr>
                    <td>${payload.batch_id || event.event_id}</td>
                    <td>${event.event_type || "--"}</td>
                    <td>${event.status || "--"}</td>
                    <td>${payload.total_products ?? "--"}</td>
                    <td>${payload.successful_rows ?? "--"}</td>
                    <td>${payload.failed_rows ?? "--"}</td>
                    <td>${payload.skipped_rows ?? "--"}</td>
                    <td>${event.created_at ? new Date(event.created_at).toLocaleString() : "--"}</td>
                </tr>
            `;
        }).join("");

    // Review Queue
    //-----------------------------------

    const review = await fetch(`${API}/review-queue`);
    const reviewData = await review.json();

    document.getElementById("reviewCount").innerText =
        reviewData.total_items;

    const table = document.getElementById("reviewTable");

    const reviewItems = reviewData.items;

    const rowsPerPage = 25;

    let currentPage = 1;

    function renderReviewTable(page) {

        const start = (page - 1) * rowsPerPage;
        const end = start + rowsPerPage;

        const pageItems = reviewItems.slice(start, end);

        const rows = pageItems.map(item => `
            <tr>
                <td>
                    <div class="d-flex align-items-center gap-2">
                        <a href="http://127.0.0.1:5000/image/${item.image_name}" target="_blank">
                            <img
                                src="http://127.0.0.1:5000/image/${item.image_name}"
                                alt="Product image"
                                style="width:50px;height:50px;object-fit:cover;border-radius:6px;cursor:pointer;"
                            >
                        </a>
                        <span>${item.title}</span>
                    </div>
                </td>
                <td>${item.category}</td>
                <td>
                    <strong>${item.reason || "Manual Review"}</strong>
                    <div class="small text-muted">
                        Mismatch: ${
                            Number.isFinite(Number(item.mismatch_score))
                                ? (Number(item.mismatch_score) * 100).toFixed(1) + "%"
                                : "--"
                        }
                    </div>
                </td>
                <td>${(Number(item.confidence ?? 0) * 100).toFixed(1)}%</td>
                <td>${(Number(item.duplicate_score ?? 0) * 100).toFixed(1)}%</td>
                <td>${item.created_at}</td>
                <td>
                    <button class="btn btn-success btn-sm approveBtn"
                        data-id="${item.id}">
                        Approve
                    </button>

                    <button class="btn btn-warning btn-sm rejectBtn"
                        data-id="${item.id}">
                        Reject
                    </button>

                    <button class="btn btn-danger btn-sm deleteBtn"
                        data-id="${item.id}">
                        Delete
                    </button>
                </td>
            </tr>
        `).join("");

        table.innerHTML = rows;

        renderPagination();
    }

    function renderPagination() {

        const totalPages =
            Math.ceil(reviewItems.length / rowsPerPage);

        let pagination =
            document.getElementById("reviewPagination");

        if (!pagination) {

            pagination = document.createElement("div");

            pagination.id = "reviewPagination";

            pagination.className =
                "d-flex justify-content-center mt-3";

            table.parentElement.appendChild(pagination);
        }

        pagination.innerHTML = "";

        for (let page = 1; page <= totalPages; page++) {

            const button =
                document.createElement("button");

            button.className =
                `btn btn-sm me-1 ${
                    page === currentPage
                        ? "btn-primary"
                        : "btn-outline-primary"
                }`;

            button.innerText = page;

            button.onclick = () => {

                currentPage = page;

                renderReviewTable(currentPage);

            };

            pagination.appendChild(button);
        }
    }

    renderReviewTable(currentPage);
}

// Review actions: register these handlers only once.
async function readApiResponse(response) {
    let data = {};

    try {
        data = await response.json();
    } catch (_) {
        // The API may return an empty or non-JSON response.
    }

    if (!response.ok || data.success === false) {
        throw new Error(
            data.detail || data.message || `Request failed (HTTP ${response.status})`
        );
    }

    return data;
}

document.addEventListener("click", async (e) => {
    const button = e.target.closest(".approveBtn");
    if (!button || button.disabled) return;

    const id = button.dataset.id;
    button.disabled = true;
    button.innerHTML = "Approving...";

    try {
        const response = await fetch(`${API}/review-queue/${id}/approve`, {
            method: "PUT"
        });
        const data = await readApiResponse(response);

        alert(data.message || "Review approved successfully.");
        await loadDashboard();
    } catch (error) {
        alert(`Could not approve review: ${error.message}`);
    } finally {
        button.disabled = false;
        button.innerHTML = "Approve";
    }
});

document.addEventListener("click", async (e) => {
    const button = e.target.closest(".rejectBtn");
    if (!button) return;

    const id = button.dataset.id;
    const rejectModal = new bootstrap.Modal(
        document.getElementById("rejectReviewModal")
    );

    const categorySelect = document.getElementById("correctedCategory");
    const feedbackInput = document.getElementById("reviewerFeedback");
    const confirmRejectBtn = document.getElementById("confirmRejectBtn");

    categorySelect.value = "";
    feedbackInput.value = "";

    confirmRejectBtn.onclick = async () => {
        if (confirmRejectBtn.disabled) return;

        const selectedCategory = categorySelect.value;
        const feedback = feedbackInput.value;

        if (!selectedCategory) {
            alert("Please select a corrected category.");
            return;
        }

        confirmRejectBtn.disabled = true;
        confirmRejectBtn.innerHTML = "Rejecting...";

        try {
            const params = new URLSearchParams({
                corrected_category: selectedCategory,
                feedback: feedback
            });

            const response = await fetch(
                `${API}/review-queue/${id}/reject?${params}`,
                { method: "PUT" }
            );
            const data = await readApiResponse(response);

            alert(data.message || "Review rejected successfully.");
            rejectModal.hide();
            await loadDashboard();
        } catch (error) {
            alert(`Could not reject review: ${error.message}`);
        } finally {
            confirmRejectBtn.disabled = false;
            confirmRejectBtn.innerHTML = "Confirm Reject";
        }
    };

    rejectModal.show();
});

document.addEventListener("click", async (e) => {
    const button = e.target.closest(".deleteBtn");
    if (!button || button.disabled) return;

    if (!confirm("Delete this review item?")) return;

    const id = button.dataset.id;
    button.disabled = true;
    button.innerHTML = "Deleting...";

    try {
        const response = await fetch(`${API}/review-queue/${id}`, {
            method: "DELETE"
        });
        const data = await readApiResponse(response);

        alert(data.message || "Review deleted successfully.");
        await loadDashboard();
    } catch (error) {
        alert(`Could not delete review: ${error.message}`);
    } finally {
        button.disabled = false;
        button.innerHTML = "Delete";
    }
});

loadDashboard();

//----------------------------------------------------
// Dashboard Charts
//----------------------------------------------------

async function loadCharts() {

    const response =
        await fetch(`${API}/dashboard-charts`);

    const data =
        await response.json();

    //------------------------------------------------
    // Category Chart
    //------------------------------------------------

    new Chart(

        document.getElementById("categoryChart"),

        {

            type: "bar",

            data: {

                labels: Object.keys(data.categories),

                datasets: [{

                    label: "Products",

                    data: Object.values(data.categories)

                }]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false

            }

        }

    );

    //------------------------------------------------

    //-----------------------------------
    // Review Queue Chart
    //------------------------------------------------

    const reasonCanvas =
        document.getElementById("reasonChart");

    if (data.reasons.length === 0) {

        reasonCanvas.parentElement.innerHTML = `

            <div class="d-flex justify-content-center align-items-center h-100">

                <h6 class="text-muted">
                    No Review Queue Items
                </h6>

            </div>

        `;

    }

    else {

        new Chart(

            reasonCanvas,

            {

                type: "pie",

                data: {

                    labels: data.reasons.map(x => x.reason),

                    datasets: [{

                        data: data.reasons.map(x => x.total)

                    }]

                },

                options: {

                    responsive: true,

                    maintainAspectRatio: false

                }

            }

        );

    }

}

loadCharts();
