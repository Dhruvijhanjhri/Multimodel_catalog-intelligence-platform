const API = "http://127.0.0.1:8000";

async function loadDashboard() {

    //-----------------------------------
    // Metrics
    //-----------------------------------

    const metrics = await fetch(`${API}/metrics`);
    const metricsData = await metrics.json();

    document.getElementById("totalProducts").innerText =
        metricsData.dataset.total_products;

    document.getElementById("totalEmbeddings").innerText =
        metricsData.embeddings.total_embeddings;

    document.getElementById("accuracy").innerText =
        (metricsData.model.test_accuracy * 100).toFixed(2) + "%";

    //-----------------------------------
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
                <td>${item.reason}</td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
                <td>${(item.duplicate_score * 100).toFixed(1)}%</td>
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
    //--------------------------------------------------
    // Approve
    //--------------------------------------------------

    document.addEventListener("click", async (e) => {

        if (!e.target.classList.contains("approveBtn"))
            return;

        const id = e.target.dataset.id;

        e.target.disabled = true;
        e.target.innerHTML = "Approving...";

        const response = await fetch(
            `${API}/review-queue/${id}/approve`,
            {
                method: "PUT"
            }
        );

        const data = await response.json();

        alert(data.message);

        loadDashboard();

    });

    //--------------------------------------------------
    // Reject
    //--------------------------------------------------

    document.addEventListener("click", async (e) => {

        if (!e.target.classList.contains("rejectBtn"))
            return;

        const id = e.target.dataset.id;

        const rejectModal = new bootstrap.Modal(
            document.getElementById("rejectReviewModal")
        );

        const categorySelect =
            document.getElementById("correctedCategory");

        const feedbackInput =
            document.getElementById("reviewerFeedback");

        const confirmRejectBtn =
            document.getElementById("confirmRejectBtn");

        categorySelect.value = "";
        feedbackInput.value = "";

        confirmRejectBtn.onclick = async () => {

            const selectedCategory = categorySelect.value;
            const feedback = feedbackInput.value;

            if (!selectedCategory) {
                alert("Please select a corrected category.");
                return;
            }

            confirmRejectBtn.disabled = true;
            confirmRejectBtn.innerHTML = "Rejecting...";

            const response = await fetch(
                `${API}/review-queue/${id}/reject?corrected_category=${encodeURIComponent(selectedCategory)}&feedback=${encodeURIComponent(feedback)}`,
                {
                    method: "PUT"
                }
            );

            const data = await response.json();

            alert(data.message);

            rejectModal.hide();

            confirmRejectBtn.disabled = false;
            confirmRejectBtn.innerHTML = "Confirm Reject";

            loadDashboard();

        };

        rejectModal.show();

    });

    //--------------------------------------------------
    // Delete
    //--------------------------------------------------

    document.addEventListener("click", async (e) => {

        if (!e.target.classList.contains("deleteBtn"))
            return;

        if (!confirm("Delete this review item?"))
            return;

        const id = e.target.dataset.id;

        await fetch(
            `${API}/review-queue/${id}`,
            {
                method: "DELETE"
            }
        );

        alert("Review Deleted Successfully");

        loadDashboard();

    });
}

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