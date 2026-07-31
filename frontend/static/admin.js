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

    table.innerHTML = "";

    reviewData.items.forEach(item => {

        table.innerHTML += `
        <tr>
            <td>${item.title}</td>
            <td>${item.category}</td>
            <td>${item.reason}</td>
            <td>${(item.confidence*100).toFixed(1)}%</td>
            <td>${(item.duplicate_score*100).toFixed(1)}%</td>
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
        `;

    });

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

        e.target.disabled = true;
        e.target.innerHTML = "Rejecting...";

        const response = await fetch(
            `${API}/review-queue/${id}/reject`,
            {
                method: "PUT"
            }
        );

        const data = await response.json();

        alert(data.message);

        loadDashboard();

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