const API = "http://127.0.0.1:8001";

const monitoringCharts = {};

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value ?? "--";
    }
}

function formatNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString() : "--";
}

function formatPercent(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "--";
    }
    return `${(Number(value) * 100).toFixed(2)}%`;
}

async function fetchJSON(endpoint) {
    const response = await fetch(`${API}${endpoint}`);

    if (!response.ok) {
        throw new Error(`${endpoint} returned HTTP ${response.status}`);
    }

    return response.json();
}

function renderChart(id, config) {
    const canvas = document.getElementById(id);

    if (!canvas || typeof Chart === "undefined") {
        return;
    }

    if (monitoringCharts[id]) {
        monitoringCharts[id].destroy();
    }

    monitoringCharts[id] = new Chart(canvas, config);
}

function renderRateChart(id, labels, values) {
    renderChart(id, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Rate",
                data: values,
                backgroundColor: ["#4263eb", "#f59f00"],
                borderRadius: 8,
                maxBarThickness: 58
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: value => `${value}%`
                    },
                    grid: {
                        color: "#edf0f5"
                    }
                },
                y: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: context => `${context.parsed.x}%`
                    }
                }
            }
        }
    });
}

async function loadMonitoringDashboard() {
    try {
        const metricsData = await fetchJSON("/metrics");

        const model = metricsData.model || {};
        const prediction = metricsData.prediction_monitoring || {};
        const feedback = metricsData.feedback_monitoring || {};
        const ingestion = metricsData.ingestion_monitoring || {};
        const queue = metricsData.review_queue_monitoring || {};

        setText("accuracy", formatPercent(model.test_accuracy));
        setText("averageConfidence", formatPercent(prediction.average_confidence));
        setText("lowConfidenceRate", formatPercent(prediction.low_confidence_rate));

        const modelVersions = Object.keys(prediction.predictions_by_model_version || {});
        setText("activeModel", modelVersions.length ? modelVersions[0] : "N/A");

        setText("humanReviews", formatNumber(feedback.total_reviews));
        setText("approvalRate", formatPercent(feedback.approval_rate));
        setText("correctionRate", formatPercent(feedback.correction_rate));

        setText("totalIngestionEvents", formatNumber(ingestion.total_events));
        setText("processedIngestionEvents", formatNumber(ingestion.processed_events));
        setText("failedIngestionEvents", formatNumber(ingestion.failed_events));
        setText("ingestionFailureRate", formatPercent(ingestion.failure_rate));
        setText("queueProcessingRate", formatPercent(queue.processing_rate));

        const qualityData = await fetchJSON("/human-feedback-quality");

        setText(
            "validatedAccuracy",
            qualityData.validated_accuracy === null ||
            qualityData.validated_accuracy === undefined
                ? "--"
                : formatPercent(qualityData.validated_accuracy)
        );
        setText("validatedPredictions", formatNumber(qualityData.validated_predictions));
        setText(
            "minimumValidatedSamples",
            formatNumber(qualityData.minimum_validated_samples)
        );

        setText(
            "validatedAccuracyStatus",
            qualityData.status === "READY"
                ? "Validation sample threshold met"
                : `Insufficient data · ${formatNumber(qualityData.validated_predictions)} of ${formatNumber(qualityData.minimum_validated_samples)} samples`
        );
        setText("humanFeedbackQualityStatus", qualityData.status || "UNKNOWN");

        const driftData = await fetchJSON("/drift-monitoring");

        setText("overallDriftStatus", driftData.overall_status || "UNKNOWN");
        setText(
            "categoryDriftStatus",
            driftData.category_distribution?.status || "UNKNOWN"
        );
        setText(
            "confidenceDriftStatus",
            driftData.confidence_distribution?.status || "UNKNOWN"
        );

        renderRateChart(
            "feedbackChart",
            ["Approval rate", "Correction rate"],
            [
                Number(feedback.approval_rate || 0) * 100,
                Number(feedback.correction_rate || 0) * 100
            ]
        );

        renderRateChart(
            "confidenceChart",
            ["Average confidence", "Low-confidence rate"],
            [
                Number(prediction.average_confidence || 0) * 100,
                Number(prediction.low_confidence_rate || 0) * 100
            ]
        );

        const totalEvents = Number(ingestion.total_events || 0);
        const processedEvents = Number(ingestion.processed_events || 0);
        const failedEvents = Number(ingestion.failed_events || 0);
        const otherEvents = Math.max(
            0,
            totalEvents - processedEvents - failedEvents
        );

        renderChart("ingestionChart", {
            type: "doughnut",
            data: {
                labels: ["Processed", "Failed", "Other / unclassified"],
                datasets: [{
                    data: [processedEvents, failedEvents, otherEvents],
                    backgroundColor: ["#2f9e44", "#e03131", "#adb5bd"],
                    borderWidth: 0,
                    hoverOffset: 7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "66%",
                plugins: {
                    legend: {
                        position: "bottom"
                    }
                }
            }
        });

    } catch (error) {
        console.error("Unable to load monitoring dashboard:", error);

        const page = document.querySelector(".monitoring-page");
        if (page && !document.getElementById("monitoringLoadError")) {
            const notice = document.createElement("div");
            notice.id = "monitoringLoadError";
            notice.className = "alert alert-warning";
            notice.setAttribute("role", "alert");
            notice.textContent =
                "Some monitoring data could not be loaded. Check that the backend API is running and refresh this page.";
            page.prepend(notice);
        }
    }
}

loadMonitoringDashboard();
