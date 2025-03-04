document.addEventListener("DOMContentLoaded", function () {
    fetch("/data")
        .then(response => response.json())
        .then(data => {
            if (!data || data.length === 0) {
                console.error("No data received.");
                return;
            }

            // Convert and filter out invalid data
            const filteredData = data
                .map(item => ({
                    question: item.market_question,
                    chance: parseFloat(item.market_chance)
                }))
                .filter(item => !isNaN(item.chance)); // Remove invalid numbers

            if (filteredData.length === 0) {
                console.error("No valid market chances found.");
                return;
            }

            // Sort by market chance (descending) and limit number of displayed items
            const topData = filteredData
                .sort((a, b) => b.chance - a.chance)
                .slice(0, 20); // Display only top 20

            const labels = topData.map(item => item.question);
            const marketChances = topData.map(item => item.chance);

            // Debugging log
            console.log("Market Chances Data:", marketChances);

            // Destroy existing chart if it exists
            if (window.marketChartInstance) {
                window.marketChartInstance.destroy();
            }

            const ctx = document.getElementById("marketChart").getContext("2d");
            window.marketChartInstance = new Chart(ctx, {
                type: "bar",
                data: {
                    labels: labels,
                    datasets: [{
                        label: "Market Chance",
                        data: marketChances,
                        backgroundColor: "rgba(75, 192, 192, 0.7)",
                        borderColor: "rgba(75, 192, 192, 1)",
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            ticks: {
                                maxRotation: 45, // Rotate long labels
                                minRotation: 0,
                                autoSkip: true,
                                maxTicksLimit: 10
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: "Market Chance (%)"
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    return `${context.dataset.label}: ${context.raw.toFixed(2)}%`;
                                }
                            }
                        }
                    }
                }
            });
        })
        .catch(error => console.error("Error fetching data:", error));
});
