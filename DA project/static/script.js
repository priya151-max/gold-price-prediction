let chart; // Declare the chart variable globally

// Load today's key data
async function loadInitialData() {
    try {
        const response = await fetch('/data');
        const data = await response.json();

        document.getElementById('today-date').textContent = formatDate(new Date(data.date));
        document.getElementById('today-price').textContent = parseFloat(data.price).toFixed(2);
        document.getElementById('high-price').textContent = '₹' + parseFloat(data.high).toFixed(2);
        document.getElementById('low-price').textContent = '₹' + parseFloat(data.low).toFixed(2);
        document.getElementById('change-rs').textContent = '₹' + parseFloat(data.change_rs).toFixed(2);
        document.getElementById('change-percent').textContent = parseFloat(data.change_percent).toFixed(2) + '%';

        // Update the gauge based on today's change percentage
        updateGauge(data.change_rs, data.price);
    } catch (error) {
        console.error("Error fetching initial data:", error);
    }
}

// Function to update the gauge based on today's price change
function updateGauge(changeRs, currentPrice) {
    const openPrice = currentPrice - changeRs; // Assuming open price is the price minus change in Rs.
    let pointerPosition;
    let recommendation;

    const changePercent = (changeRs / openPrice) * 100;

    if (changePercent > 3) {
        recommendation = 'strong-buy';
        pointerPosition = '80%'; // Strong Buy
    } else if (changePercent > 1) {
        recommendation = 'buy';
        pointerPosition = '60%'; // Buy
    } else if (changePercent === 0) {
        recommendation = 'neutral';
        pointerPosition = '40%'; // Neutral
    } else if (changePercent < 0 && changePercent >= -3) {
        recommendation = 'sell';
        pointerPosition = '20%'; // Sell
    } else {
        recommendation = 'strong-sell';
        pointerPosition = '0%'; // Strong Sell
    }

    document.getElementById('gauge-pointer').style.left = pointerPosition;
    console.log(`Today's recommendation based on changes: ${recommendation}`);
}

// Animate metric rows on scroll
window.addEventListener('scroll', () => {
    document.querySelectorAll('.metric-row').forEach(row => {
        const rect = row.getBoundingClientRect();
        if (rect.top < window.innerHeight - 100) {
            row.classList.add('visible');
        }
    });
});

// Fetch all gold price data
async function fetchGoldData() {
    const response = await fetch('/gold-data');
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    return await response.json();
}

// Update the chart and today's data
async function updateDashboard(timeframe) {
    try {
        const goldData = await fetchGoldData();
        const today = new Date(goldData.today_date);
        const filteredData = filterDataByTimeframe(goldData.all_data, timeframe, today);

        // Update displayed numeric values
        document.getElementById('today-date').textContent = formatDate(today);
        document.getElementById('today-price').textContent = goldData.today_price.toFixed(2);
        document.getElementById('high-price').textContent = '₹ 9404.47';
        document.getElementById('low-price').textContent = '₹ 9354.35';
        document.getElementById('change-rs').textContent = '₹' + goldData.change_in_rupes.toFixed(2);
        document.getElementById('change-percent').textContent = `${goldData.change_in_percent.toFixed(2)}%`;

        // Prepare chart data
        const labels = filteredData.map(d => new Date(d.Date).toLocaleDateString('en-IN'));
        const prices = filteredData.map(d => parseFloat(d.Price));

        if (chart) {
            chart.data.labels = labels;
            chart.data.datasets[0].data = prices;
            chart.update();
        } else {
            const ctx = document.getElementById('goldChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Gold Price (INR/gram)',
                        data: prices,
                        borderColor: '#d4af37',
                        backgroundColor: 'rgba(255, 215, 0, 0.2)',
                        borderWidth: 2,
                        pointBackgroundColor: '#f2c94c',
                        pointBorderColor: '#fff',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#333',
                                font: { size: 14 }
                            }
                        },
                        tooltip: {
                            backgroundColor: '#fffbea',
                            titleColor: '#d4af37',
                            bodyColor: '#000',
                            borderColor: '#d4af37',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#333' },
                            grid: { color: 'rgba(0, 0, 0, 0.05)' }
                        },
                        y: {
                            ticks: {
                                color: '#333',
                                callback: value => '₹' + value.toLocaleString()
                            },
                            grid: { color: 'rgba(0, 0, 0, 0.05)' }
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error("Error loading dashboard:", error);
        alert("Error fetching gold data.");
    }
}
function updateGauge(changeRs, currentPrice) {
    const openPrice = currentPrice - changeRs;
    const changePercent = (changeRs / openPrice) * 100;

    let rotationDeg;

    if (changePercent <= -3) {
        rotationDeg = -80; // Strong Sell
    } else if (changePercent < -1) {
        rotationDeg = -40; // Sell
    } else if (changePercent >= -1 && changePercent <= 1) {
        rotationDeg = 0; // Neutral
    } else if (changePercent > 1 && changePercent <= 3) {
        rotationDeg = 40; // Buy
    } else if (changePercent > 3) {
        rotationDeg = 80; // Strong Buy
    }

    const pointer = document.getElementById('gauge-fill');
    if (pointer) {
        pointer.style.transform = `rotate(${rotationDeg}deg)`;
    }
}


// Timeframe-based filter
function filterDataByTimeframe(allData, timeframe, today) {
    return allData.filter(d => {
        const date = new Date(d.Date);
        switch (timeframe) {
            case 'week':
                return date >= new Date(today.getFullYear(), today.getMonth(), today.getDate() - 7);
            case 'month':
                return date >= new Date(today.getFullYear(), today.getMonth() - 1, today.getDate());
            case '1y':
                return date >= new Date(today.getFullYear() - 1, today.getMonth(), today.getDate());
            case '10y':
                return date >= new Date(today.getFullYear() - 10, today.getMonth(), today.getDate());
            case '25y':
                return date >= new Date(today.getFullYear() - 25, today.getMonth(), today.getDate());
            default:
                return true;
        }
    });
}

// Date formatter
function formatDate(date) {
    return date.toLocaleDateString('en-IN', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Tab click handler
function changeTab(timeframe) {
    updateDashboard(timeframe);
}

// Load initial state
loadInitialData().then(() => {
    updateDashboard('week');
});