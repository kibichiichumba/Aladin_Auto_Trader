/**
 * Chart Utils - Helper functions for chart creation and manipulation
 * Used for creating candlestick charts, technical indicators and other chart visualizations
 */

// Register Candlestick Chart Element for Chart.js
if (typeof Chart !== 'undefined') {
    // Register candlestick element
    Chart.defaults.elements.candlestick = {
        borderWidth: 1,
        borderSkipped: '',
        borderColor: 'rgba(0,0,0,0.4)',
        backgroundColor: 'rgba(0,0,0,0.75)',
    };

    // Define a simple base element class if Chart.Element doesn't exist
    const ElementBase = typeof Chart.Element !== 'undefined' ? 
        Chart.Element : 
        class ElementBase {
            constructor() {
                this.x = undefined;
                this.y = undefined;
                this.width = undefined;
                this.height = undefined;
            }
        };

    class CandlestickElement extends ElementBase {
        constructor(cfg) {
            super();
            
            this.x = undefined;
            this.o = undefined;
            this.h = undefined;
            this.l = undefined;
            this.c = undefined;
            
            if (cfg) {
                Object.assign(this, cfg);
            }
        }
        
        draw(ctx) {
            const {x, o, h, l, c} = this;
            
            const borderColor = c >= o ? 'rgb(46, 204, 113)' : 'rgb(231, 76, 60)';
            const backgroundColor = c >= o ? 'rgba(46, 204, 113, 0.5)' : 'rgba(231, 76, 60, 0.5)';
            
            ctx.strokeStyle = borderColor;
            ctx.fillStyle = backgroundColor;
            
            // Draw candle body
            ctx.fillRect(x - 4, o, 8, c - o);
            ctx.strokeRect(x - 4, o, 8, c - o);
            
            // Draw wicks
            ctx.beginPath();
            ctx.moveTo(x, h);
            ctx.lineTo(x, Math.min(o, c));
            ctx.moveTo(x, Math.max(o, c));
            ctx.lineTo(x, l);
            ctx.stroke();
        }
        
        height() {
            return this.h - this.l;
        }
        
        inRange(mouseX, mouseY) {
            const rect = {
                left: this.x - 4,
                right: this.x + 4,
                top: Math.min(this.o, this.c),
                bottom: Math.max(this.o, this.c)
            };
            
            return mouseX >= rect.left && mouseX <= rect.right && 
                   mouseY >= rect.top && mouseY <= rect.bottom;
        }
    }
    
    class CandlestickController extends Chart.DatasetController {
        constructor(chart, datasetIndex) {
            super(chart, datasetIndex);
            
            this.cachedMeta._dataset = {
                type: 'candlestick',
            };
        }
        
        update(mode) {
            const meta = this._cachedMeta;
            const dataset = this.getDataset();
            
            // Create points for each data item
            meta.data = meta.data || [];
            const points = meta.data;
            
            // For responsiveness
            this.updateElements(points, 0, points.length, mode);
        }
        
        updateElements(points, start, count, mode) {
            const dataset = this.getDataset();
            const meta = this._cachedMeta;
            const xScale = this._getIndexScale();
            const yScale = this._getValueScale();
            
            for (let i = start; i < start + count; i++) {
                const parsed = this.getParsed(i);
                const x = xScale.getPixelForValue(i);
                const o = yScale.getPixelForValue(parsed.o);
                const h = yScale.getPixelForValue(parsed.h);
                const l = yScale.getPixelForValue(parsed.l);
                const c = yScale.getPixelForValue(parsed.c);
                
                const properties = {
                    x: x,
                    o: o,
                    h: h,
                    l: l,
                    c: c
                };
                
                // Create or update the element
                if (!points[i]) {
                    points[i] = new CandlestickElement(properties);
                } else {
                    Object.assign(points[i], properties);
                }
            }
        }
        
        _getValueScale() {
            return this._cachedMeta.vScale;
        }
        
        _getIndexScale() {
            return this._cachedMeta.iScale;
        }
    }
    
    CandlestickController.id = 'candlestick';
    CandlestickController.defaults = {};
    
    Chart.register(CandlestickController);
    Chart.register(CandlestickElement);
}

// Helper function to format OHLC data for Chart.js
function formatOHLCData(marketData) {
    return marketData.map((item, index) => {
        return {
            x: index,
            o: item.open,
            h: item.high,
            l: item.low,
            c: item.close
        };
    });
}

// Helper function to create a candlestick chart
function createCandlestickChart(ctx, marketData, options = {}) {
    const data = formatOHLCData(marketData);
    const labels = marketData.map(item => {
        const date = new Date(item.timestamp);
        return date.toLocaleTimeString();
    });
    
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: 'Time'
                }
            },
            y: {
                display: true,
                title: {
                    display: true,
                    text: 'Price'
                }
            }
        },
        ...options
    };
    
    return new Chart(ctx, {
        type: 'candlestick',
        data: {
            labels: labels,
            datasets: [{
                label: 'OHLC',
                data: data
            }]
        },
        options: chartOptions
    });
}

// Helper function to add Moving Average to a chart
function addMovingAverage(chart, data, period, label, color) {
    const closePrices = data.map(item => item.close);
    const maData = calculateSMA(closePrices, period);
    
    chart.data.datasets.push({
        label: label,
        data: maData,
        borderColor: color,
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        type: 'line'
    });
    
    chart.update();
}

// Calculate Simple Moving Average
function calculateSMA(data, period) {
    const result = [];
    
    // Fill with nulls until we have enough data
    for (let i = 0; i < period - 1; i++) {
        result.push(null);
    }
    
    // Calculate SMA for each period
    for (let i = period - 1; i < data.length; i++) {
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += data[i - j];
        }
        result.push(sum / period);
    }
    
    return result;
}

// Calculate Exponential Moving Average
function calculateEMA(data, period) {
    const k = 2 / (period + 1);
    const result = [];
    
    // First EMA is SMA
    let ema = calculateSMA(data.slice(0, period), period)[period - 1];
    
    // Fill with nulls until we have enough data
    for (let i = 0; i < period - 1; i++) {
        result.push(null);
    }
    
    result.push(ema);
    
    // Calculate EMA for remaining data
    for (let i = period; i < data.length; i++) {
        ema = (data[i] - ema) * k + ema;
        result.push(ema);
    }
    
    return result;
}

// Calculate Relative Strength Index (RSI)
function calculateRSI(data, period = 14) {
    const result = [];
    const gains = [];
    const losses = [];
    
    // Calculate price changes
    for (let i = 1; i < data.length; i++) {
        const change = data[i] - data[i - 1];
        gains.push(change > 0 ? change : 0);
        losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    // Fill with nulls until we have enough data
    for (let i = 0; i < period; i++) {
        result.push(null);
    }
    
    // Calculate first average gain and loss
    let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
    
    // Calculate RSI
    for (let i = period; i < data.length; i++) {
        // Update average gain and loss
        avgGain = ((avgGain * (period - 1)) + gains[i - 1]) / period;
        avgLoss = ((avgLoss * (period - 1)) + losses[i - 1]) / period;
        
        // Calculate RS and RSI
        const rs = avgGain / avgLoss;
        const rsi = 100 - (100 / (1 + rs));
        
        result.push(rsi);
    }
    
    return result;
}

// Add RSI indicator to chart
function addRSI(chart, data, period = 14) {
    const closePrices = data.map(item => item.close);
    const rsiData = calculateRSI(closePrices, period);
    
    // Create new canvas for RSI
    const container = chart.canvas.parentNode;
    const rsiCanvas = document.createElement('canvas');
    container.appendChild(rsiCanvas);
    
    const rsiChart = new Chart(rsiCanvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: chart.data.labels,
            datasets: [{
                label: `RSI (${period})`,
                data: rsiData,
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true
                },
                y: {
                    display: true,
                    min: 0,
                    max: 100,
                    grid: {
                        color: ctx => {
                            if (ctx.tick.value === 30 || ctx.tick.value === 70) {
                                return 'rgba(255, 0, 0, 0.3)';
                            }
                            return 'rgba(0, 0, 0, 0.1)';
                        }
                    }
                }
            }
        }
    });
    
    return rsiChart;
}

// Add MACD indicator to chart
function addMACD(chart, data) {
    const closePrices = data.map(item => item.close);
    const ema12 = calculateEMA(closePrices, 12);
    const ema26 = calculateEMA(closePrices, 26);
    
    // Calculate MACD line
    const macdLine = [];
    for (let i = 0; i < closePrices.length; i++) {
        if (ema12[i] === null || ema26[i] === null) {
            macdLine.push(null);
        } else {
            macdLine.push(ema12[i] - ema26[i]);
        }
    }
    
    // Calculate Signal line (9-day EMA of MACD line)
    const signalLine = calculateEMA(macdLine.filter(x => x !== null), 9);
    const fullSignalLine = [];
    
    // Pad signal line with nulls
    for (let i = 0; i < macdLine.length - signalLine.length; i++) {
        fullSignalLine.push(null);
    }
    fullSignalLine.push(...signalLine);
    
    // Calculate histogram
    const histogram = [];
    for (let i = 0; i < macdLine.length; i++) {
        if (macdLine[i] === null || fullSignalLine[i] === null) {
            histogram.push(null);
        } else {
            histogram.push(macdLine[i] - fullSignalLine[i]);
        }
    }
    
    // Create new canvas for MACD
    const container = chart.canvas.parentNode;
    const macdCanvas = document.createElement('canvas');
    container.appendChild(macdCanvas);
    
    const macdChart = new Chart(macdCanvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: chart.data.labels,
            datasets: [
                {
                    label: 'MACD Histogram',
                    data: histogram,
                    backgroundColor: ctx => {
                        if (ctx.raw === null) return 'rgba(0, 0, 0, 0)';
                        return ctx.raw >= 0 ? 'rgba(46, 204, 113, 0.5)' : 'rgba(231, 76, 60, 0.5)';
                    },
                    borderColor: ctx => {
                        if (ctx.raw === null) return 'rgba(0, 0, 0, 0)';
                        return ctx.raw >= 0 ? 'rgba(46, 204, 113, 1)' : 'rgba(231, 76, 60, 1)';
                    },
                    borderWidth: 1,
                    type: 'bar'
                },
                {
                    label: 'MACD Line',
                    data: macdLine,
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    type: 'line'
                },
                {
                    label: 'Signal Line',
                    data: fullSignalLine,
                    borderColor: 'rgba(243, 156, 18, 1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    type: 'line'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true
                },
                y: {
                    display: true
                }
            }
        }
    });
    
    return macdChart;
}

// Add Bollinger Bands to chart
function addBollingerBands(chart, data, period = 20, stdDev = 2) {
    const closePrices = data.map(item => item.close);
    
    // Calculate SMA
    const sma = calculateSMA(closePrices, period);
    
    // Calculate Standard Deviation
    const upperBand = [];
    const lowerBand = [];
    
    for (let i = 0; i < closePrices.length; i++) {
        if (i < period - 1) {
            upperBand.push(null);
            lowerBand.push(null);
            continue;
        }
        
        // Calculate standard deviation
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += Math.pow(closePrices[i - j] - sma[i], 2);
        }
        const std = Math.sqrt(sum / period);
        
        // Calculate upper and lower bands
        upperBand.push(sma[i] + (stdDev * std));
        lowerBand.push(sma[i] - (stdDev * std));
    }
    
    // Add to chart
    chart.data.datasets.push(
        {
            label: 'Upper Band',
            data: upperBand,
            borderColor: 'rgba(52, 152, 219, 0.7)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            type: 'line'
        },
        {
            label: 'SMA ' + period,
            data: sma,
            borderColor: 'rgba(243, 156, 18, 0.7)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            type: 'line'
        },
        {
            label: 'Lower Band',
            data: lowerBand,
            borderColor: 'rgba(52, 152, 219, 0.7)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            type: 'line'
        }
    );
    
    chart.update();
}
