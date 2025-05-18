/* <script src="https://cdn.plot.ly/plotly-2.25.2.min.js" charset="utf-8"></script>
<script src="https://cdn.jsdelivr.net/npm/danfojs@1.2.0/lib/bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"> </script> */
// import * as dfd from "danfojs";
// import Plotly from 'plotly.js-dist-min';

const fileInput = document.getElementById('fileInput');
const openFileButton = document.getElementById('openFile');
const fileInfo = document.getElementById('fileInfo');
const errorInfo = document.getElementById('errorInfo');
const docIcon = document.getElementById('docIcon');
const fileSize = document.getElementById('fileSize');
const goToAnalyze = document.getElementById('goToAnalyze');

openFileButton.addEventListener('click', () => {
fileInput.click();
})

fileInput.addEventListener('change', (event) => {
const file = event.target.files[0];
if (file) {
    if (file.size > 200000000) {
    fileInfo.textContent = null;
    fileSize.textContent = null;
    docIcon.className = 'no_display';
    errorInfo.textContent = 'File is too large.';
    }
    else {
        //File selected that meets requirements
    errorInfo.textContent = null;
    docIcon.className = 'success';
    fileInfo.textContent = file.name;
    fileSize.textContent = niceBytes(file.size);

    df = csvToDf(file.name);
    printDfHead(df, "dataframe");
    
    goToAnalyze.className = "";
    }
}
else {
    fileInfo.textContent = null;
    fileSize.textContent = null;
    docIcon.className = 'no_display';
    errorInfo.textContent = 'No file selected.';
}
})

const units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'];
   
function niceBytes(x){

  let l = 0, n = parseInt(x, 10) || 0;

  while(n >= 1024 && ++l){
      n = n/1024;
  }
  
  return(n.toFixed(n < 10 && l > 0 ? 1 : 0) + ' ' + units[l]);
}

function corr(x, y) {
    let sumX = 0,
      sumY = 0,
      sumXY = 0,
      sumX2 = 0,
      sumY2 = 0;

    const minLength = x.length = y.length = Math.min(x.length, y.length),
      reduce = (xi, idx) => {
        const yi = y[idx];
        sumX += xi;
        sumY += yi;
        sumXY += xi * yi;
        sumX2 += xi * xi;
        sumY2 += yi * yi;
      }

  x.forEach(reduce);
  
  return (minLength * sumXY - sumX * sumY) /
  Math.sqrt((minLength * sumX2 - sumX * sumX) * (minLength * sumY2 - sumY * sumY));
  }

function csvToDf(file) {
    dfd.readCSV(file)
    .then(df => {
        return df;
    }).catch(err => {
    console.log(err);
    })
}

function printDfHead(df, htmlElement) {
    df.head().plot(htmlElement).table(); //Print the table to html
}

function corrPlot(df) {

    let zValues = [];
    let dfCopy = df.copy();
    let columnsLength = dfCopy.shape[1];
    let columnsToDrop = [];
    let numericColumns = dfCopy.selectDtypes([
        'int32',
        'float32',
    ]);

    // Drop columns with high cardinality (many unique values)
    for (let i = 0; i < columnsLength; i++) {
        let column = dfCopy.columns[i];

        // Skip if a numeric column as it will have lots of unique values
        // but this doesn't matter :)
        if (numericColumns.$columns.includes(column)) {
        continue;
        }

        let uniqueValuesCount = dfCopy.column(column).unique().$data.length;

        if (uniqueValuesCount > 5) {
        columnsToDrop.push(column);
        }
    }

    dfCopy.drop({ columns: columnsToDrop, inplace: true });

    // Create dummy columns for categoric variables
    let dummies = dfCopy.getDummies(dfCopy);
    // Uncomment to debug: console.log("DUMMIES", dummies);
    columnsLength = dummies.$columns.length;

    for (let i = 0; i < columnsLength; i++) {
        let column = dummies.$columns[i];
        // Uncomment to debug: console.log("COMPARING", column);
        let correlations = [];

        for (let j = 0; j < columnsLength; j++) {
        let comparisonColumn = dummies.$columns[j];
        // Uncomment to debug: console.log("TO", comparisonColumn);

        let pearsonCorrelation = corr(
            dummies[column].$data,
            dummies[comparisonColumn].$data
        ).toFixed(2)

        correlations.push(
            pearsonCorrelation
        );
        }

        zValues.push(correlations);
    }

    var xValues = dummies.$columns;
    var yValues = dummies.$columns;

    var colorscaleValue = [
        [0, '#3D9970'],
        [1, '#001f3f']
    ];

    var data = [{
        x: xValues,
        y: yValues,
        z: zValues,
        type: 'heatmap',
        colorscale: colorscaleValue,
        showscale: false
    }];

    var layout = {
        autosize: false,
        width: window.innerWidth - 650,
        height: 700,
        annotations: [],
        xaxis: {
        ticks: '',
        side: 'top'
        },
        yaxis: {
        ticks: '',
        ticksuffix: ' ',
        autosize: false
        }
    };

    for (var i = 0; i < yValues.length; i++) {
        for (var j = 0; j < xValues.length; j++) {
        var currentValue = zValues[i][j];
        if (currentValue != 0.0) {
            var textColor = 'white';
        } else {
            var textColor = 'black';
        }
        var result = {
            xref: 'x1',
            yref: 'y1',
            x: xValues[j],
            y: yValues[i],
            text: zValues[i][j],
            font: {
            family: 'Arial',
            size: 12,
            color: 'rgb(50, 171, 96)'
            },
            showarrow: false,
            font: {
            color: textColor
            }
        };
        layout.annotations.push(result);
        }
    }

    Plotly.newPlot('correlation-heatmap', data, layout);  //prints the correlation plot to html
}

function createModel (data) {
    const model = tf.sequential({
        layers: [
        tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
        tf.layers.dense({units: 10, activation: 'softmax'}),
        ]
    });

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Generate dummy data.
    const data2 = tf.randomNormal([100, 784]); //replace with data parameter passed into function
    const labels = tf.randomUniform([100, 10]);

    function onBatchEnd(batch, logs) {
        console.log('Accuracy', logs.acc);
    }

    // Train for 5 epochs with batch size of 32.
    model.fit(data2, labels, {
        epochs: 5,
        batchSize: 32,
        callbacks: {onBatchEnd}
    }).then(info => {
        console.log('Final accuracy', info.history.acc);
    });
}